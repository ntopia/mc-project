#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <CL/cl.h>

#define ReLU(x) (((x) > 0) ? (x) : 0)
#define N_CPU 4
#define N_GPU 16

typedef struct opencl_context {
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_command_queue cmd_queue;
    cl_mem buf_inputs, buf_outputs;
    cl_mem buf_filters, buf_biases;
    cl_mem buf[2];
    cl_kernel kernel_pooling, kernel_convolution[2];
} opencl_context;

inline void swap_cl_mem(opencl_context* ctx) {
    cl_mem tmp = ctx->buf[0];
    ctx->buf[0] = ctx->buf[1];
    ctx->buf[1] = tmp;
    cl_kernel k = ctx->kernel_convolution[0];
    ctx->kernel_convolution[0] = ctx->kernel_convolution[1];
    ctx->kernel_convolution[1] = k;
}

opencl_context gpu[N_GPU];

static void pooling_layer(int gpu_id, int N, int D) {
    clSetKernelArg(gpu[gpu_id].kernel_pooling, 0, sizeof(cl_mem), (void*)&(gpu[gpu_id].buf[0]));
    clSetKernelArg(gpu[gpu_id].kernel_pooling, 1, sizeof(cl_mem), (void*)&(gpu[gpu_id].buf[1]));
    clSetKernelArg(gpu[gpu_id].kernel_pooling, 2, sizeof(int), (void*)&N);
    clSetKernelArg(gpu[gpu_id].kernel_pooling, 3, sizeof(int), (void*)&D);

    size_t global_work_size[] = { D, N * N };
    size_t global_work_offset[] = { 0, 0 };
    size_t local_work_size[] = { 1, N };
    cl_event ev;
    clEnqueueNDRangeKernel(gpu[gpu_id].cmd_queue, gpu[gpu_id].kernel_pooling, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, &ev);
    clWaitForEvents(1, &ev);

    swap_cl_mem(&gpu[gpu_id]);
}

static void convolution_layer(int gpu_id, float* filters, float* biases, int N, int D1, int D2) {
    clEnqueueWriteBuffer(gpu[gpu_id].cmd_queue, gpu[gpu_id].buf_filters, CL_FALSE, 0, sizeof(float) * 3 * 3 * D1 * D2, (void*)filters, 0, NULL, NULL);
    clEnqueueWriteBuffer(gpu[gpu_id].cmd_queue, gpu[gpu_id].buf_biases, CL_FALSE, 0, sizeof(float) * D2, (void*)biases, 0, NULL, NULL);

    clSetKernelArg(gpu[gpu_id].kernel_convolution[0], 0, sizeof(cl_mem), (void*)&(gpu[gpu_id].buf[0]));
    clSetKernelArg(gpu[gpu_id].kernel_convolution[0], 1, sizeof(cl_mem), (void*)&(gpu[gpu_id].buf_filters));
    clSetKernelArg(gpu[gpu_id].kernel_convolution[0], 2, sizeof(cl_mem), (void*)&(gpu[gpu_id].buf_biases));
    clSetKernelArg(gpu[gpu_id].kernel_convolution[0], 3, sizeof(cl_mem), (void*)&(gpu[gpu_id].buf[1]));
    clSetKernelArg(gpu[gpu_id].kernel_convolution[0], 4, sizeof(int), (void*)&N);
    clSetKernelArg(gpu[gpu_id].kernel_convolution[0], 5, sizeof(int), (void*)&D1);
    clSetKernelArg(gpu[gpu_id].kernel_convolution[0], 6, sizeof(int), (void*)&D2);

    size_t global_work_size[] = { D2, N * N };
    size_t global_work_offset[] = { 0, 0 };
    size_t local_work_size[] = { 1, N };
    cl_event ev;
    clEnqueueNDRangeKernel(gpu[gpu_id].cmd_queue, gpu[gpu_id].kernel_convolution[0], 2, global_work_offset, global_work_size, local_work_size, 0, NULL, &ev);
    clWaitForEvents(1, &ev);

    swap_cl_mem(&gpu[gpu_id]);
}

static void fc_layer(int gpu_id, float* input_neuron, float* output_neuron, float* weights, float* biases, int N, int M) {
    for (int i = 0; i < M; ++i) {
        float sum = 0;
        for (int j = 0; j < N; ++j) {
            sum += input_neuron[j] * weights[i * N + j];
        }
        output_neuron[i] = ReLU(sum + biases[i]);
    }
}

static void softmax(float* output) {
    float max = output[0];
    for (int i = 1; i < 1000; ++i) {
        max = (output[i] > max) ? output[i] : max;
    }
    float sum = 0;
    for (int i = 0; i < 1000; ++i) {
        sum += exp(output[i] - max);
    }
    for (int i = 0; i < 1000; ++i) {
        output[i] = exp(output[i] - max) / sum;
    }
}

static int find_max(float* fc) {
    int maxid = 0;
    float maxval = 0;
    for (int i = 0; i < 1000; ++i) {
        if (maxval < fc[i]) {
            maxval = fc[i];
            maxid = i;
        }
    }
    return maxid;
}

static float* get_param(float** array, int size) {
    float* subarray = *array;
    *array += size;
    return subarray;
}

void read_kernel(const char* filename, char** dst) {
    FILE* infp = fopen(filename, "r");
    fseek(infp, -1, SEEK_END);
    size_t kernel_len = (size_t)ftell(infp);
    fseek(infp, 0, SEEK_SET);

    *dst = (char*)malloc(sizeof(char) * (kernel_len + 1));
    fread(*dst, sizeof(char), kernel_len, infp);
    (*dst)[kernel_len] = '\0';
    fclose(infp);
}

void printBuildFailure(opencl_context* ctx) {
    printf("failed to build program...\n");
    size_t loglen;
    char* logbuf;
    clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &loglen);
    logbuf = (char*)malloc(loglen);
    clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG, loglen, logbuf, NULL);
    printf("%s\n", logbuf);
} 

int init_opencl() {
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id tmp_gid[N_GPU];
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, N_GPU, tmp_gid, NULL) != 0) {
        printf("failed to initialize device...\n");
        return 1;
    }
    for (int k = 0; k < N_GPU; ++k) {
        gpu[k].device = tmp_gid[k];
        gpu[k].context = clCreateContext(NULL, 1, &gpu[k].device, NULL, NULL, NULL);
    }

    char* kernel_source_gpu;
    read_kernel("./kernel-gpu.cl", &kernel_source_gpu);
    for (int k = 0; k < N_GPU; ++k) {
        gpu[k].program = clCreateProgramWithSource(gpu[k].context, 1, (const char**)&kernel_source_gpu, NULL, NULL);
        if (clBuildProgram(gpu[k].program, 1, &gpu[k].device, NULL, NULL, NULL) != CL_SUCCESS) {
            printBuildFailure(&gpu[k]);
            return 1;
        }
        gpu[k].kernel_pooling = clCreateKernel(gpu[k].program, "pooling_layer", NULL);
        gpu[k].kernel_convolution[0] = clCreateKernel(gpu[k].program, "convolution_layer", NULL);
        gpu[k].kernel_convolution[1] = clCreateKernel(gpu[k].program, "convolution_layer", NULL);
    }

    for (int k = 0; k < N_GPU; ++k) {
        gpu[k].cmd_queue = clCreateCommandQueue(gpu[k].context, gpu[k].device, 0, NULL);
        gpu[k].buf[0] = clCreateBuffer(gpu[k].context, CL_MEM_READ_WRITE, sizeof(float) * 224 * 224 * 64, NULL, NULL);
        gpu[k].buf[1] = clCreateBuffer(gpu[k].context, CL_MEM_READ_WRITE, sizeof(float) * 224 * 224 * 64, NULL, NULL);
        gpu[k].buf_filters = clCreateBuffer(gpu[k].context, CL_MEM_READ_ONLY, sizeof(float) * 3 * 3 * 512 * 512, NULL, NULL);
        gpu[k].buf_biases = clCreateBuffer(gpu[k].context, CL_MEM_READ_ONLY, sizeof(float) * 512, NULL, NULL);
    }
    return 0;
}


typedef struct vggnet_thread_args {
    int images_st, images_ed;
    int thread_id;
} vggnet_thread_args;

float* g_images;
int* g_labels;
float* g_confidences;

// Filters & Weights
float *f1_1, *f1_2, *f2_1, *f2_2, *f3_1, *f3_2, *f3_3,
    *f4_1, *f4_2, *f4_3, *f5_1, *f5_2, *f5_3,
    *w1, *w2, *w3;
// Biases
float *b1_1, *b1_2, *b2_1, *b2_2, *b3_1, *b3_2, *b3_3,
    *b4_1, *b4_2, *b4_3, *b5_1, *b5_2, *b5_3, *b1, *b2, *b3;


void* vggnet_thread(void* arg) {
    vggnet_thread_args* targ = (vggnet_thread_args*)arg;
    int images_st = targ->images_st, images_ed = targ->images_ed;
    int thread_id = targ->thread_id;

    // Pooling layers
    float *p5;
    // Fully connected layers
    float *fc1, *fc2, *fc3;

    p5 = (float*)malloc(sizeof(float) * 7 * 7 * 512);

    fc1 = (float*)malloc(sizeof(float) * 4096);
    fc2 = (float*)malloc(sizeof(float) * 4096);
    fc3 = (float*)malloc(sizeof(float) * 1000);

    for (int i = images_st; i < images_ed; ++i) {
        float* image = g_images + i * 224 * 224 * 3;
        clEnqueueWriteBuffer(gpu[thread_id].cmd_queue, gpu[thread_id].buf[0], CL_TRUE, 0, sizeof(float) * 224 * 224 * 3, (void*)image, 0, NULL, NULL);

        convolution_layer(thread_id, f1_1, b1_1, 224, 3, 64);
        convolution_layer(thread_id, f1_2, b1_2, 224, 64, 64);
        pooling_layer(thread_id, 112, 64);

        convolution_layer(thread_id, f2_1, b2_1, 112, 64, 128);
        convolution_layer(thread_id, f2_2, b2_2, 112, 128, 128);
        pooling_layer(thread_id, 56, 128);

        convolution_layer(thread_id, f3_1, b3_1, 56, 128, 256);
        convolution_layer(thread_id, f3_2, b3_2, 56, 256, 256);
        convolution_layer(thread_id, f3_3, b3_3, 56, 256, 256);
        pooling_layer(thread_id, 28, 256);

        convolution_layer(thread_id, f4_1, b4_1, 28, 256, 512);
        convolution_layer(thread_id, f4_2, b4_2, 28, 512, 512);
        convolution_layer(thread_id, f4_3, b4_3, 28, 512, 512);
        pooling_layer(thread_id, 14, 512);

        convolution_layer(thread_id, f5_1, b5_1, 14, 512, 512);
        convolution_layer(thread_id, f5_2, b5_2, 14, 512, 512);
        convolution_layer(thread_id, f5_3, b5_3, 14, 512, 512);
        pooling_layer(thread_id, 7, 512);

        clEnqueueReadBuffer(gpu[thread_id].cmd_queue, gpu[thread_id].buf[0], CL_TRUE, 0, sizeof(float) * 7 * 7 * 512, (void*)p5, 0, NULL, NULL);

        fc_layer(thread_id, p5, fc1, w1, b1, 7 * 7 * 512, 4096);
        fc_layer(thread_id, fc1, fc2, w2, b2, 4096, 4096);
        fc_layer(thread_id, fc2, fc3, w3, b3, 4096, 1000);

        softmax(fc3);

        g_labels[i] = find_max(fc3);
        g_confidences[i] = fc3[g_labels[i]];
    }

    free(p5);

    free(fc1);
    free(fc2);
    free(fc3);

    return NULL;
}

void vggnet(float* images, float* network, int* labels, float* confidences, int num_images) {
    if (init_opencl() != 0) {
        return;
    }
    g_images = images;
    g_labels = labels;
    g_confidences = confidences;

    f1_1 = get_param(&network, 3 * 3 * 3 * 64);
    b1_1 = get_param(&network, 64);
    f1_2 = get_param(&network, 3 * 3 * 64 * 64);
    b1_2 = get_param(&network, 64);

    f2_1 = get_param(&network, 3 * 3 * 64 * 128);
    b2_1 = get_param(&network, 128);
    f2_2 = get_param(&network, 3 * 3 * 128 * 128);
    b2_2 = get_param(&network, 128);

    f3_1 = get_param(&network, 3 * 3 * 128 * 256);
    b3_1 = get_param(&network, 256);
    f3_2 = get_param(&network, 3 * 3 * 256 * 256);
    b3_2 = get_param(&network, 256);
    f3_3 = get_param(&network, 3 * 3 * 256 * 256);
    b3_3 = get_param(&network, 256);

    f4_1 = get_param(&network, 3 * 3 * 256 * 512);
    b4_1 = get_param(&network, 512);
    f4_2 = get_param(&network, 3 * 3 * 512 * 512);
    b4_2 = get_param(&network, 512);
    f4_3 = get_param(&network, 3 * 3 * 512 * 512);
    b4_3 = get_param(&network, 512);

    f5_1 = get_param(&network, 3 * 3 * 512 * 512);
    b5_1 = get_param(&network, 512);
    f5_2 = get_param(&network, 3 * 3 * 512 * 512);
    b5_2 = get_param(&network, 512);
    f5_3 = get_param(&network, 3 * 3 * 512 * 512);
    b5_3 = get_param(&network, 512);

    w1 = get_param(&network, 7 * 7 * 512 * 4096);
    b1 = get_param(&network, 4096);
    w2 = get_param(&network, 4096 * 4096);
    b2 = get_param(&network, 4096);
    w3 = get_param(&network, 4096 * 1000);
    b3 = get_param(&network, 1000);


    int elapsed = 0;
    pthread_t threads[N_GPU - 1];
    vggnet_thread_args arg[N_GPU];
    for (int k = 0; k < N_GPU; ++k) {
        int sz = (num_images / N_GPU) + (k < (num_images % N_GPU) ? 1 : 0);
        arg[k].images_st = elapsed;
        arg[k].images_ed = elapsed + sz;
        arg[k].thread_id = k;
        if (k < N_GPU - 1) {
            pthread_create(&threads[k], NULL, &vggnet_thread, &arg[k]);
        }
        else {
            vggnet_thread((void*)&arg[N_GPU - 1]);
        }
        elapsed += sz;
    }
    for (int k = 0; k < N_GPU - 1; ++k) {
        pthread_join(threads[k], NULL);
    }
}
