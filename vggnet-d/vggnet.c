#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <CL/cl.h>
#include <mpi.h>

#define ReLU(x) (((x) > 0) ? (x) : 0)

typedef struct opencl_context {
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_command_queue cmd_queue;
    cl_mem buf_outputs;
    cl_kernel kernel_pooling, kernel_convolution, kernel_fc;
} opencl_context;

opencl_context cpu, gpu[4];

static void pooling_layer(int gpu_id, float* inputs, float* outputs, int N, int D) {
    cl_mem buf_inputs = clCreateBuffer(gpu[gpu_id].context, CL_MEM_USE_HOST_PTR, sizeof(float) * D * N * N * 4, (void*)inputs, NULL);

    clSetKernelArg(gpu[gpu_id].kernel_pooling, 0, sizeof(cl_mem), (void*)&buf_inputs);
    clSetKernelArg(gpu[gpu_id].kernel_pooling, 1, sizeof(cl_mem), (void*)&(gpu[gpu_id].buf_outputs));
    clSetKernelArg(gpu[gpu_id].kernel_pooling, 2, sizeof(int), (void*)&N);
    clSetKernelArg(gpu[gpu_id].kernel_pooling, 3, sizeof(int), (void*)&D);

    size_t global_work_size[] = { D, N * N };
    size_t global_work_offset[] = { 0, 0 };
    size_t local_work_size[] = { 1, N };
    clEnqueueNDRangeKernel(gpu[gpu_id].cmd_queue, gpu[gpu_id].kernel_pooling, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);
    clEnqueueReadBuffer(gpu[gpu_id].cmd_queue, gpu[gpu_id].buf_outputs, CL_TRUE, 0, sizeof(float) * D * N * N, (void*)outputs, 0, NULL, NULL);

    clReleaseMemObject(buf_inputs);
}

static void convolution_layer(int gpu_id, float* inputs, float* outputs, float* filters, float* biases, int N, int D1, int D2) {
    cl_mem buf_inputs = clCreateBuffer(gpu[gpu_id].context, CL_MEM_USE_HOST_PTR, sizeof(float) * N * N * D1, (void*)inputs, NULL);
    cl_mem buf_filters = clCreateBuffer(gpu[gpu_id].context, CL_MEM_USE_HOST_PTR, sizeof(float) * 3 * 3 * D1 * D2, (void*)filters, NULL);
    cl_mem buf_biases = clCreateBuffer(gpu[gpu_id].context, CL_MEM_USE_HOST_PTR, sizeof(float) * D2, (void*)biases, NULL);

    clSetKernelArg(gpu[gpu_id].kernel_convolution, 0, sizeof(cl_mem), (void*)&buf_inputs);
    clSetKernelArg(gpu[gpu_id].kernel_convolution, 1, sizeof(cl_mem), (void*)&buf_filters);
    clSetKernelArg(gpu[gpu_id].kernel_convolution, 2, sizeof(cl_mem), (void*)&buf_biases);
    clSetKernelArg(gpu[gpu_id].kernel_convolution, 3, sizeof(cl_mem), (void*)&(gpu[gpu_id].buf_outputs));
    clSetKernelArg(gpu[gpu_id].kernel_convolution, 4, sizeof(int), (void*)&N);
    clSetKernelArg(gpu[gpu_id].kernel_convolution, 5, sizeof(int), (void*)&D1);
    clSetKernelArg(gpu[gpu_id].kernel_convolution, 6, sizeof(int), (void*)&D2);

    size_t global_work_size[] = { D2, N * N };
    size_t global_work_offset[] = { 0, 0 };
    size_t local_work_size[] = { 1, N };
    clEnqueueNDRangeKernel(gpu[gpu_id].cmd_queue, gpu[gpu_id].kernel_convolution, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);
    clEnqueueReadBuffer(gpu[gpu_id].cmd_queue, gpu[gpu_id].buf_outputs, CL_TRUE, 0, sizeof(float) * N * N * D2, (void*)outputs, 0, NULL, NULL);

    clReleaseMemObject(buf_inputs);
    clReleaseMemObject(buf_filters);
    clReleaseMemObject(buf_biases);
}

static void fc_layer(int gpu_id, float* input_neuron, float* output_neuron, float* weights, float* biases, int N, int M) {
    cl_mem buf_inputs = clCreateBuffer(cpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * N, (void*)input_neuron, NULL);
    cl_mem buf_weights = clCreateBuffer(cpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * N * M, (void*)weights, NULL);
    cl_mem buf_biases = clCreateBuffer(cpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * M, (void*)biases, NULL);
    cl_mem buf_outputs = clCreateBuffer(cpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * M, (void*)output_neuron, NULL);

    clSetKernelArg(cpu.kernel_fc, 0, sizeof(cl_mem), (void*)&buf_inputs);
    clSetKernelArg(cpu.kernel_fc, 1, sizeof(cl_mem), (void*)&buf_weights);
    clSetKernelArg(cpu.kernel_fc, 2, sizeof(cl_mem), (void*)&buf_biases);
    clSetKernelArg(cpu.kernel_fc, 3, sizeof(cl_mem), (void*)&buf_outputs);
    clSetKernelArg(cpu.kernel_fc, 4, sizeof(int), (void*)&N);
    clSetKernelArg(cpu.kernel_fc, 5, sizeof(int), (void*)&M);

    size_t global_work_size = M;
    size_t global_work_offset = 0;
    size_t local_work_size = M / 8;
    cl_event event;
    clEnqueueNDRangeKernel(cpu.cmd_queue, cpu.kernel_fc, 1, &global_work_offset, &global_work_size, &local_work_size, 0, NULL, &event);
    clWaitForEvents(1, &event);

    clReleaseMemObject(buf_inputs);
    clReleaseMemObject(buf_weights);
    clReleaseMemObject(buf_biases);
    clReleaseMemObject(buf_outputs);
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
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &cpu.device, NULL) != 0) {
        printf("failed to initialize device...\n");
        return 1;
    }
    cl_device_id tmp_gid[4];
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 4, tmp_gid, NULL) != 0) {
        printf("failed to initialize device...\n");
        return 1;
    }
    cpu.context = clCreateContext(NULL, 1, &cpu.device, NULL, NULL, NULL);
    for (int k = 0; k < 4; ++k) {
        gpu[k].device = tmp_gid[k];
        gpu[k].context = clCreateContext(NULL, 1, &gpu[k].device, NULL, NULL, NULL);
    }

    char* kernel_source_cpu;
    read_kernel("./kernel-cpu.cl", &kernel_source_cpu);
    cpu.program = clCreateProgramWithSource(cpu.context, 1, (const char**)&kernel_source_cpu, NULL, NULL);
    if (clBuildProgram(cpu.program, 1, &cpu.device, "-cl-denorms-are-zero -cl-fast-relaxed-math", NULL, NULL) != CL_SUCCESS) {
        printBuildFailure(&cpu);
        return 1;
    }
    cpu.kernel_pooling = clCreateKernel(cpu.program, "pooling_layer", NULL);
    cpu.kernel_convolution = clCreateKernel(cpu.program, "convolution_layer", NULL);
    cpu.kernel_fc = clCreateKernel(cpu.program, "fc_layer", NULL);

    char* kernel_source_gpu;
    read_kernel("./kernel-gpu.cl", &kernel_source_gpu);
    for (int k = 0; k < 4; ++k) {
        gpu[k].program = clCreateProgramWithSource(gpu[k].context, 1, (const char**)&kernel_source_gpu, NULL, NULL);
        if (clBuildProgram(gpu[k].program, 1, &gpu[k].device, "-cl-denorms-are-zero -cl-fast-relaxed-math", NULL, NULL) != CL_SUCCESS) {
            printBuildFailure(&gpu[k]);
            return 1;
        }
        gpu[k].kernel_pooling = clCreateKernel(gpu[k].program, "pooling_layer", NULL);
        gpu[k].kernel_convolution = clCreateKernel(gpu[k].program, "convolution_layer", NULL);
        gpu[k].kernel_fc = clCreateKernel(gpu[k].program, "fc_layer", NULL);
    }

    cpu.cmd_queue = clCreateCommandQueue(cpu.context, cpu.device, 0, NULL);
    for (int k = 0; k < 4; ++k) {
        gpu[k].cmd_queue = clCreateCommandQueue(gpu[k].context, gpu[k].device, 0, NULL);
        gpu[k].buf_outputs = clCreateBuffer(gpu[k].context, CL_MEM_READ_WRITE, sizeof(float) * 224 * 224 * 64, NULL, NULL);
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

    // Convolution layers
    float *c1_1, *c1_2, *c2_1, *c2_2, *c3_1, *c3_2, *c3_3,
        *c4_1, *c4_2, *c4_3, *c5_1, *c5_2, *c5_3;
    // Pooling layers
    float *p1, *p2, *p3, *p4, *p5;
    // Fully connected layers
    float *fc1, *fc2, *fc3;

    c1_1 = (float*)malloc(sizeof(float) * 224 * 224 * 64);
    c1_2 = (float*)malloc(sizeof(float) * 224 * 224 * 64);

    p1 = (float*)malloc(sizeof(float) * 112 * 112 * 64);

    c2_1 = (float*)malloc(sizeof(float) * 112 * 112 * 128);
    c2_2 = (float*)malloc(sizeof(float) * 112 * 112 * 128);

    p2 = (float*)malloc(sizeof(float) * 56 * 56 * 128);

    c3_1 = (float*)malloc(sizeof(float) * 56 * 56 * 256);
    c3_2 = (float*)malloc(sizeof(float) * 56 * 56 * 256);
    c3_3 = (float*)malloc(sizeof(float) * 56 * 56 * 256);

    p3 = (float*)malloc(sizeof(float) * 28 * 28 * 256);

    c4_1 = (float*)malloc(sizeof(float) * 28 * 28 * 512);
    c4_2 = (float*)malloc(sizeof(float) * 28 * 28 * 512);
    c4_3 = (float*)malloc(sizeof(float) * 28 * 28 * 512);

    p4 = (float*)malloc(sizeof(float) * 14 * 14 * 512);

    c5_1 = (float*)malloc(sizeof(float) * 14 * 14 * 512);
    c5_2 = (float*)malloc(sizeof(float) * 14 * 14 * 512);
    c5_3 = (float*)malloc(sizeof(float) * 14 * 14 * 512);

    p5 = (float*)malloc(sizeof(float) * 7 * 7 * 512);

    fc1 = (float*)malloc(sizeof(float) * 4096);
    fc2 = (float*)malloc(sizeof(float) * 4096);
    fc3 = (float*)malloc(sizeof(float) * 1000);

    for (int i = images_st; i < images_ed; ++i) {
        float* image = g_images + i * 224 * 224 * 3;

        convolution_layer(thread_id, image, c1_1, f1_1, b1_1, 224, 3, 64);
        convolution_layer(thread_id, c1_1, c1_2, f1_2, b1_2, 224, 64, 64);
        pooling_layer(thread_id, c1_2, p1, 112, 64);

        convolution_layer(thread_id, p1, c2_1, f2_1, b2_1, 112, 64, 128);
        convolution_layer(thread_id, c2_1, c2_2, f2_2, b2_2, 112, 128, 128);
        pooling_layer(thread_id, c2_2, p2, 56, 128);

        convolution_layer(thread_id, p2, c3_1, f3_1, b3_1, 56, 128, 256);
        convolution_layer(thread_id, c3_1, c3_2, f3_2, b3_2, 56, 256, 256);
        convolution_layer(thread_id, c3_2, c3_3, f3_3, b3_3, 56, 256, 256);
        pooling_layer(thread_id, c3_3, p3, 28, 256);

        convolution_layer(thread_id, p3, c4_1, f4_1, b4_1, 28, 256, 512);
        convolution_layer(thread_id, c4_1, c4_2, f4_2, b4_2, 28, 512, 512);
        convolution_layer(thread_id, c4_2, c4_3, f4_3, b4_3, 28, 512, 512);
        pooling_layer(thread_id, c4_3, p4, 14, 512);

        convolution_layer(thread_id, p4, c5_1, f5_1, b5_1, 14, 512, 512);
        convolution_layer(thread_id, c5_1, c5_2, f5_2, b5_2, 14, 512, 512);
        convolution_layer(thread_id, c5_2, c5_3, f5_3, b5_3, 14, 512, 512);
        pooling_layer(thread_id, c5_3, p5, 7, 512);

        fc_layer(thread_id, p5, fc1, w1, b1, 7 * 7 * 512, 4096);
        fc_layer(thread_id, fc1, fc2, w2, b2, 4096, 4096);
        fc_layer(thread_id, fc2, fc3, w3, b3, 4096, 1000);

        softmax(fc3);

        g_labels[i] = find_max(fc3);
        g_confidences[i] = fc3[g_labels[i]];
    }

    free(c1_1);
    free(c1_2);
    free(p1);

    free(c2_1);
    free(c2_2);
    free(p2);

    free(c3_1);
    free(c3_2);
    free(c3_3);
    free(p3);

    free(c4_1);
    free(c4_2);
    free(c4_3);
    free(p4);

    free(c5_1);
    free(c5_2);
    free(c5_3);
    free(p5);

    free(fc1);
    free(fc2);
    free(fc3);

    return NULL;
}

void vggnet(float* images, float* network, int* labels, float* confidences, int images_st, int images_ed, int num_all_images, int task_id) {
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


    int num_images = images_ed - images_st;
    int elapsed = images_st;
    pthread_t threads[3];
    vggnet_thread_args arg[4];
    for (int k = 0; k < 4; ++k) {
        int sz = (num_images / 4) + (k < (num_images % 4) ? 1 : 0);
        arg[k].images_st = elapsed;
        arg[k].images_ed = elapsed + sz;
        arg[k].thread_id = k;
        if (k < 3) {
            pthread_create(&threads[k], NULL, &vggnet_thread, &arg[k]);
        }
        else {
            vggnet_thread((void*)&arg[3]);
        }
        elapsed += sz;
    }
    for (int k = 0; k < 3; ++k) {
        pthread_join(threads[k], NULL);
    }

    int st[4], cnt[4];
    for (int k = 0; k < 4; ++k) {
        cnt[k] = (num_all_images / 4) + (k < (num_all_images % 4) ? 1 : 0);
        st[k] = (k == 0) ? 0 : st[k - 1] + cnt[k - 1];
    }

    MPI_Request req[6];
    if (task_id == 0) {
        for (int k = 1; k < 4; ++k) {
            MPI_Irecv(labels + st[k], cnt[k], MPI_INT, k, 11, MPI_COMM_WORLD, &req[(k - 1) * 2]);
            MPI_Irecv(confidences + st[k], cnt[k], MPI_FLOAT, k, 11, MPI_COMM_WORLD, &req[(k - 1) * 2 + 1]);
        }
        MPI_Waitall(6, req, MPI_STATUSES_IGNORE);

        for (int k = 1; k < 4; ++k) {
            MPI_Isend(labels, num_all_images, MPI_INT, k, 22, MPI_COMM_WORLD, &req[(k - 1) * 2]);
            MPI_Isend(confidences, num_all_images, MPI_FLOAT, k, 22, MPI_COMM_WORLD, &req[(k - 1) * 2 + 1]);
        }
        MPI_Waitall(6, req, MPI_STATUSES_IGNORE);
    }
    else {
        MPI_Isend(labels + images_st, num_images, MPI_INT, 0, 11, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(confidences + images_st, num_images, MPI_FLOAT, 0, 11, MPI_COMM_WORLD, &req[1]);
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

        MPI_Irecv(labels, num_all_images, MPI_INT, 0, 22, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(confidences, num_all_images, MPI_FLOAT, 0, 22, MPI_COMM_WORLD, &req[1]);
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    }
}
