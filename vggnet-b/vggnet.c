#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <CL/cl.h>

#define ReLU(x) (((x) > 0) ? (x) : 0)

typedef struct opencl_context {
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_command_queue cmd_queue;
    cl_mem buf[2];
} opencl_context;

inline void swap_cl_mem(opencl_context* ctx) {
    cl_mem tmp = ctx->buf[0];
    ctx->buf[0] = ctx->buf[1];
    ctx->buf[1] = tmp;
}

opencl_context gpu;

static void pooling_layer(int N, int D) {
    cl_kernel kernel = clCreateKernel(gpu.program, "pooling_layer", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&(gpu.buf[0]));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&(gpu.buf[1]));
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&D);

    size_t global_work_size[] = { D, N * N };
    size_t global_work_offset[] = { 0, 0 };
    size_t local_work_size[] = { 1, N };
    clEnqueueNDRangeKernel(gpu.cmd_queue, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);

    clReleaseKernel(kernel);
    swap_cl_mem(&gpu);
}

static void convolution_1row_0_layer(float* filters, float* biases, int N, int D1, int D2) {
    cl_mem buf_filters = clCreateBuffer(gpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * 3 * 3 * D1 * D2, (void*)filters, NULL);
    cl_mem buf_biases = clCreateBuffer(gpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * D2, (void*)biases, NULL);

    cl_kernel kernel = clCreateKernel(gpu.program, "convolution_1row_0_layer", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&(gpu.buf[0]));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_filters);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buf_biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&(gpu.buf[1]));
    clSetKernelArg(kernel, 4, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 5, sizeof(int), (void*)&D1);

    size_t global_work_size[] = { D2, N * N };
    size_t global_work_offset[] = { 0, 0 };
    size_t local_work_size[] = { 1, N };
    clEnqueueNDRangeKernel(gpu.cmd_queue, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);

    clReleaseMemObject(buf_filters);
    clReleaseMemObject(buf_biases);
    clReleaseKernel(kernel);
    swap_cl_mem(&gpu);
}

static void convolution_1row_1_layer(float* filters, float* biases, int N, int D1, int D2) {
    cl_mem buf_filters = clCreateBuffer(gpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * 3 * 3 * D1 * D2, (void*)filters, NULL);
    cl_mem buf_biases = clCreateBuffer(gpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * D2, (void*)biases, NULL);

    cl_kernel kernel = clCreateKernel(gpu.program, "convolution_1row_1_layer", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&(gpu.buf[0]));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_filters);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buf_biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&(gpu.buf[1]));
    clSetKernelArg(kernel, 4, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 5, sizeof(int), (void*)&D1);

    size_t global_work_size[] = { D2, N * N };
    size_t global_work_offset[] = { 0, 0 };
    size_t local_work_size[] = { 1, N };
    clEnqueueNDRangeKernel(gpu.cmd_queue, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);

    clReleaseMemObject(buf_filters);
    clReleaseMemObject(buf_biases);
    clReleaseKernel(kernel);
    swap_cl_mem(&gpu);
}

static void convolution_2row_layer(float* filters, float* biases, int N, int D1, int D2) {
    cl_mem buf_filters = clCreateBuffer(gpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * 3 * 3 * D1 * D2, (void*)filters, NULL);
    cl_mem buf_biases = clCreateBuffer(gpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * D2, (void*)biases, NULL);

    cl_kernel kernel = clCreateKernel(gpu.program, "convolution_2row_layer", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&(gpu.buf[0]));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_filters);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buf_biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&(gpu.buf[1]));
    clSetKernelArg(kernel, 4, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 5, sizeof(int), (void*)&D1);

    size_t global_work_size[] = { D2, N * N };
    size_t global_work_offset[] = { 0, 0 };
    size_t local_work_size[] = { 1, N * 2 };
    clEnqueueNDRangeKernel(gpu.cmd_queue, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);

    clReleaseMemObject(buf_filters);
    clReleaseMemObject(buf_biases);
    clReleaseKernel(kernel);
    swap_cl_mem(&gpu);
}

static void convolution_4row_layer(float* filters, float* biases, int N, int D1, int D2) {
    cl_mem buf_filters = clCreateBuffer(gpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * 3 * 3 * D1 * D2, (void*)filters, NULL);
    cl_mem buf_biases = clCreateBuffer(gpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * D2, (void*)biases, NULL);

    cl_kernel kernel = clCreateKernel(gpu.program, "convolution_4row_layer", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&(gpu.buf[0]));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_filters);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buf_biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&(gpu.buf[1]));
    clSetKernelArg(kernel, 4, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 5, sizeof(int), (void*)&D1);

    size_t global_work_size[] = { N * N, D2 };
    size_t global_work_offset[] = { 0, 0 };
    size_t local_work_size[] = { N, 4 };
    clEnqueueNDRangeKernel(gpu.cmd_queue, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);

    clReleaseMemObject(buf_filters);
    clReleaseMemObject(buf_biases);
    clReleaseKernel(kernel);
    swap_cl_mem(&gpu);
}

static void convolution_break_layer(float* filters, float* biases, int N, int D1, int D2) {
    cl_mem buf_filters = clCreateBuffer(gpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * 3 * 3 * D1 * D2, (void*)filters, NULL);
    cl_mem buf_biases = clCreateBuffer(gpu.context, CL_MEM_USE_HOST_PTR, sizeof(float) * D2, (void*)biases, NULL);

    cl_kernel kernel = clCreateKernel(gpu.program, "convolution_break_layer", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&(gpu.buf[0]));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_filters);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buf_biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&(gpu.buf[1]));
    clSetKernelArg(kernel, 4, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 5, sizeof(int), (void*)&D1);

    size_t global_work_size[] = { D2 * N * N };
    size_t global_work_offset[] = { 0 };
    size_t local_work_size[] = { 256 };
    clEnqueueNDRangeKernel(gpu.cmd_queue, kernel, 1, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);

    clReleaseMemObject(buf_filters);
    clReleaseMemObject(buf_biases);
    clReleaseKernel(kernel);
    swap_cl_mem(&gpu);
}

static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int N, int M) {
    for (int i = 0; i < M; i += 2) {
        float sum0 = 0, sum1 = 0;
        for (int j = 0; j < N; ++j) {
            sum0 += input_neuron[j] * weights[i * N + j];
            sum1 += input_neuron[j] * weights[(i + 1) * N + j];
        }
        sum0 += biases[i];
        sum1 += biases[i + 1];
        output_neuron[i] = ReLU(sum0);
        output_neuron[i + 1] = ReLU(sum1);
    }
}

static void softmax(float* output) {
    float max = output[0];
    for (int i = 1; i < 1000; ++i) {
        if (max < output[i]) max = output[i];
    }
    float sum = 0;
    for (int i = 0; i < 1000; ++i) {
        output[i] = exp(output[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < 1000; ++i) {
        output[i] /= sum;
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

void read_kernel(const char* filename, char** dst, size_t* len) {
    FILE* infp = fopen(filename, "r");
    fseek(infp, 0, SEEK_END);
    *len = (size_t)ftell(infp);
    fseek(infp, 0, SEEK_SET);

    *dst = (char*)malloc(sizeof(char) * (*len));
    fread(*dst, sizeof(char), *len, infp);
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

#define KERNEL_COMPILE_OPTION "-cl-denorms-are-zero -cl-strict-aliasing -cl-no-signed-zeros -cl-fast-relaxed-math"
char* kernel_binary_gpu;
size_t kernel_binary_len;

int init_opencl() {
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpu.device, NULL) != 0) {
        printf("failed to initialize device...\n");
        return 1;
    }
    gpu.context = clCreateContext(NULL, 1, &gpu.device, NULL, NULL, NULL);
    read_kernel("./kernel-gpu.cl.bin", &kernel_binary_gpu, &kernel_binary_len);
    return 0;
}

// Filters & Weights
float *f1_1, *f1_2, *f2_1, *f2_2, *f3_1, *f3_2, *f3_3,
    *f4_1, *f4_2, *f4_3, *f5_1, *f5_2, *f5_3,
    *w1, *w2, *w3;
// Biases
float *b1_1, *b1_2, *b2_1, *b2_2, *b3_1, *b3_2, *b3_3,
    *b4_1, *b4_2, *b4_3, *b5_1, *b5_2, *b5_3, *b1, *b2, *b3;

int* g_labels;
float* g_confidences;
float stage[1040][7 * 7 * 512];

typedef struct fc_layer_thread_args {
    int id;
} fc_layer_thread_args;

void* fc_layer_thread(void* varg) {
    fc_layer_thread_args* arg = (fc_layer_thread_args*)varg;
    int id = arg->id;

    // Fully connected layers
    float *fc1, *fc2, *fc3;
    posix_memalign((void**)&fc1, 256, sizeof(float) * 4096);
    posix_memalign((void**)&fc2, 256, sizeof(float) * 4096);
    posix_memalign((void**)&fc3, 256, sizeof(float) * 1024);
 
    fc_layer(stage[id], fc1, w1, b1, 7 * 7 * 512, 4096);
    fc_layer(fc1, fc2, w2, b2, 4096, 4096);
    fc_layer(fc2, fc3, w3, b3, 4096, 1000);

    softmax(fc3);

    g_labels[id] = find_max(fc3);
    g_confidences[id] = fc3[g_labels[id]];

    free(fc1);
    free(fc2);
    free(fc3);
    return NULL;
}

pthread_t fc_threads[1040];
fc_layer_thread_args fc_arg[1040];

void vggnet(float* images, float* network, int* labels, float* confidences, int num_images) {
    if (init_opencl() != 0) {
        return;
    }
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

    gpu.cmd_queue = clCreateCommandQueue(gpu.context, gpu.device, 0, NULL);
    gpu.buf[0] = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, sizeof(float) * 224 * 224 * 64, NULL, NULL);
    gpu.buf[1] = clCreateBuffer(gpu.context, CL_MEM_READ_WRITE, sizeof(float) * 224 * 224 * 64, NULL, NULL);

    gpu.program = clCreateProgramWithBinary(gpu.context, 1, &gpu.device, &kernel_binary_len, (const unsigned char**)&kernel_binary_gpu, NULL, NULL);
    if (clBuildProgram(gpu.program, 1, &gpu.device, KERNEL_COMPILE_OPTION, NULL, NULL) != CL_SUCCESS) {
        printBuildFailure(&gpu);
        return;
    }

    for (int i = 0; i < num_images; ++i) {
        float* image = images + i * 224 * 224 * 3;
        clEnqueueWriteBuffer(gpu.cmd_queue, gpu.buf[0], CL_FALSE, 0, sizeof(float) * 224 * 224 * 3, (void*)image, 0, NULL, NULL);

        convolution_1row_0_layer(f1_1, b1_1, 224, 3, 64);
        convolution_1row_0_layer(f1_2, b1_2, 224, 64, 64);
        pooling_layer(112, 64);

        convolution_1row_1_layer(f2_1, b2_1, 112, 64, 128);
        convolution_1row_1_layer(f2_2, b2_2, 112, 128, 128);
        pooling_layer(56, 128);

        convolution_4row_layer(f3_1, b3_1, 56, 128, 256);
        convolution_4row_layer(f3_2, b3_2, 56, 256, 256);
        convolution_4row_layer(f3_3, b3_3, 56, 256, 256);
        pooling_layer(28, 256);

        convolution_2row_layer(f4_1, b4_1, 28, 256, 512);
        convolution_2row_layer(f4_2, b4_2, 28, 512, 512);
        convolution_2row_layer(f4_3, b4_3, 28, 512, 512);
        pooling_layer(14, 512);

        convolution_break_layer(f5_1, b5_1, 14, 512, 512);
        convolution_break_layer(f5_2, b5_2, 14, 512, 512);
        convolution_break_layer(f5_3, b5_3, 14, 512, 512);
        pooling_layer(7, 512);

        clEnqueueReadBuffer(gpu.cmd_queue, gpu.buf[0], CL_TRUE, 0, sizeof(float) * 7 * 7 * 512, (void*)stage[i], 0, NULL, NULL);

        fc_arg[i].id = i;
        pthread_create(&fc_threads[i], NULL, &fc_layer_thread, (void*)&fc_arg[i]);
    }

    for (int i = 0; i < num_images; ++i) {
        pthread_join(fc_threads[i], NULL);
    }
}
