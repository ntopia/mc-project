#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define ReLU(x) (((x) > 0) ? (x) : 0)

cl_device_id device_cpu, device_gpu;
cl_context context_cpu, context_gpu;
cl_program program_cpu, program_gpu;
cl_command_queue cmd_queue_cpu, cmd_queue_gpu;
cl_mem buf_gpu_outputs;

static void pooling_layer(float* inputs, float* outputs, int N, int D) {
    cl_mem buf_input = clCreateBuffer(context_gpu, CL_MEM_USE_HOST_PTR, sizeof(float) * D * N * N * 4, (void*)inputs, NULL);

    cl_kernel kernel = clCreateKernel(program_gpu, "pooling_layer", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_gpu_outputs);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&D);

    size_t global_work_size[] = { D, N * N };
    size_t global_work_offset[] = { 0, 0 };
    size_t local_work_size[] = { 1, N };
    clEnqueueNDRangeKernel(cmd_queue_gpu, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);

    clEnqueueReadBuffer(cmd_queue_gpu, buf_gpu_outputs, CL_TRUE, 0, sizeof(float) * D * N * N, (void*)outputs, 0, NULL, NULL);
    clReleaseMemObject(buf_input);
}

static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int N, int D1, int D2) {
    cl_mem buf_inputs = clCreateBuffer(context_gpu, CL_MEM_USE_HOST_PTR, sizeof(float) * N * N * D1, (void*)inputs, NULL);
    cl_mem buf_filters = clCreateBuffer(context_gpu, CL_MEM_USE_HOST_PTR, sizeof(float) * 3 * 3 * D1 * D2, (void*)filters, NULL);
    cl_mem buf_biases = clCreateBuffer(context_gpu, CL_MEM_USE_HOST_PTR, sizeof(float) * D2, (void*)biases, NULL);

    cl_kernel kernel = clCreateKernel(program_gpu, "convolution_layer", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_inputs);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_filters);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buf_biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buf_gpu_outputs);
    clSetKernelArg(kernel, 4, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 5, sizeof(int), (void*)&D1);
    clSetKernelArg(kernel, 6, sizeof(int), (void*)&D2);

    size_t global_work_size[] = { D2, N * N };
    size_t global_work_offset[] = { 0, 0 };
    size_t local_work_size[] = { 1, N };
    cl_event event;
    clEnqueueNDRangeKernel(cmd_queue_gpu, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, NULL, &event);

    clEnqueueReadBuffer(cmd_queue_gpu, buf_gpu_outputs, CL_TRUE, 0, sizeof(float) * N * N * D2, (void*)outputs, 0, NULL, NULL);
    clReleaseMemObject(buf_inputs);
    clReleaseMemObject(buf_filters);
    clReleaseMemObject(buf_biases);
}

static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int N, int M) {
    cl_mem buf_input = clCreateBuffer(context_cpu, CL_MEM_USE_HOST_PTR, sizeof(float) * N, (void*)input_neuron, NULL);
    cl_mem buf_weights = clCreateBuffer(context_cpu, CL_MEM_USE_HOST_PTR, sizeof(float) * N * M, (void*)weights, NULL);
    cl_mem buf_biases = clCreateBuffer(context_cpu, CL_MEM_USE_HOST_PTR, sizeof(float) * M, (void*)biases, NULL);
    cl_mem buf_output = clCreateBuffer(context_cpu, CL_MEM_USE_HOST_PTR, sizeof(float) * M, (void*)output_neuron, NULL);

    cl_kernel kernel = clCreateKernel(program_cpu, "fc_layer", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buf_biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buf_output);
    clSetKernelArg(kernel, 4, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 5, sizeof(int), (void*)&M);

    size_t global_work_size = M;
    size_t global_work_offset = 0;
    size_t local_work_size = M / 8;
    cl_event event;
    clEnqueueNDRangeKernel(cmd_queue_cpu, kernel, 1, &global_work_offset, &global_work_size, &local_work_size, 0, NULL, &event);
    clWaitForEvents(1, &event);
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

int init_opencl() {
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device_cpu, NULL) != 0) {
        printf("failed to initialize device...\n");
        return 1;
    }
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_gpu, NULL) != 0) {
        printf("failed to initialize device...\n");
        return 1;
    }
    context_cpu = clCreateContext(NULL, 1, &device_cpu, NULL, NULL, NULL);
    context_gpu = clCreateContext(NULL, 1, &device_gpu, NULL, NULL, NULL);

    char* kernel_source_cpu;
    char* kernel_source_gpu;
    read_kernel("./kernel-cpu.cl", &kernel_source_cpu);
    read_kernel("./kernel-gpu.cl", &kernel_source_gpu);
    program_cpu = clCreateProgramWithSource(context_cpu, 1, (const char**)&kernel_source_cpu, NULL, NULL);
    if (clBuildProgram(program_cpu, 1, &device_cpu, "-cl-denorms-are-zero -cl-fast-relaxed-math", NULL, NULL) != CL_SUCCESS) {
        printf("failed to build program...\n");
        size_t loglen;
        char* logbuf;
        clGetProgramBuildInfo(program_cpu, device_cpu, CL_PROGRAM_BUILD_LOG, 0, NULL, &loglen);
        logbuf = (char*)malloc(loglen);
        clGetProgramBuildInfo(program_cpu, device_cpu, CL_PROGRAM_BUILD_LOG, loglen, logbuf, NULL);
        printf("%s\n", logbuf);
        return 1;
    }
    program_gpu = clCreateProgramWithSource(context_gpu, 1, (const char**)&kernel_source_gpu, NULL, NULL);
    if (clBuildProgram(program_gpu, 1, &device_gpu, "-cl-denorms-are-zero -cl-fast-relaxed-math", NULL, NULL) != CL_SUCCESS) {
        printf("failed to build program...\n");
        size_t loglen;
        char* logbuf;
        clGetProgramBuildInfo(program_gpu, device_gpu, CL_PROGRAM_BUILD_LOG, 0, NULL, &loglen);
        logbuf = (char*)malloc(loglen);
        clGetProgramBuildInfo(program_gpu, device_gpu, CL_PROGRAM_BUILD_LOG, loglen, logbuf, NULL);
        printf("%s\n", logbuf);
        return 1;
    }

    cmd_queue_cpu = clCreateCommandQueue(context_cpu, device_cpu, 0, NULL);
    cmd_queue_gpu = clCreateCommandQueue(context_gpu, device_gpu, 0, NULL);
    buf_gpu_outputs = clCreateBuffer(context_gpu, CL_MEM_READ_WRITE, sizeof(float) * 224 * 224 * 64, NULL, NULL);
    return 0;
}

void vggnet(float* images, float* network, int* labels, float* confidences, int num_images) {
    if (init_opencl() != 0) {
        return;
    }
    // Convolution layers
    float *c1_1, *c1_2, *c2_1, *c2_2, *c3_1, *c3_2, *c3_3,
        *c4_1, *c4_2, *c4_3, *c5_1, *c5_2, *c5_3;
    // Pooling layers
    float *p1, *p2, *p3, *p4, *p5;
    // Fully connected layers
    float *fc1, *fc2, *fc3;
    // Filters & Weights
    float *f1_1, *f1_2, *f2_1, *f2_2, *f3_1, *f3_2, *f3_3,
        *f4_1, *f4_2, *f4_3, *f5_1, *f5_2, *f5_3,
        *w1, *w2, *w3;
    // Biases
    float *b1_1, *b1_2, *b2_1, *b2_2, *b3_1, *b3_2, *b3_3,
        *b4_1, *b4_2, *b4_3, *b5_1, *b5_2, *b5_3, *b1, *b2, *b3;

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

    for (int i = 0; i < num_images; ++i) {
        float* image = images + i * 224 * 224 * 3;

        convolution_layer(image, c1_1, f1_1, b1_1, 224, 3, 64);
        convolution_layer(c1_1, c1_2, f1_2, b1_2, 224, 64, 64);
        pooling_layer(c1_2, p1, 112, 64);

        convolution_layer(p1, c2_1, f2_1, b2_1, 112, 64, 128);
        convolution_layer(c2_1, c2_2, f2_2, b2_2, 112, 128, 128);
        pooling_layer(c2_2, p2, 56, 128);

        convolution_layer(p2, c3_1, f3_1, b3_1, 56, 128, 256);
        convolution_layer(c3_1, c3_2, f3_2, b3_2, 56, 256, 256);
        convolution_layer(c3_2, c3_3, f3_3, b3_3, 56, 256, 256);
        pooling_layer(c3_3, p3, 28, 256);

        convolution_layer(p3, c4_1, f4_1, b4_1, 28, 256, 512);
        convolution_layer(c4_1, c4_2, f4_2, b4_2, 28, 512, 512);
        convolution_layer(c4_2, c4_3, f4_3, b4_3, 28, 512, 512);
        pooling_layer(c4_3, p4, 14, 512);

        convolution_layer(p4, c5_1, f5_1, b5_1, 14, 512, 512);
        convolution_layer(c5_1, c5_2, f5_2, b5_2, 14, 512, 512);
        convolution_layer(c5_2, c5_3, f5_3, b5_3, 14, 512, 512);
        pooling_layer(c5_3, p5, 7, 512);

        fc_layer(p5, fc1, w1, b1, 7 * 7 * 512, 4096);
        fc_layer(fc1, fc2, w2, b2, 4096, 4096);
        fc_layer(fc2, fc3, w3, b3, 4096, 1000);

        softmax(fc3);

        labels[i] = find_max(fc3);
        confidences[i] = fc3[labels[i]];
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
}
