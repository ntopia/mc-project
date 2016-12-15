
#define ReLU(x) (((x) > 0) ? (x) : 0)

__kernel void pooling_layer(    __global const float* restrict inputs,
                                __global float* restrict outputs,
                                int n, int d) {
    int pid = get_global_id(0);
    int i = get_group_id(1);
    int j = get_global_id(1) % n;
    float maxv;

    maxv =           inputs[pid * n * n * 4 + (i * 2 + 0) * n * 2 + j * 2 + 0];
    maxv = max(maxv, inputs[pid * n * n * 4 + (i * 2 + 0) * n * 2 + j * 2 + 1]);
    barrier(CLK_GLOBAL_MEM_FENCE);
    maxv = max(maxv, inputs[pid * n * n * 4 + (i * 2 + 1) * n * 2 + j * 2 + 0]);
    maxv = max(maxv, inputs[pid * n * n * 4 + (i * 2 + 1) * n * 2 + j * 2 + 1]);
    outputs[pid * n * n + i * n + j] = maxv;
}

__kernel void convolution_layer(    __global const float* restrict inputs,
                                    __global const float* restrict filters,
                                    __constant const float* restrict biases,
                                    __global float* restrict outputs,
                                    int n, int d1, int d2) {
    int opic = get_global_id(0);
    int i = get_group_id(1);
    int j = get_global_id(1) % n;
    int p, k, x;
    float sum = 0, v;
    __local float input[256];
    input[0] = 0;
    input[n + 1] = 0;

    for (p = 0; p < d1; ++p) {
        for (k = 0; k < 3; ++k) {
            x = i + k - 1;
            if (0 <= x && x < n) {
                v = inputs[n * n * p + x * n + j];
            }
            else {
                v = 0;
            }
            input[j + 1] = v;
            barrier(CLK_LOCAL_MEM_FENCE);

            sum += input[j]     * filters[3 * 3 * (opic * d1 + p) + k * 3 + 0];
            sum += input[j + 1] * filters[3 * 3 * (opic * d1 + p) + k * 3 + 1];
            sum += input[j + 2] * filters[3 * 3 * (opic * d1 + p) + k * 3 + 2];
        }
    }
    outputs[n * n * opic + i * n + j] = ReLU(sum + biases[opic]);
}

__kernel void fc_layer( __constant const float4* input_neuron,
                        __global const float4* weights,
                        __constant const float* biases,
                        __global float* output_neuron,
                        int n, int m) {

    int id = get_global_id(0);
    float sum = 0;
    int i, j;

    sum = biases[id];
    n /= 4;
    for (i = 0; i < n; ++i) {
        sum += dot(input_neuron[i], weights[id * n + i]);
    }
    output_neuron[id] = ReLU(sum);
}
