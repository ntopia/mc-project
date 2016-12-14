
#define ReLU(x) (((x) > 0) ? (x) : 0)

__kernel void pooling_layer(    __global const float* restrict inputs,
                                __global float* restrict outputs,
                                int n, int d) {
    int i = get_global_id(0) / n;
    int j = get_global_id(0) % n;
    int pid = get_global_id(1);
    float maxv;

    maxv =           inputs[pid * n * n * 4 + (i * 2 + 0) * n * 2 + j * 2 + 0];
    maxv = max(maxv, inputs[pid * n * n * 4 + (i * 2 + 0) * n * 2 + j * 2 + 1]);
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
    int p, k, l, x, y;
    float sum = 0, v;
    __local float input[256];

    for (k = 0; k < 3; ++k) {
        x = i + k - 1;
        for (p = 0; p < d1; p++) {
            if (0 <= x && x < n) {
                v = inputs[n * n * p + x * n + j];
            }
            else {
                v = 0;
            }
            input[j] = v;
            barrier(CLK_LOCAL_MEM_FENCE);

            sum += (j - 1 >= 0) ? input[j - 1] * filters[3 * 3 * (opic * d1 + p) + k * 3 + 0] : 0;
            sum +=                input[j]     * filters[3 * 3 * (opic * d1 + p) + k * 3 + 1];
            sum += (j + 1 < n) ?  input[j + 1] * filters[3 * 3 * (opic * d1 + p) + k * 3 + 2] : 0;
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
