
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

__kernel void convolution_layer(    __global const float* inputs,
                                    __global const float* filters,
                                    __constant const float* biases,
                                    __global float* outputs,
                                    int n, int d1, int d2) {
    int id = get_global_id(0);
    int b, p, q, i, j, k, l, x, y;
    float sum;
    float filter[3];

    for (p = 0; p < 4; ++p) {
        b = id * 4 + p;

        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                outputs[n * n * b + i * n + j] = 0;
            }
        }

        for (q = 0; q < d1; ++q) {
            for (k = 0; k < 3; ++k) {
                filter[0] = filters[3 * 3 * (b * d1 + q) + k * 3 + 0];
                filter[1] = filters[3 * 3 * (b * d1 + q) + k * 3 + 1];
                filter[2] = filters[3 * 3 * (b * d1 + q) + k * 3 + 2];

                for (i = 0; i < n; ++i) {
                    x = i + k - 1;
                    if (x < 0 || x >= n) continue;

                    for (j = 0; j < n; ++j) {
                        sum = 0;
                        sum += (j - 1 >= 0) ? inputs[n * n * q + x * n + j + 0 - 1] * filter[0] : 0;
                        sum +=                inputs[n * n * q + x * n + j + 1 - 1] * filter[1];
                        sum += (j + 1 < n)  ? inputs[n * n * q + x * n + j + 2 - 1] * filter[2] : 0;
                        outputs[n * n * b + i * n + j] += sum;
                    }
                }
            }
        }

        for (k = 0; k < n * n; ++k) {
            outputs[n * n * b + k] = ReLU(outputs[n * n * b + k] + biases[b]);
        }
    }
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
