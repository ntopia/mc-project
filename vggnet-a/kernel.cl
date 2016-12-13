
#define ReLU(x) (((x) > 0) ? (x) : 0)

__kernel void pooling_layer(    __global const float* restrict inputs,
                                __global float* restrict outputs,
                                int n, int d) {
    int id = get_global_id(0);
    int i, j;
    float maxv;

    inputs += n * n * id * 4;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            maxv =           inputs[(i * 2 + 0) * n * 2 + j * 2 + 0];
            maxv = max(maxv, inputs[(i * 2 + 0) * n * 2 + j * 2 + 1]);
            maxv = max(maxv, inputs[(i * 2 + 1) * n * 2 + j * 2 + 0]);
            maxv = max(maxv, inputs[(i * 2 + 1) * n * 2 + j * 2 + 1]);
            outputs[n * n * id + i * n + j] = maxv;
        }
    }
}

__kernel void convolution_layer(    __global const float* inputs,
                                    __global const float* filters,
                                    __global const float* biases,
                                    __global float* outputs,
                                    int n, int d1, int d2) {
    int id = get_global_id(0);
    int p, q, i, j, k, l, x, y;
    float sum;
    float filter[3][3];

    for (q = 0; q < d1; ++q) {
        for (k = 0; k < 3; ++k) {
            for (l = 0; l < 3; ++l) {
                filter[k][l] = filters[3 * 3 * (id * d1 + q) + k * 3 + l];
            }
        }

        for (k = 0; k < 3; ++k) {
            for (i = 0; i < n; ++i) {
                x = i + k - 1;
                if (x < 0 || x >= n) continue;

                for (j = 0; j < n; ++j) {
                    sum = 0;
                    for (l = 0; l < 3; ++l) {
                        y = j + l - 1;
                        if (0 <= y && y < n) {
                            sum += inputs[n * n * q + x * n + y] * filter[k][l];
                        }
                    }
                    outputs[n * n * id + i * n + j] += sum;
                }
            }
        }
    }

    for (k = 0; k < n * n; ++k) {
        outputs[n * n * id + k] = ReLU(outputs[n * n * id + k] + biases[id]);
    }
}

__kernel void fc_layer( __global const float* input_neuron,
                        __global const float* weights,
                        __global const float* biases,
                        __global float* output_neuron,
                        int n, int m) {

    int id = get_global_id(0);
    float sum = 0;
    int i, j;

    sum = biases[id];
    for (i = 0; i < n; ++i) {
        sum += input_neuron[i] * weights[id * n + i];
    }
    output_neuron[id] = ReLU(sum);
}
