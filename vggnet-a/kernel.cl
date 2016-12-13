
#define ReLU(x) (((x) > 0) ? (x) : 0)

__kernel void pooling_layer(    __global const float* inputs,
                                __global float* outputs,
                                int n, int d) {
    int id = get_global_id(0);
    int i, j, k, l;
    int offset;
    float maxv, pixel;

    offset = id * n * n;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            maxv = 0;
            for (k = 0; k < 2; ++k) {
                for (l = 0; l < 2; ++l) {
                    pixel = inputs[offset * 4 + (i * 2 + k) * 2 * n + j * 2 + l];
                    maxv = (maxv < pixel) ? pixel : maxv;
                }
            }
            outputs[offset + i * n + j] = maxv;
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

        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                sum = 0;
                for (k = 0; k < 3; ++k) {
                    for (l = 0; l < 3; ++l) {
                        x = i + k - 1;
                        y = j + l - 1;
                        if (0 <= x && x < n && 0 <= y && y < n) {
                            sum += inputs[n * n * q + x * n + y] * filter[k][l];
                        }
                    }
                }
                outputs[n * n * id + i * n + j] += sum;
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
