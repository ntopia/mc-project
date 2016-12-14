
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
    int i = get_global_id(0) / n;
    int j = get_global_id(0) % n;
    int qid = get_global_id(1);
    int lid = get_local_id(1);
    int localsz = get_local_size(1);
    int p, q, k, l, x, y;
    float sum = 0, v;
    __local float input[512][3][3];

    int lmt = d1 * 3 * 3;
    for (p = lid; p < lmt; p += 32) {
        q = p / 9;
        k = (p % 9) / 3;
        l = p % 3;
        x = i + k - 1;
        y = j + l - 1;
        if (0 <= x && x < n && 0 <= y && y < n) {
            v = inputs[n * n * q + x * n + y];
        }
        else {
            v = 0;
        }
        input[q][k][l] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (p = 0; p < d1; ++p) {
        sum += input[p][0][0] * filters[3 * 3 * (qid * d1 + p) + 0 * 3 + 0];
        sum += input[p][0][1] * filters[3 * 3 * (qid * d1 + p) + 0 * 3 + 1];
        sum += input[p][0][2] * filters[3 * 3 * (qid * d1 + p) + 0 * 3 + 2];
        sum += input[p][1][0] * filters[3 * 3 * (qid * d1 + p) + 1 * 3 + 0];
        sum += input[p][1][1] * filters[3 * 3 * (qid * d1 + p) + 1 * 3 + 1];
        sum += input[p][1][2] * filters[3 * 3 * (qid * d1 + p) + 1 * 3 + 2];
        sum += input[p][2][0] * filters[3 * 3 * (qid * d1 + p) + 2 * 3 + 0];
        sum += input[p][2][1] * filters[3 * 3 * (qid * d1 + p) + 2 * 3 + 1];
        sum += input[p][2][2] * filters[3 * 3 * (qid * d1 + p) + 2 * 3 + 2];
    }
    outputs[n * n * qid + i * n + j] = ReLU(sum + biases[qid]);
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
