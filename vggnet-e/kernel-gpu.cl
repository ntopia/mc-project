
#define ReLU(x) (((x) > 0) ? (x) : 0)

__kernel void pooling_layer(    __global const float* restrict inputs,
                                __global float* restrict outputs,
                                int n, int d) {
    int pid = get_global_id(0);
    int i = get_group_id(1);
    int j = get_global_id(1) % n;
    float maxv;

    maxv = max( inputs[pid * n * n * 4 + (i * 2 + 0) * n * 2 + j * 2 + 0],
                inputs[pid * n * n * 4 + (i * 2 + 0) * n * 2 + j * 2 + 1] );
    barrier(CLK_GLOBAL_MEM_FENCE);
    maxv = max(maxv, inputs[pid * n * n * 4 + (i * 2 + 1) * n * 2 + j * 2 + 0]);
    maxv = max(maxv, inputs[pid * n * n * 4 + (i * 2 + 1) * n * 2 + j * 2 + 1]);
    outputs[pid * n * n + i * n + j] = maxv;
}

__kernel void convolution_1row_0_layer(    __global const float* restrict inputs,
                                    __global const float* restrict filters,
                                    __constant const float* restrict biases,
                                    __global float* restrict outputs,
                                    int n, int d1) {
    int opic = get_global_id(0);
    int i = get_group_id(1);
    int j = get_local_id(1);
    int p, k, x;
    float sum = 0, v;
    __local float input[3][226];
    __local float bias;

    if (j == 0) {
        bias = biases[opic];
        input[0][0] = input[0][n + 1] = 0;
        input[1][0] = input[1][n + 1] = 0;
        input[2][0] = input[2][n + 1] = 0;
    }

    for (p = 0; p < d1; ++p) {
        input[0][j + 1] = (i > 0) ?     inputs[n * n * p + (i - 1) * n + j] : 0;
        input[1][j + 1] =               inputs[n * n * p + i * n + j];
        input[2][j + 1] = (i + 1 < n) ? inputs[n * n * p + (i + 1) * n + j] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        sum += input[0][j]     * filters[3 * 3 * (opic * d1 + p) + 0 * 3 + 0];
        sum += input[0][j + 1] * filters[3 * 3 * (opic * d1 + p) + 0 * 3 + 1];
        sum += input[0][j + 2] * filters[3 * 3 * (opic * d1 + p) + 0 * 3 + 2];
        sum += input[1][j]     * filters[3 * 3 * (opic * d1 + p) + 1 * 3 + 0];
        sum += input[1][j + 1] * filters[3 * 3 * (opic * d1 + p) + 1 * 3 + 1];
        sum += input[1][j + 2] * filters[3 * 3 * (opic * d1 + p) + 1 * 3 + 2];
        sum += input[2][j]     * filters[3 * 3 * (opic * d1 + p) + 2 * 3 + 0];
        sum += input[2][j + 1] * filters[3 * 3 * (opic * d1 + p) + 2 * 3 + 1];
        sum += input[2][j + 2] * filters[3 * 3 * (opic * d1 + p) + 2 * 3 + 2];
    }
    outputs[n * n * opic + i * n + j] = ReLU(sum + bias);
}

__kernel void convolution_1row_1_layer(    __global const float* restrict inputs,
                                    __global const float* restrict filters,
                                    __constant const float* restrict biases,
                                    __global float* restrict outputs,
                                    int n, int d1) {
    int opic = get_global_id(0);
    int i = get_group_id(1);
    int j = get_local_id(1);
    int p, k, x;
    float sum = 0, v;
    __local float input[226];
    __local float bias;

    if (j == 0) {
        bias = biases[opic];
        input[0] = input[n + 1] = 0;
    }

    for (p = 0; p < d1; ++p) {
        for (k = 0; k < 3; ++k) {
            x = i + k - 1;
            input[j + 1] = (0 <= x && x < n) ? inputs[n * n * p + x * n + j] : 0;
            barrier(CLK_LOCAL_MEM_FENCE);

            sum += input[j]     * filters[3 * 3 * (opic * d1 + p) + k * 3 + 0];
            sum += input[j + 1] * filters[3 * 3 * (opic * d1 + p) + k * 3 + 1];
            sum += input[j + 2] * filters[3 * 3 * (opic * d1 + p) + k * 3 + 2];
        }
    }
    outputs[n * n * opic + i * n + j] = ReLU(sum + bias);
}

__kernel void convolution_2row_layer(    __global const float* restrict inputs,
                                    __global const float* restrict filters,
                                    __constant const float* restrict biases,
                                    __global float* restrict outputs,
                                    int n, int d1) {
    int opic = get_global_id(0);
    int di = get_local_id(1) / n;
    int i = get_group_id(1) * 2 + di;
    int j = get_local_id(1) % n;
    int p, k;
    float sum = 0, v;
    __local float input[4][64];
    __local float bias;

    if (get_local_id(1) == 0) {
        bias = biases[opic];
        input[0][0] = input[0][n + 1] = 0;
        input[1][0] = input[1][n + 1] = 0;
        input[2][0] = input[2][n + 1] = 0;
        input[3][0] = input[3][n + 1] = 0;
    }

    for (p = 0; p < d1; ++p) {
        input[di][j + 1] = (i > 0) ? inputs[n * n * p + (i - 1) * n + j] : 0;
        input[di + 2][j + 1] = (i + 1 < n) ? inputs[n * n * p + (i + 1) * n + j] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        sum += input[di    ][j    ] * filters[3 * 3 * (opic * d1 + p) + 0 * 3 + 0];
        sum += input[di    ][j + 1] * filters[3 * 3 * (opic * d1 + p) + 0 * 3 + 1];
        sum += input[di    ][j + 2] * filters[3 * 3 * (opic * d1 + p) + 0 * 3 + 2];
        sum += input[di + 1][j    ] * filters[3 * 3 * (opic * d1 + p) + 1 * 3 + 0];
        sum += input[di + 1][j + 1] * filters[3 * 3 * (opic * d1 + p) + 1 * 3 + 1];
        sum += input[di + 1][j + 2] * filters[3 * 3 * (opic * d1 + p) + 1 * 3 + 2];
        sum += input[di + 2][j    ] * filters[3 * 3 * (opic * d1 + p) + 2 * 3 + 0];
        sum += input[di + 2][j + 1] * filters[3 * 3 * (opic * d1 + p) + 2 * 3 + 1];
        sum += input[di + 2][j + 2] * filters[3 * 3 * (opic * d1 + p) + 2 * 3 + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    outputs[n * n * opic + i * n + j] = ReLU(sum + bias);
}

__kernel void convolution_4row_layer(    __global const float* restrict inputs,
                                    __global const float* restrict filters,
                                    __constant const float* restrict biases,
                                    __global float* restrict outputs,
                                    int n, int d1) {
    int opic = get_global_id(1);
    int offset = get_local_id(1);
    int i = get_group_id(0);
    int j = get_local_id(0);
    int p, k;
    float sum = 0, v;
    __local float input[3][60];

    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
        input[0][0] = input[0][n + 1] = 0;
        input[1][0] = input[1][n + 1] = 0;
        input[2][0] = input[2][n + 1] = 0;
    }

    for (p = 0; p < d1; ++p) {
        if (offset < 3) {
            k = i + offset - 1;
            input[offset][j + 1] = (0 <= k && k < n) ? inputs[n * n * p + k * n + j] : 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        sum += input[0][j    ] * filters[3 * 3 * (opic * d1 + p) + 0 * 3 + 0];
        sum += input[0][j + 1] * filters[3 * 3 * (opic * d1 + p) + 0 * 3 + 1];
        sum += input[0][j + 2] * filters[3 * 3 * (opic * d1 + p) + 0 * 3 + 2];
        sum += input[1][j    ] * filters[3 * 3 * (opic * d1 + p) + 1 * 3 + 0];
        sum += input[1][j + 1] * filters[3 * 3 * (opic * d1 + p) + 1 * 3 + 1];
        sum += input[1][j + 2] * filters[3 * 3 * (opic * d1 + p) + 1 * 3 + 2];
        sum += input[2][j    ] * filters[3 * 3 * (opic * d1 + p) + 2 * 3 + 0];
        sum += input[2][j + 1] * filters[3 * 3 * (opic * d1 + p) + 2 * 3 + 1];
        sum += input[2][j + 2] * filters[3 * 3 * (opic * d1 + p) + 2 * 3 + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    outputs[n * n * opic + i * n + j] = ReLU(sum + biases[opic]);
}

__kernel void convolution_break_layer(    __global const float* restrict inputs,
                                    __global const float* restrict filters,
                                    __constant const float* restrict biases,
                                    __global float* restrict outputs,
                                    int n, int d1) {
    int gid = get_global_id(0);
    int opic = gid / (n * n);
    int i = (gid % (n * n)) / n;
    int j = gid % n;
    int p, k, x;
    float sum = 0;

    for (p = 0; p < d1; ++p) {
        for (k = 0; k < 3; ++k) {
            x = i + k - 1;
            if (0 <= x && x < n) {
                sum += ((j - 1 >= 0) ? inputs[n * n * p + x * n + j - 1] : 0) * filters[3 * 3 * (opic * d1 + p) + k * 3 + 0];
                sum +=                 inputs[n * n * p + x * n + j]          * filters[3 * 3 * (opic * d1 + p) + k * 3 + 1];
                sum += ((j + 1 < n)  ? inputs[n * n * p + x * n + j + 1] : 0) * filters[3 * 3 * (opic * d1 + p) + k * 3 + 2];
            }
        }
    }
    outputs[n * n * opic + i * n + j] = ReLU(sum + biases[opic]);
}
