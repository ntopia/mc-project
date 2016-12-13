#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void pooling2x2(float * input, float * output, int N)
{
  int i, j, k, l;
  for(i = 0; i < N; i++)
  {
    for(j = 0; j < N; j++)
    {
      float max = 0;
      for(k = 0; k < 2; k++)
      {
        for(l = 0; l < 2; l++)
        {
          float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
          max = (max > pixel) ? max : pixel;
        }
      }
      output[i * N + j] = max;
    }
  }
}

static void pooling_layer(float * inputs, float * outputs, int N, int D)
{
  int i;
  for(i = 0; i < D; i++)
  {
    float * input = inputs + i * N * N * 4;
    float * output = outputs + i * N * N;
    pooling2x2(input, output, N);
  }
}

static void convolution3x3(float * input, float * output, float * filter, int N)
{
  int i, j, k, l;
  for(i = 0; i < N; i++)
  {
    for(j = 0; j < N; j++)
    {
      float sum = 0;
      for(k = 0; k < 3; k++)
      {
        for(l = 0; l < 3; l++)
        {
          int x = i + k - 1;
          int y = j + l - 1; 
          if(x >= 0 && x < N && y >= 0 && y < N)
            sum += input[x * N + y] * filter[k * 3 + l];
        }
      }
      output[i * N + j] += sum;
    }
  }
}

#define ReLU(x) (((x)>0)?(x):0)
static void convolution_layer(float * inputs, float * outputs, float * filters, float * biases, int N, int D1, int D2)
{
  int i, j;

  memset(outputs, 0, sizeof(float) * N * N * D2);

  for(j = 0; j < D2; j++)
  {
    for(i = 0; i < D1; i++)
    {
      float * input = inputs + N * N * i;
      float * output = outputs + N * N * j;
      float * filter = filters + 3 * 3 * (j * D1 + i);
      convolution3x3(input, output, filter, N); 
    }
  }

  for(i = 0; i < D2; i++)
  {
    float * output = outputs + N * N * i;
    float bias = biases[i];
    for(j = 0; j < N * N; j++)
    {
      output[j] = ReLU(output[j] + bias);
    }
  }
}

static void fc_layer(float * input_neuron, float * output_neuron, float * weights, float * biases, int N, int M)
{
  int i, j;
  for(j = 0; j < M; j++)
  {
    float sum = 0;
    for(i = 0; i < N; i++)
    {
      sum += input_neuron[i] * weights[j * N + i];
    }
    sum += biases[j];
    output_neuron[j] = ReLU(sum);
  }
}

static void softmax(float * output)
{
  int i;
  float max = output[0];
  for(i = 1; i < 1000; i++)
  {
    max = (output[i] > max)?output[i]:max;
  }
  float sum = 0;
  for(i = 0; i < 1000; i++)
  {
    sum += exp(output[i] - max);
  }
  for(i = 0; i < 1000; i++)
  {
    output[i] = exp(output[i] - max) / sum;
  }
}

static int find_max(float * fc)
{
  int i;
  int maxid = 0;
  float maxval = 0;
  for(i = 0; i < 1000; i++)
  {
    if(maxval < fc[i])
    {
      maxval = fc[i];
      maxid = i;
    }
  }
  return maxid;
}


static float * get_param(float ** array, int size)
{
  float * subarray = *array;
  *array += size;
  return subarray;
}

void vggnet(float * images, float * network, int * labels, float * confidences, int num_images)
{
  float *c1_1, *c1_2, *c2_1, *c2_2, *c3_1, *c3_2, *c3_3, *c4_1, *c4_2, *c4_3, *c5_1, *c5_2, *c5_3; // Convolution layers
  float *p1, *p2, *p3, *p4, *p5; // Pooling layers
  float *fc1, *fc2, *fc3; // Fully connected layers
  float *f1_1, *f1_2, *f2_1, *f2_2, *f3_1, *f3_2, *f3_3, *f4_1, *f4_2, *f4_3, *f5_1, *f5_2, *f5_3, *w1, *w2, *w3; // Filters and weights
  float *b1_1, *b1_2, *b2_1, *b2_2, *b3_1, *b3_2, *b3_3, *b4_1, *b4_2, *b4_3, *b5_1, *b5_2, *b5_3, *b1, *b2, *b3; // Biases
  int i;

  c1_1 = (float *)malloc(sizeof(float) * 224 * 224 * 64);
  c1_2 = (float *)malloc(sizeof(float) * 224 * 224 * 64);

  p1 = (float *)malloc(sizeof(float) * 112 * 112 * 64);

  c2_1 = (float *)malloc(sizeof(float) * 112 * 112 * 128);
  c2_2 = (float *)malloc(sizeof(float) * 112 * 112 * 128);

  p2 = (float *)malloc(sizeof(float) * 56 * 56 * 128);

  c3_1 = (float *)malloc(sizeof(float) * 56 * 56 * 256);
  c3_2 = (float *)malloc(sizeof(float) * 56 * 56 * 256);
  c3_3 = (float *)malloc(sizeof(float) * 56 * 56 * 256);

  p3 = (float *)malloc(sizeof(float) * 28 * 28 * 256);

  c4_1 = (float *)malloc(sizeof(float) * 28 * 28 * 512);
  c4_2 = (float *)malloc(sizeof(float) * 28 * 28 * 512);
  c4_3 = (float *)malloc(sizeof(float) * 28 * 28 * 512);

  p4 = (float *)malloc(sizeof(float) * 14 * 14 * 512);

  c5_1 = (float *)malloc(sizeof(float) * 14 * 14 * 512);
  c5_2 = (float *)malloc(sizeof(float) * 14 * 14 * 512);
  c5_3 = (float *)malloc(sizeof(float) * 14 * 14 * 512);

  p5 = (float *)malloc(sizeof(float) * 7 * 7 * 512);

  fc1 = (float *)malloc(sizeof(float) * 4096);
  fc2 = (float *)malloc(sizeof(float) * 4096);
  fc3 = (float *)malloc(sizeof(float) * 1000);

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

  for(i = 0; i < num_images; i++)
  {
    float * image = images + i * 224 * 224 * 3;

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
