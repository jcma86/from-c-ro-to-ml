#ifndef __MONKEY_NN_H__
#define __MONKEY_NN_H__

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TO_STR(x) #x
#define STRINGIFY(x) TO_STR(x)

#define DATA_TYPE float
#define MAT_AT(m, r, c) (m).data[(r) * (m).stride + (c)]
#define MAT_PRINT(m) mat_print(m, #m)

typedef enum {
    ACTFX_LINEAR,
    ACTFX_SIGM,
} ActivationFx;

// Matrix
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;

    DATA_TYPE* data;
} Mat;

DATA_TYPE randv();
DATA_TYPE sigmoid(DATA_TYPE x);

Mat mat_create(size_t rows, size_t cols);
Mat mat_row(Mat m, size_t row);
void mat_cpy(Mat dst, Mat m);
void mat_rand(Mat m, DATA_TYPE min, DATA_TYPE max);
void mat_set(Mat m, DATA_TYPE value);
void mat_mult(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat b);
void mat_act(Mat m);
void mat_print(Mat m, char* id);

// NN
#define NN_INPUT(nn) (nn).o[0]
#define NN_OUTPUT(nn) (nn).o[(nn).count]

typedef struct {
    size_t count;
    Mat* w;
    Mat* b;
    Mat* o;
} NN;

NN nn_create(size_t* layers, size_t count);
void nn_rand(NN nn, DATA_TYPE min, DATA_TYPE max);
void nn_forward(NN nn);
DATA_TYPE nn_cost(NN nn, Mat m_in, Mat m_out);
void nn_finite_diff(NN nn, NN d, Mat m_in, Mat m_out, DATA_TYPE eps);
void nn_update_params(NN nn, NN d, DATA_TYPE learning_rate);

#endif