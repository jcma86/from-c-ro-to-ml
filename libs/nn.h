#ifndef __MONKEY_NN_H__
#define __MONKEY_NN_H__

#include <assert.h>
#include <math.h>
#include <raylib.h>
#include <raymath.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TO_STR(x) #x
#define STRINGIFY(x) TO_STR(x)
#define max(a, b)               \
    ({                          \
        __typeof__(a) _a = (a); \
        __typeof__(b) _b = (b); \
        _a > _b ? _a : _b;      \
    })

#define min(a, b)               \
    ({                          \
        __typeof__(a) _a = (a); \
        __typeof__(b) _b = (b); \
        _a < _b ? _a : _b;      \
    })
#define ARRAY_SIZE(arr) (size_t)(sizeof((arr)) / sizeof((arr)[0]))

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

void mat_destroy(Mat m);
Mat mat_create(size_t rows, size_t cols);
Mat mat_row(Mat m, size_t row);
Mat mat_col(Mat m, size_t row);
void mat_cpy(Mat dst, Mat m);
void mat_rand(Mat m, DATA_TYPE min, DATA_TYPE max);
void mat_set(Mat m, DATA_TYPE value);
void mat_mult(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat b);
void mat_act(Mat m, ActivationFx act);
void mat_derivative(Mat m, ActivationFx act);
void mat_print(Mat m, char* id);

// NN
#define NN_INPUT(nn) (nn).o[0]
#define NN_OUTPUT(nn) (nn).o[(nn).n_layers - 1]

typedef enum {
    NN_LAYER_INPUT,
    NN_LAYER_HIDDEN,
    NN_LAYER_OUTPUT,
} LAYER_TYPE;

typedef struct {
    size_t layer;
    size_t index;

    Mat w;
    DATA_TYPE* b;
    DATA_TYPE* o;
} Neuron;

typedef struct {
    size_t layer;
    size_t count;
    Neuron* neurons;
    LAYER_TYPE type;
} Layer;

typedef struct {
    size_t n_layers;
    DATA_TYPE* cost;
    DATA_TYPE* lr;
    Mat* w;
    Mat* b;
    Mat* o;
    ActivationFx* act_fx;
} NN;

void nn_destroy(NN nn);
NN nn_create(size_t* layers, size_t count, DATA_TYPE learning_rate);
void nn_set_act_fx(NN nn, size_t layer, ActivationFx fx);
Neuron nn_get_neuron(NN nn, size_t layer, size_t n);
Layer nn_get_layer(NN nn, size_t index);
void nn_zero(NN nn);
void nn_rand(NN nn, DATA_TYPE min, DATA_TYPE max);
void nn_forward(NN nn);
DATA_TYPE nn_cost(NN nn, Mat m_in, Mat m_out, size_t initial_example, size_t batch_size);
void nn_finite_diff(NN nn, NN d, Mat m_in, Mat m_out, size_t initial_example, size_t batch_size, DATA_TYPE eps);
void nn_backprop(NN nn, NN nnd, Mat m_in, Mat m_out, size_t initial_example, size_t batch_size);

void nn_update_params(NN nn, NN d);

// NN Visualizer
#ifndef NNUI_WIDTH
#define NNUI_WIDTH 900
#endif
#ifndef NNUI_HEIGHT
#define NNUI_HEIGHT 600
#endif
#ifndef NNUI_NEURON_RADIUS
#define NNUI_NEURON_RADIUS 20
#endif
#ifndef NNUI_FONT_SIZE
#define NNUI_FONT_SIZE 20
#endif
#ifndef FONT_SPACE_BETWEEN
#define FONT_SPACE_BETWEEN 5
#endif

#ifndef NNUI_COST_CHART_POINTS
#define NNUI_COST_CHART_POINTS 5000
#endif

typedef struct {
    Vector2* coords;
} UI_Layer;

typedef struct {
    size_t count;
    Layer* layers;
    UI_Layer* ui;
    DATA_TYPE* lr;
} UI_NN;

bool nnui_should_close();
size_t nnui_get_current_example();
void nnui_end();
void nnui_init(NN nn, size_t n_examples, int fps);
void nnui_render();
void nnui_reset_cam();
void nnui_add_point_to_chart(DATA_TYPE point);
bool nnui_was_key_pressed(int key);
void nnui_set_status_message(char message[256]);

#endif