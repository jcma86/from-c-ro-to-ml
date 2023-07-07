#include "nn.h"

const char* mat_datatype_printf(void)
{
    if (strcmp(STRINGIFY(DATA_TYPE), "int") == 0)
        return "%d";
    if (strcmp(STRINGIFY(DATA_TYPE), "float") == 0)
        return "%.4f";
    if (strcmp(STRINGIFY(DATA_TYPE), "double") == 0)
        return "%.7lf";

    assert(0 && "Printf format for data type undfined");

    return "";
}

DATA_TYPE fx_sigmoid(DATA_TYPE x)
{
    if (strcmp(STRINGIFY(DATA_TYPE), "float") == 0) {
        return 1.0 / (1.0 + expf(-x));
    }
    if (strcmp(STRINGIFY(DATA_TYPE), "double") == 0) {
        return 1.0 / (1.0 + exp(-x));
    }

    printf("sigmoid not implemented for \"%s\"", STRINGIFY(DATA_TYPE));
    assert(0 == 1);

    return 0.0;
}

DATA_TYPE fx_tanh(DATA_TYPE x)
{
    if (strcmp(STRINGIFY(DATA_TYPE), "float") == 0) {
        return (expf(x) - expf(-x)) / (expf(x) + expf(-x));
    }
    if (strcmp(STRINGIFY(DATA_TYPE), "double") == 0) {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    printf("tanh not implemented for \"%s\"", STRINGIFY(DATA_TYPE));
    assert(0 == 1);

    return 0.0;
}

DATA_TYPE fx_elu(DATA_TYPE x)
{
    if (strcmp(STRINGIFY(DATA_TYPE), "float") == 0) {
        return ELU_PARAM * (expf(x) - 1.0);
    }
    if (strcmp(STRINGIFY(DATA_TYPE), "double") == 0) {
        return ELU_PARAM * (exp(x) - 1.0);
    }

    printf("tanh not implemented for \"%s\"", STRINGIFY(DATA_TYPE));
    assert(0 == 1);

    return 0.0;
}

DATA_TYPE activation_handler(DATA_TYPE x, ActivationFx act)
{
    switch (act) {
    case ACTFX_LINEAR:
        return x;
    case ACTFX_STEP:
        return x < 0.0 ? 0.0 : 1.0;
    case ACTFX_SIGM:
        return fx_sigmoid(x);
    case ACTFX_TANH:
        return fx_tanh(x);
    case ACTFX_RELU:
        return max(RELU_PARAM * x, x);
    case ACTFX_ELU:
        return x >= 0.0 ? x : fx_elu(x);
    }

    assert(0 && "Activation function not defined.");

    return 0.0;
}

DATA_TYPE derivative_handler(DATA_TYPE x, ActivationFx act)
{
    switch (act) {
    case ACTFX_LINEAR:
        return 1;
    case ACTFX_STEP:
        return x <= 0.0 ? 0.0 : 1.0;
    case ACTFX_SIGM:
        return x * (1 - x);
    case ACTFX_TANH:
        return 1.0 - (fx_tanh(x) * fx_tanh(x));
    case ACTFX_RELU:
        return x >= 0.0 ? 1.0 : RELU_PARAM;
    case ACTFX_ELU:
        return x >= 0.0 ? 1.0 : x + RELU_PARAM;
    }

    assert(0 && "Activation function not defined.");

    return 0.0;
}

DATA_TYPE randv()
{
    return (DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX;
}

void mat_destroy(Mat m)
{
    if (m.data != NULL)
        free(m.data);
    m.data = NULL;
}

Mat mat_create(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = malloc(rows * cols * sizeof(DATA_TYPE));

    assert(m.data != NULL);

    return m;
}

Mat mat_row(Mat m, size_t row)
{
    Mat rm;
    rm.rows = 1;
    rm.cols = m.cols;
    rm.stride = m.stride;
    rm.data = &MAT_AT(m, row, 0);

    return rm;
}

Mat mat_col(Mat m, size_t col)
{
    assert(col >= 0 && col < m.cols);
    Mat rm;
    rm.rows = m.rows;
    rm.cols = 1;
    rm.stride = m.stride;
    rm.data = &MAT_AT(m, 0, col);

    return rm;
}

void mat_cpy(Mat dst, Mat m)
{
    assert(dst.rows == m.rows);
    assert(dst.cols == m.cols);

    for (size_t r = 0; r < dst.rows; r++) {
        for (size_t c = 0; c < dst.cols; c++) {
            MAT_AT(dst, r, c) = MAT_AT(m, r, c);
        }
    }
}

void mat_shuffle_rows(Mat m)
{
    assert(m.data != NULL);
    for (size_t r1 = 1; r1 < m.rows; r1++) {
        size_t r2 = rand() % (m.rows - r1) + r1;
        mat_swap_rows(m, r1, r2);
    }
}

void mat_swap_rows(Mat m, size_t r1, size_t r2)
{
    assert(r1 < m.rows);
    assert(r2 < m.rows);

    if (r1 == r2)
        return;

    for (size_t c = 0; c < m.cols; c++) {
        DATA_TYPE tmp = MAT_AT(m, r1, c);
        MAT_AT(m, r1, c) = MAT_AT(m, r2, c);
        MAT_AT(m, r2, c) = tmp;
    }
}

void mat_rand(Mat m, DATA_TYPE min, DATA_TYPE max)
{
    for (size_t r = 0; r < m.rows; r++) {
        for (size_t c = 0; c < m.cols; c++) {
            MAT_AT(m, r, c) = randv() * (max - min) + min;
        }
    }
}

void mat_set(Mat m, DATA_TYPE value)
{
    for (size_t r = 0; r < m.rows; r++) {
        for (size_t c = 0; c < m.cols; c++) {
            MAT_AT(m, r, c) = value;
        }
    }
}

void mat_mult(Mat dst, Mat a, Mat b)
{
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);

    size_t n = a.cols;
    for (size_t r = 0; r < dst.rows; r++) {
        for (size_t c = 0; c < dst.cols; c++) {
            MAT_AT(dst, r, c) = 0;
            for (size_t i = 0; i < n; i++) {
                MAT_AT(dst, r, c) += MAT_AT(a, r, i) * MAT_AT(b, i, c);
            }
        }
    }
}

void mat_sum(Mat dst, Mat b)
{
    assert(dst.rows == b.rows);
    assert(dst.cols == b.cols);

    for (size_t r = 0; r < dst.rows; r++) {
        for (size_t c = 0; c < dst.cols; c++) {
            MAT_AT(dst, r, c) += MAT_AT(b, r, c);
        }
    }
}

void mat_act(Mat m, ActivationFx act)
{
    for (size_t r = 0; r < m.rows; r++) {
        for (size_t c = 0; c < m.cols; c++) {
            MAT_AT(m, r, c) = activation_handler(MAT_AT(m, r, c), act);
        }
    }
}

void mat_derivative(Mat m, ActivationFx act)
{
    for (size_t r = 0; r < m.rows; r++) {
        for (size_t c = 0; c < m.cols; c++) {
            MAT_AT(m, r, c) = derivative_handler(MAT_AT(m, r, c), act);
        }
    }
}

void mat_print(Mat m, char* id)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%s ", mat_datatype_printf());

    printf("%s = [\n", id);
    for (size_t r = 0; r < m.rows; r++) {
        printf("    ");
        for (size_t c = 0; c < m.cols; c++) {
            printf(buf, MAT_AT(m, r, c));
        }
        printf("\n");
    }
    printf("]\n");
}

void nn_destroy(NN nn)
{
    if (nn.b != NULL) {
        for (size_t i = 0; i < nn.n_layers - 1; i++)
            mat_destroy(nn.b[i]);
        free(nn.b);
        nn.b = NULL;
    }
    if (nn.w != NULL) {
        for (size_t i = 0; i < nn.n_layers - 1; i++)
            mat_destroy(nn.w[i]);
        free(nn.w);
        nn.w = NULL;
    }
    if (nn.o != NULL) {
        for (size_t i = 0; i < nn.n_layers; i++)
            mat_destroy(nn.o[i]);
        free(nn.o);
        nn.o = NULL;
    }
    if (nn.act_fx != NULL) {
        free(nn.act_fx);
        nn.act_fx = NULL;
    }
    if (nn.cost != NULL) {
        free(nn.cost);
        nn.cost = NULL;
    }
    if (nn.lr != NULL) {
        free(nn.lr);
        nn.lr = NULL;
    }
}

NN nn_load_from_file(const char* filename)
{
    assert(filename != NULL);

    FILE* f;
    f = fopen("./nn_0.00000.nn", "rb");

    char buffer[10];
    fread(buffer, sizeof(char), 10, f);
    if (strcmp(buffer, STRINGIFY(DATA_TYPE)) != 0) {
        printf("Wrong data type, expected: %s, got: %s\n", STRINGIFY(DATA_TYPE), buffer);
        exit(1);
    }

    size_t n_layers;
    fread(&n_layers, sizeof(size_t), 1, f);

    size_t layers[n_layers];
    fread(&layers, sizeof(size_t), n_layers, f);

    NN nn = nn_create(layers, n_layers, 0.1);

    for (size_t l = 1; l < n_layers; l++) {
        size_t t = layers[l - 1] * layers[l];
        fread(nn.w[l - 1].data, sizeof(DATA_TYPE), t, f);
        MAT_PRINT(nn.w[l - 1]);
    }

    for (size_t l = 1; l < n_layers; l++) {
        size_t t = layers[l];
        fread(nn.b[l - 1].data, sizeof(DATA_TYPE), t, f);
        MAT_PRINT(nn.b[l - 1]);
    }
    fread(nn.act_fx, sizeof(ActivationFx), nn.n_layers - 1, f);

    fclose(f);

    return nn;
}

NN nn_create(size_t* layers, size_t count, DATA_TYPE learning_rate)
{
    assert(count > 0);

    NN nn;
    nn.n_layers = count;
    nn.layers = layers;

    nn.w = malloc(sizeof(Mat) * (nn.n_layers - 1));
    assert(nn.w != NULL);
    nn.b = malloc(sizeof(Mat) * (nn.n_layers - 1));
    assert(nn.b != NULL);
    nn.act_fx = malloc(sizeof(Mat) * (nn.n_layers - 1));
    assert(nn.act_fx != NULL);
    nn.o = malloc(sizeof(Mat) * nn.n_layers);
    assert(nn.o != NULL);

    nn.cost = malloc(sizeof(DATA_TYPE));
    nn.lr = malloc(sizeof(DATA_TYPE));
    *nn.cost = 0.0;
    *nn.lr = min(max(0.0, learning_rate), 1.0);

    nn.o[0] = mat_create(1, layers[0]);
    mat_set(nn.o[0], 0);
    for (size_t i = 1; i < nn.n_layers; i++) {
        nn.w[i - 1] = mat_create(layers[i - 1], layers[i]);
        nn.b[i - 1] = mat_create(1, layers[i]);
        nn.act_fx[i - 1] = ACTFX_SIGM;
        nn.o[i] = mat_create(1, layers[i]);
    }

    return nn;
}

void nn_set_act_fx(NN nn, size_t layer, ActivationFx fx)
{
    assert(layer > 0 && layer < nn.n_layers);
    nn.act_fx[layer - 1] = fx;
}

Neuron nn_get_neuron(NN nn, size_t layer, size_t index)
{
    assert(layer < nn.n_layers);
    assert(index < nn.o[layer].cols);

    Neuron n;
    n.layer = layer;
    n.index = index;
    n.o = &MAT_AT(nn.o[layer], 0, index);
    n.b = NULL;
    if (layer > 0) {
        n.b = &MAT_AT(nn.b[layer - 1], 0, index);
        n.w = mat_col(nn.w[layer - 1], index);
    }

    return n;
}

Layer nn_get_layer(NN nn, size_t index)
{
    assert(index < nn.n_layers);

    Layer l;
    l.layer = index;
    l.count = nn.o[index].cols;
    l.neurons = malloc(sizeof(Neuron) * l.count);
    assert(l.neurons != NULL);

    for (size_t n = 0; n < l.count; n++)
        l.neurons[n] = nn_get_neuron(nn, index, n);

    if (index == 0)
        l.type = NN_LAYER_INPUT;
    else if (index == nn.n_layers)
        l.type = NN_LAYER_OUTPUT;
    else
        l.type = NN_LAYER_HIDDEN;

    return l;
}

void nn_free_layer(Layer l)
{
    if (l.neurons != NULL) {
        free(l.neurons);
        l.neurons = NULL;
    }
}

void nn_zero(NN nn)
{
    mat_set(nn.o[nn.n_layers - 1], (DATA_TYPE)0);
    for (size_t i = 0; i < nn.n_layers - 1; i++) {
        mat_set(nn.w[i], (DATA_TYPE)0);
        mat_set(nn.b[i], (DATA_TYPE)0);
        mat_set(nn.o[i], (DATA_TYPE)0);
    }
}

void nn_rand(NN nn, DATA_TYPE min, DATA_TYPE max)
{
    for (size_t i = 0; i < nn.n_layers - 1; i++) {
        mat_rand(nn.w[i], min, max);
        mat_rand(nn.b[i], min, max);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.n_layers - 1; i++) {
        mat_mult(nn.o[i + 1], nn.o[i], nn.w[i]);
        mat_sum(nn.o[i + 1], nn.b[i]);
        mat_act(nn.o[i + 1], nn.act_fx[i]);
    }
}

DATA_TYPE nn_cost(NN nn, Mat m_in, Mat m_out, size_t initial_example, size_t batch_size)
{
    assert(m_in.rows == m_out.rows);
    assert(initial_example + batch_size <= m_out.rows);
    assert(m_out.cols == NN_OUTPUT(nn).cols);

    DATA_TYPE err = 0;

    for (size_t i = initial_example; i < (initial_example + batch_size); i++) {
        Mat x = mat_row(m_in, i);
        Mat y = mat_row(m_out, i);

        mat_cpy(NN_INPUT(nn), x);
        nn_forward(nn);

        for (size_t o = 0; o < m_out.cols; o++) {
            DATA_TYPE d = MAT_AT(NN_OUTPUT(nn), 0, o) - MAT_AT(y, 0, o);
            err += d * d;
        }
    }

    return err / batch_size;
}

void nn_finite_diff(NN nn, NN nnd, Mat m_in, Mat m_out, size_t initial_example, size_t batch_size, DATA_TYPE eps)
{
    DATA_TYPE tmp;
    DATA_TYPE cc = nn_cost(nn, m_in, m_out, initial_example, batch_size);

    for (size_t i = 0; i < nn.n_layers - 1; i++) {
        for (size_t r = 0; r < nn.w[i].rows; r++) {
            for (size_t c = 0; c < nn.w[i].cols; c++) {
                tmp = MAT_AT(nn.w[i], r, c);
                MAT_AT(nn.w[i], r, c) += eps;
                MAT_AT(nnd.w[i], r, c) = (nn_cost(nn, m_in, m_out, initial_example, batch_size) - cc) / eps;
                MAT_AT(nn.w[i], r, c) = tmp;
            }
        }

        for (size_t r = 0; r < nn.b[i].rows; r++) {
            for (size_t c = 0; c < nn.b[i].cols; c++) {
                tmp = MAT_AT(nn.b[i], r, c);
                MAT_AT(nn.b[i], r, c) += eps;
                MAT_AT(nnd.b[i], r, c) = (nn_cost(nn, m_in, m_out, initial_example, batch_size) - cc) / eps;
                MAT_AT(nn.b[i], r, c) = tmp;
            }
        }
    }

    // *nn.cost = cc;
}

void nn_backprop(NN nn, NN nnd, Mat m_in, Mat m_out, size_t initial_example, size_t batch_size)
{
    nn_zero(nnd);
    // DATA_TYPE cost = 0.0;
    size_t lim = min(initial_example + batch_size, m_in.rows);
    for (size_t i = initial_example; i < lim; i++) {
        mat_cpy(NN_INPUT(nn), mat_row(m_in, i));
        nn_forward(nn);

        for (size_t j = 0; j < nn.n_layers; j++)
            mat_set(nnd.o[j], 0.0);

        for (size_t j = 0; j < m_out.cols; j++) {
            DATA_TYPE dif = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(m_out, i, j);
            MAT_AT(NN_OUTPUT(nnd), 0, j) = 2 * (dif);
            // cost += dif * dif;
        }

        for (size_t l = nn.n_layers - 1; l > 0; l--) {
            for (size_t j = 0; j < nn.o[l].cols; j++) {
                DATA_TYPE a = MAT_AT(nn.o[l], 0, j);
                DATA_TYPE dc = MAT_AT(nnd.o[l], 0, j);
                DATA_TYPE da = derivative_handler(a, nn.act_fx[l - 1]);
                MAT_AT(nnd.b[l - 1], 0, j) += dc * da;

                for (size_t k = 0; k < nn.o[l - 1].cols; k++) {
                    DATA_TYPE pa = MAT_AT(nn.o[l - 1], 0, k);
                    DATA_TYPE w = MAT_AT(nn.w[l - 1], k, j);

                    MAT_AT(nnd.w[l - 1], k, j) += dc * da * pa;
                    MAT_AT(nnd.o[l - 1], 0, k) += dc * da * w;
                }
            }
        }
    }

    for (size_t i = 0; i < nnd.n_layers - 1; i++) {
        for (size_t j = 0; j < nnd.w[i].rows; j++) {
            for (size_t k = 0; k < nnd.w[i].cols; k++) {
                MAT_AT(nnd.w[i], j, k) /= batch_size;
            }
        }
        for (size_t j = 0; j < nnd.b[i].rows; j++) {
            for (size_t k = 0; k < nnd.b[i].cols; k++) {
                MAT_AT(nnd.b[i], j, k) /= batch_size;
            }
        }
    }

    // *nn.cost = cost / batch_size;
}

void nn_update_params(NN nn, NN nnd)
{
    for (size_t i = 0; i < nn.n_layers - 1; i++) {
        for (size_t r = 0; r < nn.w[i].rows; r++) {
            for (size_t c = 0; c < nn.w[i].cols; c++) {
                MAT_AT(nn.w[i], r, c) -= (*nn.lr) * MAT_AT(nnd.w[i], r, c);
            }
        }

        for (size_t r = 0; r < nn.b[i].rows; r++) {
            for (size_t c = 0; c < nn.b[i].cols; c++) {
                MAT_AT(nn.b[i], r, c) -= (*nn.lr) * MAT_AT(nnd.b[i], r, c);
            }
        }
    }
}

void nn_save_to_file(NN nn, const char* filename)
{
    assert(filename != NULL);

    FILE* file = fopen(filename, "wb");
    assert(file != NULL);

    printf("Saving to file %s\n", filename);
    char buffer[10];
    snprintf(buffer, sizeof(buffer), "%s", STRINGIFY(DATA_TYPE));

    fwrite(buffer, sizeof(buffer), 1, file);
    fwrite(&nn.n_layers, sizeof(size_t), 1, file);
    fwrite(nn.layers, sizeof(size_t), nn.n_layers, file);

    // Writing weights
    for (size_t l = 0; l < nn.n_layers - 1; l++) {
        size_t t = nn.w[l].rows * nn.w[l].cols;
        fwrite(nn.w[l].data, sizeof(DATA_TYPE), t, file);
    }

    // Writing bias
    for (size_t l = 0; l < nn.n_layers - 1; l++)
        fwrite(nn.b[l].data, sizeof(DATA_TYPE), nn.b[l].cols, file);

    // Writing Act Fxs.
    fwrite(nn.act_fx, sizeof(ActivationFx), nn.n_layers - 1, file);

    fclose(file);
    printf("File saved\n");
}

// NN Visualizer
void (*test_area_render)(NN*, void*)
    = NULL;
char __nnui_ready__ = 0;
int __fps__ = 15;
char __status__[256];
size_t __current_example__ = 0;
size_t __n_examples__ = 0;
NN* __nn__ = NULL;
void* __param__ = NULL;
bool __draw_test__ = false;

Camera2D __cam_main__ = { 0 };
Camera2D __cam_graph__ = { 0 };
Camera2D __cam_net_inputs__ = { 0 };
Camera2D __cam_cost_chart__ = { 0 };
Camera2D __cam_learning_rate__ = { 0 };
Camera2D __cam_fps__ = { 0 };
Camera2D __cam_example__ = { 0 };
Camera2D __cam_neuron_info__ = { 0 };
Camera2D __cam_neuron_iw_info__ = { 0 };
Camera2D __cam_status_bar__ = { 0 };
Camera2D __cam_test_area__ = { 0 };

const int __padding__ = 5;
const int __right_panel_width__ = 350;
const int __left_panel_width__ = NNUI_WIDTH - __right_panel_width__;
const int __inputs_panel_width__ = 320;
const int __inputs_panel_height__ = 5.5 * NNUI_FONT_SIZE;
const int __chart_panel_height__ = NNUI_HEIGHT / 2;
const int __fps_panel_width__ = 200;
const int __fps_bar_width__ = 180;
const int __example_panel_width__ = 200;
const int __example_bar_width__ = 180;
Vector2 __cam_main_offset__ = (Vector2) { .x = 0.0, .y = 0.0 };
Vector2 __cam_graph_offset__ = (Vector2) { .x = 0.0, .y = 0.0 };
Vector2 __cam_net_inputs_offset__ = (Vector2) { .x = __left_panel_width__ - (__inputs_panel_width__ + __padding__), .y = 2 * __padding__ };
Vector2 __cam_cost_chart_offset__ = (Vector2) { .x = __left_panel_width__, .y = 0.0 };
Vector2 __cam_learning_rate_offset__ = (Vector2) { .x = __left_panel_width__ + (3 * __padding__), .y = __chart_panel_height__ - 50 };
Vector2 __cam_fps_offset__ = (Vector2) { .x = 5 * __padding__, .y = 3 * __padding__ };
Vector2 __cam_example_offset__ = (Vector2) { .x = 5 * __padding__, .y = 15 * __padding__ };
Vector2 __cam_neuron_info_offset__ = (Vector2) { .x = __left_panel_width__, .y = __chart_panel_height__ };
Vector2 __cam_neuron_iw_info_offset__ = (Vector2) { .x = __left_panel_width__, .y = __chart_panel_height__ + (6 * NNUI_FONT_SIZE) };
Vector2 __cam_status_bar_offset__ = (Vector2) { .x = __padding__, .y = NNUI_HEIGHT - __padding__ - NNUI_FONT_SIZE };
Vector2 __cam_test_area_offset__ = (Vector2) { .x = NNUI_WIDTH, .y = 0 };

DATA_TYPE __cost_points__[NNUI_COST_CHART_POINTS];
float __max_cost__ = -INT32_MAX;
size_t __cost_current_point__ = 0;
Vector2 __plot_area__ = { .x = __right_panel_width__ - (4 * __padding__), .y = __chart_panel_height__ - 100 };
Vector2 __plot_area_offset__ = { .x = 2 * __padding__, .y = 100 / 2 };

UI_NN __nnui__;
Neuron __selected_neuron__ = (Neuron) { .layer = INT32_MAX };
bool paused;
bool nnui_should_close()
{
    assert(__nnui_ready__ == 1 && "You must initialize the render first => nnui_init(NN nn, size_t n_examples, int fps)");
    return WindowShouldClose();
}

void nnui_end()
{
    if (__nnui__.ui != NULL) {
        if (__nnui__.ui->coords != NULL) {
            free(__nnui__.ui->coords);
            __nnui__.ui->coords = NULL;
        }
        free(__nnui__.ui);
        __nnui__.ui = NULL;
    }
}

size_t nnui_get_current_example()
{
    return __current_example__;
}

void nnui_init(NN nn, size_t n_examples, int fps, bool draw_test_area)
{
    __draw_test__ = draw_test_area;
    if (draw_test_area)
        InitWindow(NNUI_WIDTH + NNUI_TEST_AREA_WIDTH, NNUI_HEIGHT, "Neural Network Visualizer");
    else
        InitWindow(NNUI_WIDTH, NNUI_HEIGHT, "Neural Network Visualizer");

    __n_examples__ = n_examples - 1;
    __fps__ = max(fps, 0);
    __fps__ = __fps__ > 120 ? 0 : __fps__;
    __nn__ = &nn;

    SetTargetFPS(__fps__);
    SetTraceLogLevel(LOG_NONE);

    snprintf(__status__, sizeof(__status__), "NN Visualizer Tool v0.1");

    __cam_main__.offset = __cam_main_offset__;
    __cam_graph__.offset = __cam_graph_offset__;
    __cam_net_inputs__.offset = __cam_net_inputs_offset__;
    __cam_cost_chart__.offset = __cam_cost_chart_offset__;
    __cam_learning_rate__.offset = __cam_learning_rate_offset__;
    __cam_fps__.offset = __cam_fps_offset__;
    __cam_example__.offset = __cam_example_offset__;
    __cam_neuron_info__.offset = __cam_neuron_info_offset__;
    __cam_neuron_iw_info__.offset = __cam_neuron_iw_info_offset__;
    __cam_status_bar__.offset = __cam_status_bar_offset__;
    __cam_test_area__.offset = __cam_test_area_offset__;

    __cam_main__.zoom = 1.0f;
    __cam_graph__.zoom = 1.0f;
    __cam_net_inputs__.zoom = 1.0f;
    __cam_cost_chart__.zoom = 1.0f;
    __cam_learning_rate__.zoom = 1.0f;
    __cam_fps__.zoom = 1.0f;
    __cam_example__.zoom = 1.0f;
    __cam_neuron_info__.zoom = 1.0f;
    __cam_neuron_iw_info__.zoom = 1.0f;
    __cam_status_bar__.zoom = 1.0f;
    __cam_test_area__.zoom = 1.0f;

    __nnui__.count = nn.n_layers;
    __nnui__.lr = nn.lr;
    __nnui__.layers = malloc(sizeof(Layer) * __nnui__.count);
    assert(__nnui__.layers != NULL);

    for (size_t l = 0; l < nn.n_layers; l++)
        __nnui__.layers[l] = nn_get_layer(nn, l);

    __nnui__.ui = malloc(sizeof(UI_Layer) * __nnui__.count);
    assert(__nnui__.ui != NULL);

    // (0, 0) Top left corner
    // Defining min/max space between layers
    float x_offset = __left_panel_width__ / 2;
    float space_layer = __left_panel_width__ / (__nnui__.count);
    space_layer = min(space_layer, 20 * NNUI_NEURON_RADIUS);
    space_layer = max(space_layer, 7 * NNUI_NEURON_RADIUS);

    x_offset -= space_layer * (__nnui__.count - 1) / 2;
    for (size_t l = 0; l < __nnui__.count; l++) {
        __nnui__.ui[l].coords = malloc(sizeof(Vector2) * __nnui__.layers[l].count);
        assert(__nnui__.ui[l].coords != NULL);

        float x = (l * space_layer) + x_offset;

        // Defining min/max space between neurons in layer
        float y_offset = NNUI_HEIGHT / 2;
        float space_neuron = NNUI_HEIGHT / (__nnui__.layers[l].count);
        space_neuron = min(space_neuron, 8 * NNUI_NEURON_RADIUS);
        space_neuron = max(space_neuron, 3 * NNUI_NEURON_RADIUS);
        y_offset -= space_neuron * (__nnui__.layers[l].count - 1) / 2;

        for (size_t n = 0; n < __nnui__.layers[l].count; n++) {
            float y = (n * space_neuron) + y_offset;
            __nnui__.ui[l].coords[n] = (Vector2) { .x = x, .y = y };
        }
    }

    __nnui_ready__ = 1;
}

void nnui_set_test_area_render(void (*ptr_render)(NN*, void*))
{
    test_area_render = ptr_render;
}

void nnui_render_test_area()
{
    if (test_area_render)
        test_area_render(__nn__, __param__);
}

void nnui_render_graph(Vector2 mouse_pos)
{
    for (size_t l = 0; l < __nnui__.count; l++) {
        for (size_t n = 0; n < __nnui__.layers[l].count; n++) {
            Vector2 p = __nnui__.ui[l].coords[n];
            Vector2 v = GetWorldToScreen2D((Vector2) { .x = p.x, .y = p.y }, __cam_graph__);

            // Out of screen (top)
            if (v.x < __cam_graph_offset__.x || v.y < __cam_graph_offset__.y)
                continue;
            // Out of screen (bottom)
            if (v.x > __left_panel_width__ || v.y > NNUI_HEIGHT)
                break;

            Vector2 mouse_v = GetScreenToWorld2D(mouse_pos, __cam_graph__);
            bool collition = CheckCollisionCircles(
                (Vector2) { .x = p.x, .y = p.y },
                NNUI_NEURON_RADIUS,
                (Vector2) { .x = mouse_v.x, .y = mouse_v.y },
                1);

            // Drawing neuron (red when clicked)
            if (collition) {
                __selected_neuron__ = __nnui__.layers[l].neurons[n];
                __cam_neuron_iw_info__.target = (Vector2) { .x = 0, .y = 0 };
                if (l == 0)
                    DrawRectangle(p.x - NNUI_NEURON_RADIUS, p.y - NNUI_NEURON_RADIUS, 2 * NNUI_NEURON_RADIUS, 2 * NNUI_NEURON_RADIUS, BLACK);
                else
                    DrawCircle(p.x, p.y, NNUI_NEURON_RADIUS, RED);
            } else {
                if (l == 0)
                    DrawRectangle(p.x - NNUI_NEURON_RADIUS, p.y - NNUI_NEURON_RADIUS, 2 * NNUI_NEURON_RADIUS, 2 * NNUI_NEURON_RADIUS, GRAY);
                else {
                    float h = fx_sigmoid(*(__nnui__.layers[l].neurons[n].o)) * 180.0f;
                    h -= 60.0f;
                    if (h < 0.0)
                        h = 360.0 - h;
                    DrawCircle(p.x, p.y, NNUI_NEURON_RADIUS, ColorFromHSV(h, 1.0, 1.0));
                }
            }

            // Text in neuron
            char buf[10];
            int tw;
            snprintf(buf, sizeof(buf), "l: %zu", l);
            tw = MeasureText(buf, 5);
            DrawText(buf, p.x - (tw / 2), p.y - 10, 5, BLACK);

            snprintf(buf, sizeof(buf), "n: %zu", n);
            tw = MeasureText(buf, 5);
            DrawText(buf, p.x - (tw / 2), p.y, 5, BLACK);

            // Drawing weight lines
            if (l > 0) {
                p.x -= NNUI_NEURON_RADIUS;
                for (size_t i = 0; i < __nnui__.layers[l - 1].count; i++) {
                    Vector2 pn = __nnui__.ui[l - 1].coords[i];
                    pn.x += NNUI_NEURON_RADIUS;
                    float h = fx_sigmoid(MAT_AT(__nnui__.layers[l].neurons[n].w, i, 0)) * 180.0f;
                    h -= 60.0f;
                    if (h < 0.0)
                        h = 360.0 - h;
                    DrawLineEx(pn, p, min(__cam_graph__.zoom, 0.8), ColorFromHSV(h, 1.0, 1.0));
                }
            }
        }
    }
    DrawCircle(__left_panel_width__ / 2, NNUI_HEIGHT / 2, 2, BLACK);
}

void nnui_render_example()
{
    char buffer[120];
    DrawRectangle(-2 * __padding__, -__padding__, __example_panel_width__, (2 * NNUI_FONT_SIZE) + (2 * __padding__), (Color) { 225, 225, 225, 180 });
    snprintf(buffer, sizeof(buffer), "Example: %zu", __current_example__);
    DrawText(buffer, 0, 0, NNUI_FONT_SIZE, BLACK);

    float w = __example_bar_width__;
    float y = 1.5 * NNUI_FONT_SIZE;
    DrawLineEx((Vector2) { .x = 0, .y = y }, (Vector2) { .x = w, .y = y }, 2, BROWN);
    float p = ((float)__current_example__ / __n_examples__) * w;
    DrawCircle(p, y, 8, DARKGRAY);
}

void nnui_render_fps()
{
    char buffer[120];
    DrawRectangle(-2 * __padding__, -__padding__, __fps_panel_width__, (2 * NNUI_FONT_SIZE) + (2 * __padding__), (Color) { 225, 225, 225, 180 });
    if (__fps__ == 0)
        snprintf(buffer, sizeof(buffer), "FPS: %i", GetFPS());
    else
        snprintf(buffer, sizeof(buffer), "FPS: %i (real: %i)", __fps__, GetFPS());
    DrawText(buffer, 0, 0, NNUI_FONT_SIZE, BLACK);

    float w = __fps_bar_width__;
    float y = 1.5 * NNUI_FONT_SIZE;
    DrawLineEx((Vector2) { .x = 0, .y = y }, (Vector2) { .x = w, .y = y }, 2, BROWN);
    float p = ((float)__fps__ / 120.0) * w;
    DrawCircle(p, y, 8, DARKGRAY);
}

void nnui_render_net_inputs_info()
{
    char buffer[120];
    char buffer2[120];

    for (size_t i = 0; i < __nnui__.layers[0].count; i++) {
        int y = (1.5 * NNUI_FONT_SIZE) + (i * NNUI_FONT_SIZE);
        if (i > 0)
            y += i * FONT_SPACE_BETWEEN;

        Vector2 p = GetWorldToScreen2D((Vector2) { .x = 0, .y = y }, __cam_net_inputs__);
        if (p.y < __cam_net_inputs__.offset.y)
            continue;
        if (p.y + (NNUI_FONT_SIZE / 2) > __inputs_panel_height__)
            break;

        snprintf(buffer, sizeof(buffer), "%zu", i);
        DrawText(buffer, __padding__, y + NNUI_FONT_SIZE / 4, NNUI_FONT_SIZE / 2, BLACK);

        snprintf(buffer, sizeof(buffer), "i: %s", mat_datatype_printf());
        snprintf(buffer2, sizeof(buffer2), buffer, *__nnui__.layers[0].neurons[i].o);
        DrawText(buffer2, 2 * NNUI_FONT_SIZE, y, NNUI_FONT_SIZE, BLACK);
    }

    for (size_t i = 0; i < __nnui__.layers[__nnui__.count - 1].count; i++) {
        int y = (1.5 * NNUI_FONT_SIZE) + (i * NNUI_FONT_SIZE);
        if (i > 0)
            y += i * FONT_SPACE_BETWEEN;

        Vector2 p = GetWorldToScreen2D((Vector2) { .x = 0, .y = y }, __cam_net_inputs__);
        if (p.y < __cam_net_inputs__.offset.y)
            continue;
        if (p.y + (NNUI_FONT_SIZE / 2) > __inputs_panel_height__)
            break;

        if (i >= __nnui__.layers[0].count) {
            snprintf(buffer, sizeof(buffer), "%zu", i);
            DrawText(buffer, __padding__, y + NNUI_FONT_SIZE / 4, NNUI_FONT_SIZE / 2, BLACK);
        }

        snprintf(buffer, sizeof(buffer), "o: %s", mat_datatype_printf());
        snprintf(buffer2, sizeof(buffer2), buffer, *__nnui__.layers[__nnui__.count - 1].neurons[i].o);
        DrawText(buffer2, (__inputs_panel_width__ / 2) + NNUI_FONT_SIZE, y, NNUI_FONT_SIZE, BLACK);
    }
}

void nnui_add_point_to_chart(DATA_TYPE point)
{
    if (__cost_current_point__ >= NNUI_COST_CHART_POINTS) {
        __cost_current_point__ = NNUI_COST_CHART_POINTS - 1;
        for (size_t i = 0; i < NNUI_COST_CHART_POINTS - 1; i++)
            __cost_points__[i] = __cost_points__[i + 1];
    }
    __max_cost__ = max(__max_cost__, point);
    __cost_points__[__cost_current_point__++] = point;
}

void nnui_render_cost_chart()
{
    DrawRectangle(0, __padding__, __right_panel_width__ - __padding__, __chart_panel_height__, RAYWHITE);
    DrawRectangle(__plot_area_offset__.x, __plot_area_offset__.y, __plot_area__.x, __plot_area__.y, WHITE);

    // Drawing axis
    DrawLineEx(
        (Vector2) { .x = __plot_area_offset__.x + 5, .y = __plot_area_offset__.y },
        (Vector2) { .x = __plot_area_offset__.x + 5, .y = __plot_area_offset__.y + __plot_area__.y },
        2.0,
        BLUE);

    DrawLineEx(
        (Vector2) { .x = __plot_area_offset__.x, .y = __plot_area_offset__.y + __plot_area__.y - 5 },
        (Vector2) { .x = __plot_area_offset__.x + __plot_area__.x, .y = __plot_area_offset__.y + __plot_area__.y - 5 },
        2.0,
        BLUE);

    if (__cost_current_point__ == 0)
        return;

    char buffer[120];
    char buffer2[120];

    snprintf(buffer, sizeof(buffer), "Current Cost: %s", mat_datatype_printf());
    snprintf(buffer2, sizeof(buffer2), buffer, __cost_points__[__cost_current_point__ - 1]);
    DrawText(buffer2, 3 * __padding__, 4 * __padding__, NNUI_FONT_SIZE, BLACK);

    Vector2 points[__cost_current_point__];
    float r = __max_cost__;
    float fr = (__plot_area__.y - 10);
    for (size_t i = 0; i < __cost_current_point__; i++) {
        float x = ((float)i / NNUI_COST_CHART_POINTS) * (__plot_area__.x - 10);
        x += __plot_area_offset__.x + 5;

        // Scaling coordinates to fit in chart area
        float y = (float)__cost_points__[i];
        points[i] = (Vector2) { .x = x, .y = y / r };
        points[i].y *= fr;
        points[i].y = __plot_area_offset__.y + 5 + (__plot_area__.y - 10) - points[i].y;

        // Drawing line from previous point to current
        if (i > 0)
            DrawLineV(points[i - 1], points[i], RED);
    }
}

void nnui_render_learning_rate()
{
    float w = __right_panel_width__ - (6 * __padding__);
    DrawLineEx(
        (Vector2) { .x = 0, .y = 35 },
        (Vector2) { .x = w, .y = 35 },
        2.0,
        BROWN);

    float p = *(__nnui__.lr) * w;
    DrawCircle(p, 35, 10, DARKGRAY);

    char buffer[120];
    char buffer2[120];
    snprintf(buffer, sizeof(buffer), "Learning Rate: %s", mat_datatype_printf());
    snprintf(buffer2, sizeof(buffer2), buffer, *(__nnui__.lr));
    DrawText(buffer2, __padding__, __padding__, NNUI_FONT_SIZE * 0.75, BLACK);
}

void nnui_render_neuron_info()
{
    DrawRectangle(0, 0, __right_panel_width__ - __padding__, __chart_panel_height__ - __padding__, RAYWHITE);
    if (__selected_neuron__.layer == INT32_MAX)
        return;

    char buffer[120];
    char buffer2[120];

    snprintf(buffer, sizeof(buffer), "Layer: %zu", __selected_neuron__.layer);
    DrawText(buffer, 10, FONT_SPACE_BETWEEN, NNUI_FONT_SIZE, BLACK);
    snprintf(buffer, sizeof(buffer), "Neuron: %zu", __selected_neuron__.index);
    DrawText(buffer, 10, (1 * NNUI_FONT_SIZE) + FONT_SPACE_BETWEEN, NNUI_FONT_SIZE, BLACK);
    snprintf(buffer, sizeof(buffer), "Output: %s\n", mat_datatype_printf());
    snprintf(buffer2, sizeof(buffer2), buffer, *__selected_neuron__.o);
    DrawText(buffer2, 10, (3 * NNUI_FONT_SIZE) + FONT_SPACE_BETWEEN, NNUI_FONT_SIZE, BLACK);

    snprintf(buffer, sizeof(buffer), "Bias: %s\n", mat_datatype_printf());
    if (__selected_neuron__.b != NULL)
        snprintf(buffer2, sizeof(buffer2), buffer, *__selected_neuron__.b);
    else
        snprintf(buffer2, sizeof(buffer2), buffer, 0.0);
    DrawText(buffer2, __right_panel_width__ / 2, (3 * NNUI_FONT_SIZE) + FONT_SPACE_BETWEEN, NNUI_FONT_SIZE, BLACK);

    int y = __cam_neuron_iw_info_offset__.y - __chart_panel_height__;
    DrawRectangle(
        __padding__ * 1.5,
        y,
        __right_panel_width__ - 3 * __padding__,
        __chart_panel_height__ - y - 2 * __padding__,
        WHITE);
}

void nnui_render_neuron_iw_info()
{
    if (__selected_neuron__.layer == INT32_MAX)
        return;

    int area_h = NNUI_HEIGHT - __cam_neuron_iw_info_offset__.y;

    char buffer[120];
    char buffer2[120];
    snprintf(buffer, sizeof(buffer), ": %s\n", mat_datatype_printf());
    snprintf(buffer2, sizeof(buffer2), buffer, *__selected_neuron__.o);
    if (__selected_neuron__.layer > 0) {
        for (size_t n = 0; n < __nnui__.layers[__selected_neuron__.layer - 1].count; n++) {
            int y = (n * NNUI_FONT_SIZE);
            if (n > 0)
                y += (n * FONT_SPACE_BETWEEN);
            Vector2 p = GetWorldToScreen2D((Vector2) { .x = 10, .y = y }, __cam_neuron_iw_info__);
            if (p.y + NNUI_FONT_SIZE < __cam_neuron_iw_info__.offset.y)
                continue;
            if (p.y > __cam_neuron_iw_info__.offset.y + area_h)
                break;

            if (n % 2 != 0)
                DrawRectangle(__padding__ * 1.5, y - (FONT_SPACE_BETWEEN / 2) - 2, __right_panel_width__ - 3 * __padding__, NNUI_FONT_SIZE + FONT_SPACE_BETWEEN, LIGHTGRAY);

            snprintf(buffer2, sizeof(buffer2), "%zu", n);
            DrawText(buffer2, 2 * __padding__, y + (NNUI_FONT_SIZE / 4), NNUI_FONT_SIZE / 2, BLACK);

            snprintf(buffer, sizeof(buffer), "i: %s\n", mat_datatype_printf());
            snprintf(buffer2, sizeof(buffer2), buffer, *(__nnui__.layers[__selected_neuron__.layer - 1].neurons[n].o));
            DrawText(buffer2, NNUI_FONT_SIZE * 2, y, NNUI_FONT_SIZE, BLACK);

            snprintf(buffer, sizeof(buffer), "w: %s\n", mat_datatype_printf());
            snprintf(buffer2, sizeof(buffer2), buffer, MAT_AT(__selected_neuron__.w, n, 0));
            DrawText(buffer2, __right_panel_width__ / 2, y, NNUI_FONT_SIZE, BLACK);
        }
    }
}

void nnui_set_status_message(char message[256])
{
    snprintf(__status__, sizeof(__status__), "%s", message);
}

void nnui_render_status_bar()
{
    DrawRectangle(0, 0, __left_panel_width__ - (__padding__), NNUI_FONT_SIZE + __padding__, LIGHTGRAY);
    DrawText(__status__, __padding__, __padding__, 0.75 * NNUI_FONT_SIZE, BLACK);
}

void nnui_reset_cam()
{
    __cam_graph__.zoom = 1.0f;
    __cam_graph__.offset = __cam_graph_offset__;
    __cam_graph__.target = __cam_graph_offset__;
}

bool nnui_was_key_pressed(int key)
{
    return IsKeyPressed(key);
}

void nnui_mouse_in_graph(Vector2 mousePos)
{
    if (mousePos.x >= __cam_fps_offset__.x && mousePos.x < 200
        && mousePos.y >= __cam_fps_offset__.y && mousePos.y <= 2.5 * NNUI_FONT_SIZE)
        return;

    if (mousePos.x >= __cam_net_inputs_offset__.x && mousePos.x < __cam_net_inputs_offset__.x + __inputs_panel_width__
        && mousePos.y >= __cam_net_inputs_offset__.y && mousePos.y <= __cam_net_inputs_offset__.y + __inputs_panel_height__)
        return;

    // Zoom and drag in NN area
    if (mousePos.x >= 0 && mousePos.x <= __left_panel_width__ && mousePos.y >= 0 && mousePos.y <= NNUI_HEIGHT) {
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Vector2 delta = GetMouseDelta();
            delta = Vector2Scale(delta, -1.0f / __cam_graph__.zoom);

            __cam_graph__.target = Vector2Add(__cam_graph__.target, delta);
        }

        float wheel = GetMouseWheelMove();
        if (wheel != 0) {
            Vector2 mouseWorldPos = GetScreenToWorld2D(GetMousePosition(), __cam_graph__);
            __cam_graph__.offset = GetMousePosition();
            __cam_graph__.target = mouseWorldPos;

            const float zoomIncrement = 0.025f;
            __cam_graph__.zoom += (wheel * zoomIncrement);
            if (__cam_graph__.zoom < zoomIncrement)
                __cam_graph__.zoom = zoomIncrement;
        }
    }
}

void nnui_mouse_in_neuron_iw(Vector2 mousePos)
{
    // Scroll for area with ins/weights information
    if (mousePos.x >= __left_panel_width__ && mousePos.x < NNUI_WIDTH
        && mousePos.y >= __cam_neuron_iw_info_offset__.y && mousePos.y <= NNUI_HEIGHT) {
        float wheel = GetMouseWheelMove();
        if (wheel != 0) {
            __cam_neuron_iw_info__.target.y = __cam_neuron_iw_info__.target.y - (wheel * 2.0);

            if (__cam_neuron_iw_info__.target.y < 0)
                __cam_neuron_iw_info__.target.y = 0;
            float max_scroll = (__selected_neuron__.w.rows - 1) * (FONT_SPACE_BETWEEN + NNUI_FONT_SIZE);
            if (max_scroll <= 6 * NNUI_FONT_SIZE)
                max_scroll = 0;
            if (__selected_neuron__.layer > 0 && __cam_neuron_iw_info__.target.y > max_scroll)
                __cam_neuron_iw_info__.target.y = max_scroll;
        }
    }
}

void nnui_mouse_in_net_inputs(Vector2 mousePos)
{
    // Scroll for area with ins/weights information
    if (mousePos.x >= __cam_net_inputs_offset__.x && mousePos.x < __cam_net_inputs_offset__.x + __inputs_panel_width__
        && mousePos.y >= __cam_net_inputs_offset__.y && mousePos.y <= __cam_net_inputs_offset__.y + __inputs_panel_height__) {
        float wheel = GetMouseWheelMove();
        if (wheel != 0) {
            __cam_net_inputs__.target.y = __cam_net_inputs__.target.y - (wheel * 2.0);
            if (__cam_net_inputs__.target.y < 0)
                __cam_net_inputs__.target.y = 0;
            float max_scroll = (__nnui__.layers[0].count - 1) * (FONT_SPACE_BETWEEN + NNUI_FONT_SIZE);
            if (max_scroll <= NNUI_FONT_SIZE)
                max_scroll = 0;
            if (__cam_net_inputs__.target.y > max_scroll)
                __cam_net_inputs__.target.y = max_scroll;
        }
    }
}

void nnui_mouse_in_fps(Vector2 mousePos)
{
    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        return;

    float w = __fps_bar_width__;
    Vector2 pos = GetScreenToWorld2D(mousePos, __cam_fps__);
    if (mousePos.y >= __cam_fps_offset__.y + NNUI_FONT_SIZE && mousePos.y <= __cam_fps_offset__.y + (2 * NNUI_FONT_SIZE)) {
        if (mousePos.x >= __cam_fps_offset__.x && mousePos.x < __cam_fps_offset__.x + w)
            __fps__ = (pos.x / w) * 120;
        else if (pos.x < 0)
            __fps__ = 0;
        else if (pos.x > w && pos.x < __fps_panel_width__)
            __fps__ = 120;
    }

    SetTargetFPS(__fps__);
}

void nnui_mouse_in_example(Vector2 mousePos)
{
    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        return;

    float w = __example_bar_width__;
    Vector2 pos = GetScreenToWorld2D(mousePos, __cam_example__);
    if (mousePos.y >= __cam_example_offset__.y + NNUI_FONT_SIZE && mousePos.y <= __cam_example_offset__.y + (2 * NNUI_FONT_SIZE)) {
        if (mousePos.x >= __cam_example_offset__.x && mousePos.x < __cam_example_offset__.x + w)
            __current_example__ = (pos.x / w) * __n_examples__;
        else if (pos.x < 0)
            __current_example__ = 0;
        else if (pos.x > w && pos.x < __example_panel_width__)
            __current_example__ = __n_examples__;

        SetTargetFPS(__fps__);
    }
}

void nnui_mouse_in_learning_rate(Vector2 mousePos)
{
    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        return;

    float w = __right_panel_width__ - (6 * __padding__);
    Vector2 pos = GetScreenToWorld2D(mousePos, __cam_learning_rate__);
    if (mousePos.x >= __cam_learning_rate_offset__.x && mousePos.x < __cam_learning_rate_offset__.x + w
        && mousePos.y >= __cam_learning_rate_offset__.y + 20 && mousePos.y <= __cam_learning_rate_offset__.y + 45)
        *__nnui__.lr = pos.x / w;
    else if (pos.x < 0 && pos.x > -10)
        *__nnui__.lr = 0.0f;
    else if (pos.x > w)
        *__nnui__.lr = 1.0f;
}

void nnui_render()
{
    assert(__nnui_ready__ == 1 && "You must initialize the render first => nnui_init(NN nn, size_t n_examples, int fps)");

    Vector2 mousePos = GetMousePosition();
    nnui_mouse_in_graph(mousePos);
    nnui_mouse_in_net_inputs(mousePos);
    nnui_mouse_in_neuron_iw(mousePos);
    nnui_mouse_in_fps(mousePos);
    nnui_mouse_in_example(mousePos);
    nnui_mouse_in_learning_rate(mousePos);

    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        mousePos = (Vector2) { .x = 0, .y = 0 };

    BeginDrawing();
    // ClearBackground(LIGHTGRAY);
    DrawRectangle(0, 0, NNUI_WIDTH, NNUI_HEIGHT, LIGHTGRAY);

    DrawRectangle(__padding__, __padding__, NNUI_WIDTH - (2 * __padding__), NNUI_HEIGHT - (2 * __padding__), (Color) { 45, 45, 45, 255 });

    BeginMode2D(__cam_graph__);
    nnui_render_graph(mousePos);
    EndMode2D();

    BeginMode2D(__cam_cost_chart__);
    nnui_render_cost_chart();
    EndMode2D();

    BeginMode2D(__cam_neuron_info__);
    nnui_render_neuron_info();
    EndMode2D();

    BeginMode2D(__cam_neuron_iw_info__);
    nnui_render_neuron_iw_info();
    EndMode2D();

    BeginMode2D(__cam_neuron_info__);
    if (__selected_neuron__.layer != INT32_MAX) {
        int y = __cam_neuron_iw_info_offset__.y - __chart_panel_height__;
        DrawRectangle(__padding__ * 1.5, y - 1.5 * NNUI_FONT_SIZE, __right_panel_width__ - 3 * __padding__, 1.4 * NNUI_FONT_SIZE, GRAY);
        DrawRectangle(__padding__ * 1.5, __chart_panel_height__ - 2 * __padding__, __right_panel_width__ - 3 * __padding__, __padding__, RAYWHITE);
        DrawRectangle(__padding__ * 1.5, __chart_panel_height__ - __padding__, __right_panel_width__ - 3 * __padding__, __padding__, LIGHTGRAY);
        DrawText("Inputs", NNUI_FONT_SIZE * 2, y - 1.25 * NNUI_FONT_SIZE, NNUI_FONT_SIZE, BLACK);
        DrawText("Weights", (__right_panel_width__ / 2), y - 1.25 * NNUI_FONT_SIZE, NNUI_FONT_SIZE, BLACK);
    }
    EndMode2D();

    DrawRectangle(
        __cam_net_inputs_offset__.x,
        __cam_net_inputs_offset__.y,
        __inputs_panel_width__,
        __inputs_panel_height__,
        (Color) { 225, 225, 225, 180 });

    BeginMode2D(__cam_net_inputs__);
    nnui_render_net_inputs_info();
    EndMode2D();
    DrawRectangle(__cam_net_inputs_offset__.x, __cam_net_inputs_offset__.y, __inputs_panel_width__, NNUI_FONT_SIZE + (2 * __padding__), LIGHTGRAY);
    DrawText("Inputs", __cam_net_inputs_offset__.x + (2 * NNUI_FONT_SIZE), __cam_net_inputs_offset__.y + __padding__, NNUI_FONT_SIZE, BLACK);
    DrawText("Outputs", __cam_net_inputs_offset__.x + (__inputs_panel_width__ / 2) + NNUI_FONT_SIZE, __cam_net_inputs_offset__.y + __padding__, NNUI_FONT_SIZE, BLACK);

    BeginMode2D(__cam_fps__);
    nnui_render_fps();
    EndMode2D();

    BeginMode2D(__cam_example__);
    nnui_render_example();
    EndMode2D();

    BeginMode2D(__cam_learning_rate__);
    nnui_render_learning_rate();
    EndMode2D();

    BeginMode2D(__cam_status_bar__);
    nnui_render_status_bar();
    EndMode2D();

    DrawRectangle(__left_panel_width__ - (__padding__ / 2), 0, __padding__, NNUI_HEIGHT, LIGHTGRAY);
    DrawRectangle(__left_panel_width__, __chart_panel_height__ - (__padding__ / 2), __right_panel_width__, __padding__, LIGHTGRAY);

    if (__draw_test__) {
        BeginMode2D(__cam_test_area__);
        nnui_render_test_area();
        EndMode2D();
    }
    EndDrawing();
}
