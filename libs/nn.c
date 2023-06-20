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

DATA_TYPE sigmoid(DATA_TYPE x)
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

DATA_TYPE activation_handler(DATA_TYPE x, ActivationFx act)
{
    switch (act) {
    case ACTFX_LINEAR:
        return x;
    case ACTFX_SIGM:
        return sigmoid(x);
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

void mat_act(Mat m)
{
    for (size_t r = 0; r < m.rows; r++) {
        for (size_t c = 0; c < m.cols; c++) {
            MAT_AT(m, r, c) = activation_handler(MAT_AT(m, r, c), ACTFX_SIGM);
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
        for (size_t i = 0; i < nn.count; i++)
            mat_destroy(nn.b[i]);
        free(nn.b);
        nn.b = NULL;
    }
    if (nn.w != NULL) {
        for (size_t i = 0; i < nn.count; i++)
            mat_destroy(nn.w[i]);
        free(nn.w);
        nn.w = NULL;
    }
    if (nn.o != NULL) {
        for (size_t i = 0; i < nn.count + 1; i++)
            mat_destroy(nn.o[i]);
        free(nn.o);
        nn.o = NULL;
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

// NN => XOR {2, 2, 1} (2 Ins, 2 Hidden Neu, 1 Out)
NN nn_create(size_t* layers, size_t count, DATA_TYPE learning_rate)
{
    assert(count > 0);

    NN nn;
    nn.count = count - 1;

    nn.w = malloc(sizeof(Mat) * nn.count);
    assert(nn.w != NULL);
    nn.b = malloc(sizeof(Mat) * nn.count);
    assert(nn.b != NULL);
    nn.o = malloc(sizeof(Mat) * count);
    assert(nn.o != NULL);

    nn.cost = malloc(sizeof(DATA_TYPE));
    nn.lr = malloc(sizeof(DATA_TYPE));
    *nn.cost = 0.0;
    *nn.lr = min(max(0.0, learning_rate), 1.0);

    nn.o[0] = mat_create(1, layers[0]);
    mat_set(nn.o[0], 0);
    for (size_t i = 1; i < count; i++) {
        nn.w[i - 1] = mat_create(layers[i - 1], layers[i]);
        nn.b[i - 1] = mat_create(1, layers[i]);
        nn.o[i] = mat_create(1, layers[i]);
    }

    return nn;
}

Neuron nn_get_neuron(NN nn, size_t layer, size_t index)
{
    assert(layer <= nn.count);
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
    assert(index <= nn.count);

    Layer l;
    l.layer = index;
    l.count = nn.o[index].cols;
    l.neurons = malloc(sizeof(Neuron) * l.count);
    assert(l.neurons != NULL);

    for (size_t n = 0; n < l.count; n++)
        l.neurons[n] = nn_get_neuron(nn, index, n);

    if (index == 0)
        l.type = NN_LAYER_INPUT;
    else if (index == nn.count)
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

void nn_rand(NN nn, DATA_TYPE min, DATA_TYPE max)
{
    for (size_t i = 0; i < nn.count; i++) {
        mat_rand(nn.w[i], min, max);
        mat_rand(nn.b[i], min, max);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; i++) {
        mat_mult(nn.o[i + 1], nn.o[i], nn.w[i]);
        mat_sum(nn.o[i + 1], nn.b[i]);
        mat_act(nn.o[i + 1]);
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

    for (size_t i = 0; i < nn.count; i++) {
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

    *nn.cost = cc;
}

void nn_update_params(NN nn, NN nnd)
{
    for (size_t i = 0; i < nn.count; i++) {
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

// NN Visualizer
char __nnui_ready__ = 0;

Camera2D __cam_main__ = { 0 };
Camera2D __cam_graph__ = { 0 };
Camera2D __cam_net_inputs__ = { 0 };
Camera2D __cam_cost_chart__ = { 0 };
Camera2D __cam_learning_rate__ = { 0 };
Camera2D __cam_neuron_info__ = { 0 };
Camera2D __cam_neuron_iw_info__ = { 0 };

const int __padding__ = 5;
const int __right_panel_width__ = 350;
const int __left_panel_width__ = NNUI_WIDTH - __right_panel_width__;
const int __inputs_panel_width__ = 320;
const int __inputs_panel_height__ = 5.5 * NNUI_FONT_SIZE;
const int __chart_panel_height__ = NNUI_HEIGHT / 2;
Vector2 __cam_main_offset__ = (Vector2) { .x = 0.0, .y = 0.0 };
Vector2 __cam_graph_offset__ = (Vector2) { .x = 0.0, .y = 0.0 };
Vector2 __cam_net_inputs_offset__ = (Vector2) { .x = __left_panel_width__ - (__inputs_panel_width__ + __padding__), .y = 2 * __padding__ };
Vector2 __cam_cost_chart_offset__ = (Vector2) { .x = __left_panel_width__, .y = 0.0 };
Vector2 __cam_learning_rate_offset__ = (Vector2) { .x = __left_panel_width__ + (3 * __padding__), .y = __chart_panel_height__ - 50 };
Vector2 __cam_neuron_info_offset__ = (Vector2) { .x = __left_panel_width__, .y = __chart_panel_height__ };
Vector2 __cam_neuron_iw_info_offset__ = (Vector2) { .x = __left_panel_width__, .y = __chart_panel_height__ + (6 * NNUI_FONT_SIZE) };

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

void nnui_init(NN nn)
{
    InitWindow(NNUI_WIDTH, NNUI_HEIGHT, "Neural Network Visualizer");
    if (NNUI_FPS > 0)
        SetTargetFPS(NNUI_FPS);

    __cam_main__.offset = __cam_main_offset__;
    __cam_graph__.offset = __cam_graph_offset__;
    __cam_net_inputs__.offset = __cam_net_inputs_offset__;
    __cam_cost_chart__.offset = __cam_cost_chart_offset__;
    __cam_learning_rate__.offset = __cam_learning_rate_offset__;
    __cam_neuron_info__.offset = __cam_neuron_info_offset__;
    __cam_neuron_iw_info__.offset = __cam_neuron_iw_info_offset__;

    __cam_main__.zoom = 1.0f;
    __cam_graph__.zoom = 1.0f;
    __cam_net_inputs__.zoom = 1.0f;
    __cam_cost_chart__.zoom = 1.0f;
    __cam_learning_rate__.zoom = 1.0f;
    __cam_neuron_info__.zoom = 1.0f;
    __cam_neuron_iw_info__.zoom = 1.0f;

    __nnui__.count = nn.count + 1;
    __nnui__.lr = nn.lr;
    __nnui__.layers = malloc(sizeof(Layer) * __nnui__.count);
    assert(__nnui__.layers != NULL);

    for (size_t l = 0; l <= nn.count; l++)
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
        space_neuron = min(space_neuron, 20 * NNUI_NEURON_RADIUS);
        space_neuron = max(space_neuron, 5 * NNUI_NEURON_RADIUS);
        y_offset -= space_neuron * (__nnui__.layers[l].count - 1) / 2;

        for (size_t n = 0; n < __nnui__.layers[l].count; n++) {
            float y = (n * space_neuron) + y_offset;
            __nnui__.ui[l].coords[n] = (Vector2) { .x = x, .y = y };
        }
    }

    __nnui_ready__ = 1;
}

void nnui_render_graph(Vector2 mouse_pos)
{
    DrawCircle(__left_panel_width__ / 2, NNUI_HEIGHT / 2, 2, BLACK);

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
                else
                    DrawCircle(p.x, p.y, NNUI_NEURON_RADIUS, BLUE);
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

                    DrawLineEx(pn, p, min(__cam_graph__.zoom, 0.8), BLACK);
                }
            }
            if (l < __nnui__.count - 1) {
                p.x += l == 0 ? NNUI_NEURON_RADIUS : 2 * NNUI_NEURON_RADIUS;
                for (size_t i = 0; i < __nnui__.layers[l + 1].count; i++) {
                    Vector2 pn = __nnui__.ui[l + 1].coords[i];
                    pn.x -= NNUI_NEURON_RADIUS;
                    DrawLineEx(pn, p, __cam_graph__.zoom, BLACK);
                }
            }
        }
    }
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
    for (size_t i = 0; i < __cost_current_point__; i++) {
        float x = ((float)i / NNUI_COST_CHART_POINTS) * (__plot_area__.x - 10);
        x += __plot_area_offset__.x + 5;

        float y = (float)__cost_points__[i];
        points[i] = (Vector2) { .x = x, .y = y };
    }

    float fr = (__plot_area__.y - 10);
    for (size_t i = 0; i < __cost_current_point__; i++) {
        float r = __max_cost__;
        points[i].y = points[i].y / r;
        points[i].y *= fr;
        points[i].y = __plot_area_offset__.y + 5 + (__plot_area__.y - 10) - points[i].y;
        DrawPixelV(points[i], RED);
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
            if (max_scroll <= 6 * NNUI_FONT_SIZE)
                max_scroll = 0;
            if (__cam_net_inputs__.target.y > max_scroll)
                __cam_net_inputs__.target.y = max_scroll;
        }
    }
}

void nnui_mouse_in_learning_rate(Vector2 mousePos)
{
    float w = __right_panel_width__ - (6 * __padding__);
    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        return;

    Vector2 pos = GetScreenToWorld2D(mousePos, __cam_learning_rate__);
    if (mousePos.x >= __cam_learning_rate_offset__.x && mousePos.x < __cam_learning_rate_offset__.x + w
        && mousePos.y >= __cam_learning_rate_offset__.y + 20 && mousePos.y <= __cam_learning_rate_offset__.y + 45)
        *__nnui__.lr = pos.x / w;
    else if (pos.x < 0)
        *__nnui__.lr = 0.0f;
    else if (pos.x > w)
        *__nnui__.lr = 1.0f;
}

void nnui_render()
{
    assert(__nnui_ready__ == 1);

    Vector2 mousePos = GetMousePosition();
    nnui_mouse_in_graph(mousePos);
    nnui_mouse_in_net_inputs(mousePos);
    nnui_mouse_in_neuron_iw(mousePos);
    nnui_mouse_in_learning_rate(mousePos);

    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        mousePos = (Vector2) { .x = 0, .y = 0 };

    BeginDrawing();
    ClearBackground(LIGHTGRAY);

    DrawRectangle(__padding__, __padding__, NNUI_WIDTH - (2 * __padding__), NNUI_HEIGHT - (2 * __padding__), RAYWHITE);

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

    BeginMode2D(__cam_learning_rate__);
    nnui_render_learning_rate();
    EndMode2D();

    DrawRectangle(__left_panel_width__ - (__padding__ / 2), 0, __padding__, NNUI_HEIGHT, LIGHTGRAY);
    DrawRectangle(__left_panel_width__, __chart_panel_height__ - (__padding__ / 2), __right_panel_width__, __padding__, LIGHTGRAY);

    EndDrawing();
}
