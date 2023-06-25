#include <stdio.h>
#include <time.h>

#include "../libs/nn.h"

int main()
{
    // srand(50);
    srand(time(0));
    randv();

    DATA_TYPE training_data[4][3] = {
        { 0, 0, 0 },
        { 0, 1, 1 },
        { 1, 0, 1 },
        { 1, 1, 0 },
    };

    Mat m_xor; // = mat_create(4, 3);
    m_xor.rows = sizeof(training_data) / sizeof(training_data[0]);
    m_xor.cols = sizeof(training_data[0]) / sizeof(DATA_TYPE);
    m_xor.stride = sizeof(training_data[0]) / sizeof(DATA_TYPE);
    m_xor.data = training_data[0];

    MAT_PRINT(m_xor);

    Mat m_in = {
        .rows = ARRAY_SIZE(training_data),
        .cols = 2,
        .stride = m_xor.stride,
        .data = &MAT_AT(m_xor, 0, 0),
    };

    Mat m_out = {
        .cols = m_xor.cols - m_in.cols,
        .rows = m_in.rows,
        .stride = m_in.stride,
        .data = &MAT_AT(m_xor, 0, m_in.cols),
    };

    MAT_PRINT(m_in);
    MAT_PRINT(m_out);

    size_t layers[] = { m_in.cols, 2, m_out.cols };
    size_t n_layers = ARRAY_SIZE(layers);
    DATA_TYPE eps = 0.001;
    DATA_TYPE learning_rate = 0.1;

    NN nn = nn_create(layers, n_layers, learning_rate);
    NN nnd = nn_create(layers, n_layers, learning_rate);
    nn_rand(nn, -1.0, 1.0);

    nnui_init(nn, m_in.rows, 0);
    bool paused = false;
    size_t batch_size = 1;
    nnui_set_status_message(paused ? "Training paused..." : "Training!");
    while (!nnui_should_close()) {
        if (!paused) {
            for (size_t ex = 0; ex < m_in.rows; ex += batch_size) {
                nn_finite_diff(nn, nnd, m_in, m_out, ex, batch_size, eps);
                nn_update_params(nn, nnd);
            }
        }

        mat_cpy(NN_INPUT(nn), mat_row(m_in, nnui_get_current_example()));
        nn_forward(nn);
        nnui_render();
        if (nnui_was_key_pressed(KEY_R))
            nnui_reset_cam();
        if (nnui_was_key_pressed(KEY_P)) {
            paused = !paused;
            nnui_set_status_message(paused ? "Training paused..." : "Training!");
        }

        if (!paused)
            nnui_add_point_to_chart(*nn.cost);
    }
    nnui_end();

    for (size_t x1 = 0; x1 < 2; x1++) {
        for (size_t x2 = 0; x2 < 2; x2++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = x1;
            MAT_AT(NN_INPUT(nn), 0, 1) = x2;
            nn_forward(nn);
            printf("%zu xor %zu = %f\n", x1, x2, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }
    }

    nn_destroy(nn);
    nn_destroy(nnd);

    return 0;
}
