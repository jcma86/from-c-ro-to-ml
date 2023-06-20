#include <stdio.h>
#include <time.h>

#include "../libs/nn.h"

int main()
{
    srand(50);
    // srand(time(0));
    randv();

    DATA_TYPE xor_data[4][9] = {
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 1, 0, 1, 1, 0, 1, 1 },
        { 1, 0, 1, 1, 0, 1, 1, 0, 1 },
        { 1, 1, 0, 1, 1, 0, 1, 1, 0 },
    };
    Mat m_xor; // = mat_create(4, 3);
    m_xor.rows = 4;
    m_xor.cols = 9;
    m_xor.stride = 9;
    m_xor.data = xor_data[0];

    MAT_PRINT(m_xor);

    Mat m_in = {
        .rows = ARRAY_SIZE(xor_data),
        .cols = 6,
        .stride = 9,
        .data = &MAT_AT(m_xor, 0, 0),
    };

    Mat m_out = {
        .cols = 3,
        .rows = m_in.rows,
        .stride = m_in.stride,
        .data = &MAT_AT(m_xor, 0, m_in.cols),
    };

    MAT_PRINT(m_in);
    MAT_PRINT(m_out);

    size_t layers[] = { 6, 5, 3 };
    size_t n_layers = ARRAY_SIZE(layers);
    DATA_TYPE learning_rate = 0.1;

    NN nn = nn_create(layers, n_layers, learning_rate);
    NN nnd = nn_create(layers, n_layers, learning_rate);
    nn_rand(nn, -1.0, 1.0);

    nnui_init(nn);
    bool paused = false;
    while (!nnui_should_close()) {
        for (size_t ex = 0; ex < m_in.rows; ex++) {
            if (!paused) {
                nn_finite_diff(nn, nnd, m_in, m_out, ex, 1, 1e-1);
                nn_update_params(nn, nnd);
            }

            nnui_render();
            if (nnui_was_key_pressed(KEY_R))
                nnui_reset_cam();
            if (nnui_was_key_pressed(KEY_P))
                paused = !paused;
        }
        if (!paused)
            nnui_add_point_to_chart(*nn.cost);
    }
    nnui_end();

    // for (size_t x1 = 0; x1 < 2; x1++) {
    //     for (size_t x2 = 0; x2 < 2; x2++) {
    //         MAT_AT(NN_INPUT(nn), 0, 0) = x1;
    //         MAT_AT(NN_INPUT(nn), 0, 1) = x2;
    //         nn_forward(nn);
    //         printf("%zu xor %zu = %f\n", x1, x2, MAT_AT(NN_OUTPUT(nn), 0, 0));
    //     }
    // }

    nn_destroy(nn);
    nn_destroy(nnd);

    return 0;
}
