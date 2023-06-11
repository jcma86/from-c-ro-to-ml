#include <stdio.h>
#include <time.h>

#include "../libs/nn.h"

int main()
{
    srand(50);
    // srand(time(0));
    randv();

    DATA_TYPE xor_data[4][3] = {
        { 0, 0, 0 },
        { 0, 1, 1 },
        { 1, 0, 1 },
        { 1, 1, 0 },
    };
    Mat m_xor; // = mat_create(4, 3);
    m_xor.rows = 4;
    m_xor.cols = 3;
    m_xor.stride = 3;
    m_xor.data = xor_data[0];

    MAT_PRINT(m_xor);

    Mat m_in = {
        .rows = 4,
        .cols = 2,
        .stride = 3,
        .data = &MAT_AT(m_xor, 0, 0),
    };

    Mat m_out = {
        .rows = 4,
        .cols = 1,
        .stride = 3,
        .data = &MAT_AT(m_xor, 0, m_in.cols),
    };

    MAT_PRINT(m_in);
    MAT_PRINT(m_out);

    size_t layers[] = { 2, 2, 1 };
    NN nn = nn_create(layers, 3);
    NN nnd = nn_create(layers, 3);
    nn_rand(nn, -1.0, 1.0);

    for (size_t i = 0; i < 100000; i++) {
        nn_finite_diff(nn, nnd, m_in, m_out, 1e-1);
        nn_update_params(nn, nnd, 1e-1);
        printf("%f\n", nn_cost(nn, m_in, m_out));
    }

    for (size_t x1 = 0; x1 < 2; x1++) {
        for (size_t x2 = 0; x2 < 2; x2++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = x1;
            MAT_AT(NN_INPUT(nn), 0, 1) = x2;
            nn_forward(nn);
            printf("%zu xor %zu = %f\n", x1, x2, MAT_AT(NN_OUTPUT(nn), 0, 0));
        }
    }

    return 0;
}
