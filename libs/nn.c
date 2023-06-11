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

// NN => XOR {2, 2, 1} (2 Ins, 2 Hidden Neu, 1 Out)
NN nn_create(size_t* layers, size_t count)
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

    nn.o[0] = mat_create(1, layers[0]);
    mat_set(nn.o[0], 0);
    for (size_t i = 1; i < count; i++) {
        nn.w[i - 1] = mat_create(layers[i - 1], layers[i]);
        nn.b[i - 1] = mat_create(1, layers[i]);
        nn.o[i] = mat_create(1, layers[i]);
    }

    return nn;
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

DATA_TYPE nn_cost(NN nn, Mat m_in, Mat m_out)
{
    assert(m_in.rows == m_out.rows);
    assert(m_out.cols == NN_OUTPUT(nn).cols);

    size_t ne = m_in.rows;
    DATA_TYPE err = 0;

    for (size_t i = 0; i < ne; i++) {
        Mat x = mat_row(m_in, i);
        Mat y = mat_row(m_out, i);

        mat_cpy(NN_INPUT(nn), x);
        nn_forward(nn);

        for (size_t o = 0; o < m_out.cols; o++) {
            DATA_TYPE d = MAT_AT(NN_OUTPUT(nn), 0, o) - MAT_AT(y, 0, o);
            err += d * d;
        }
    }

    return err / ne;
}

void nn_finite_diff(NN nn, NN nnd, Mat m_in, Mat m_out, DATA_TYPE eps)
{
    DATA_TYPE tmp;
    DATA_TYPE cc = nn_cost(nn, m_in, m_out);

    for (size_t i = 0; i < nn.count; i++) {
        for (size_t r = 0; r < nn.w[i].rows; r++) {
            for (size_t c = 0; c < nn.w[i].cols; c++) {
                tmp = MAT_AT(nn.w[i], r, c);
                MAT_AT(nn.w[i], r, c) += eps;
                MAT_AT(nnd.w[i], r, c) = (nn_cost(nn, m_in, m_out) - cc) / eps;
                MAT_AT(nn.w[i], r, c) = tmp;
            }
        }

        for (size_t r = 0; r < nn.b[i].rows; r++) {
            for (size_t c = 0; c < nn.b[i].cols; c++) {
                tmp = MAT_AT(nn.b[i], r, c);
                MAT_AT(nn.b[i], r, c) += eps;
                MAT_AT(nnd.b[i], r, c) = (nn_cost(nn, m_in, m_out) - cc) / eps;
                MAT_AT(nn.b[i], r, c) = tmp;
            }
        }
    }
}

void nn_update_params(NN nn, NN nnd, DATA_TYPE learning_rate)
{
    for (size_t i = 0; i < nn.count; i++) {
        for (size_t r = 0; r < nn.w[i].rows; r++) {
            for (size_t c = 0; c < nn.w[i].cols; c++) {
                MAT_AT(nn.w[i], r, c) -= learning_rate * MAT_AT(nnd.w[i], r, c);
            }
        }

        for (size_t r = 0; r < nn.b[i].rows; r++) {
            for (size_t c = 0; c < nn.b[i].cols; c++) {
                MAT_AT(nn.b[i], r, c) -= learning_rate * MAT_AT(nnd.b[i], r, c);
            }
        }
    }
}
