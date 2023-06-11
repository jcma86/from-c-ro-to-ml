#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef float sample[3];

typedef struct {
    float a_w1;
    float a_w2;
    float a_b;

    float b_w1;
    float b_w2;
    float b_b;

    float c_w1;
    float c_w2;
    float c_b;
} model;

sample training_or[] = {
    { 0.0, 0.0, 0.0 },
    { 0.0, 1.0, 1.0 },
    { 1.0, 0.0, 1.0 },
    { 1.0, 1.0, 1.0 },
};

sample training_and[] = {
    { 0.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0 },
    { 1.0, 0.0, 0.0 },
    { 1.0, 1.0, 1.0 },
};

sample training_nand[] = {
    { 0.0, 0.0, 1.0 },
    { 0.0, 1.0, 1.0 },
    { 1.0, 0.0, 1.0 },
    { 1.0, 1.0, 0.0 },
};

sample training_xor[] = {
    { 0.0, 0.0, 0.0 },
    { 0.0, 1.0, 1.0 },
    { 1.0, 0.0, 1.0 },
    { 1.0, 1.0, 0.0 },
};

sample* training_set = training_xor;
int n_training = 4;

float randf(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float forward(model m, float x1, float x2)
{
    float o_a = sigmoidf(x1 * m.a_w1 + x2 * m.a_w2 + m.a_b);
    float o_b = sigmoidf(x1 * m.b_w1 + x2 * m.b_w2 + m.b_b);

    return sigmoidf(o_a * m.c_w1 + o_b * m.c_w2 + m.c_b);
}

float cost(model m)
{
    float sum = 0.0;
    for (int i = 0; i < n_training; i++) {
        float x1 = training_set[i][0];
        float x2 = training_set[i][1];
        float y = forward(m, x1, x2);
        float d = y - training_set[i][2];
        sum += d * d;
    }
    sum /= n_training;

    return sum;
}

model init_model(void)
{
    model m;
    m.a_w1 = randf();
    m.a_w2 = randf();
    m.a_b = randf();

    m.b_w1 = randf();
    m.b_w2 = randf();
    m.b_b = randf();

    m.c_w1 = randf();
    m.c_w2 = randf();
    m.c_b = randf();

    return m;
}

model compute_deltas(model m, float eps)
{
    model dm;
    float c = cost(m);
    float tmp;

    tmp = m.a_w1;
    m.a_w1 += eps;
    dm.a_w1 = (cost(m) - c) / eps;
    m.a_w1 = tmp;
    tmp = m.a_w2;
    m.a_w2 += eps;
    dm.a_w2 = (cost(m) - c) / eps;
    m.a_w2 = tmp;
    tmp = m.a_b;
    m.a_b += eps;
    dm.a_b = (cost(m) - c) / eps;
    m.a_b = tmp;

    tmp = m.b_w1;
    m.b_w1 += eps;
    dm.b_w1 = (cost(m) - c) / eps;
    m.b_w1 = tmp;
    tmp = m.b_w2;
    m.b_w2 += eps;
    dm.b_w2 = (cost(m) - c) / eps;
    m.b_w2 = tmp;
    tmp = m.b_b;
    m.b_b += eps;
    dm.b_b = (cost(m) - c) / eps;
    m.b_b = tmp;

    tmp = m.c_w1;
    m.c_w1 += eps;
    dm.c_w1 = (cost(m) - c) / eps;
    m.c_w1 = tmp;
    tmp = m.c_w2;
    m.c_w2 += eps;
    dm.c_w2 = (cost(m) - c) / eps;
    m.c_w2 = tmp;
    tmp = m.c_b;
    m.c_b += eps;
    dm.c_b = (cost(m) - c) / eps;
    m.c_b = tmp;

    return dm;
}

model update_weights(model m, model dm, float learning_rate)
{
    m.a_w1 -= learning_rate * dm.a_w1;
    m.a_w2 -= learning_rate * dm.a_w2;
    m.a_b -= learning_rate * dm.a_b;

    m.b_w1 -= learning_rate * dm.b_w1;
    m.b_w2 -= learning_rate * dm.b_w2;
    m.b_b -= learning_rate * dm.b_b;

    m.c_w1 -= learning_rate * dm.c_w1;
    m.c_w2 -= learning_rate * dm.c_w2;
    m.c_b -= learning_rate * dm.c_b;

    return m;
}

int main()
{
    // srand(25);
    srand(time(0));
    randf();

    float eps = 1e-1;
    float learning_rate = 1e-1;

    model m = init_model();

    for (int i = 0; i < 100000; i++) {
        model dm = compute_deltas(m, eps);
        m = update_weights(m, dm, learning_rate);
        printf("%f\n", cost(m));

        // printf("c: %f\n", cost(m));
    }

    // printf("================\n");
    // printf("XOR\n");
    // for (int x1 = 0; x1 < 2; x1++)
    //     for (int x2 = 0; x2 < 2; x2++)
    //         printf("%d xor %d => %f\n", x1, x2, forward(m, x1, x2));
    // printf("================\n");
    // printf("Neuron A:\n");
    // for (int x1 = 0; x1 < 2; x1++)
    //     for (int x2 = 0; x2 < 2; x2++)
    //         printf("%d ? %d => %f\n", x1, x2, sigmoidf(x1 * m.a_w1 + x2 * m.a_w2 + m.a_b));
    // printf("================\n");
    // printf("Neuron B:\n");
    // for (int x1 = 0; x1 < 2; x1++)
    //     for (int x2 = 0; x2 < 2; x2++)
    //         printf("%d ? %d => %f\n", x1, x2, sigmoidf(x1 * m.b_w1 + x2 * m.b_w2 + m.b_b));
    // printf("================\n");
    // printf("Neuron C:\n");
    // for (int x1 = 0; x1 < 2; x1++)
    //     for (int x2 = 0; x2 < 2; x2++)
    //         printf("%d ? %d => %f\n", x1, x2, sigmoidf(x1 * m.c_w1 + x2 * m.c_w2 + m.c_b));

    return 0;
}
