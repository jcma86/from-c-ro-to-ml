#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float training_or[][3] = {
    { 0.0, 0.0, 0.0 },
    { 0.0, 1.0, 1.0 },
    { 1.0, 0.0, 1.0 },
    { 1.0, 1.0, 1.0 },
};

float training_and[][3] = {
    { 0.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0 },
    { 1.0, 0.0, 0.0 },
    { 1.0, 1.0, 1.0 },
};

float training_nand[][3] = {
    { 0.0, 0.0, 1.0 },
    { 0.0, 1.0, 1.0 },
    { 1.0, 0.0, 1.0 },
    { 1.0, 1.0, 0.0 },
};

int n_training = sizeof(training_or) / sizeof(training_or[0]);

float randf(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float cost(float w1, float w2, float b)
{
    float sum = 0.0;
    for (int i = 0; i < n_training; i++) {
        float x1 = training_or[i][0];
        float x2 = training_or[i][1];
        float y = sigmoidf(x1 * w1 + x2 * w2 + b);
        float d = y - training_or[i][2];
        sum += d * d;
    }
    sum /= n_training;

    return sum;
}

int main()
{
    srand(25);
    // srand(time(0));
    float eps = 1e-3;
    float learning_rate = 1e-1;

    float w1 = randf();
    float w2 = randf();
    float b = randf();

    w1 = randf();
    w2 = randf();
    b = randf();

    for (int i = 0; i < 100000; i++) {
        float c = cost(w1, w2, b);
        float dw1 = (cost(w1 + eps, w2, b) - c) / eps;
        float dw2 = (cost(w1, w2 + eps, b) - c) / eps;
        float db = (cost(w1, w2, b + eps) - c) / eps;
        w1 -= learning_rate * dw1;
        w2 -= learning_rate * dw2;
        b -= learning_rate * db;

        printf("w1: %f, w2: %f, b: %f => c: %f\n", w1, w2, b, cost(w1, w2, b));
    }

    for (int x1 = 0; x1 < 2; x1++)
        for (int x2 = 0; x2 < 2; x2++)
            printf("%d | %d => %f\n", x1, x2, sigmoidf(x1 * w1 + x2 * w2 + b));

    return 0;
}
