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

int n_training = sizeof(training_and) / sizeof(training_and[0]);

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
        float x1 = training_and[i][0];
        float x2 = training_and[i][1];
        float y = sigmoidf(x1 * w1 + x2 * w2 + b);
        float d = y - training_and[i][2];
        sum += d * d;
    }
    sum /= n_training;

    return sum;
}

void derivative_cost(float w1, float w2, float b, float* dw1, float* dw2, float* db)
{
    *dw1 = 0;
    *dw2 = 0;
    *db = 0;
    for (int i = 0; i < n_training; i++) {
        float x1 = training_and[i][0];
        float x2 = training_and[i][1];
        float y = training_and[i][2];
        float z = (x1 * w1) + (x2 * w2) + b;
        float sig = sigmoidf(z);

        *dw1 += 2 * (sig - y) * (sig * (1 - sig)) * x1;
        *dw2 += 2 * (sig - y) * (sig * (1 - sig)) * x2;
        *db += 2 * (sig - y) * (sig * (1 - sig));
    }
    *dw1 /= n_training;
    *dw2 /= n_training;
    *db /= n_training;
}

int main()
{
    srand(25);
    // srand(time(0));
    // float eps = 1e-3;
    float learning_rate = 1e-1;

    float w1 = randf();
    float w2 = randf();
    float b = randf();

    w1 = randf();
    w2 = randf();
    b = randf();

    for (int i = 0; i < 100000; i++) {
        float dw1;
        float dw2;
        float db;
        derivative_cost(w1, w2, b, &dw1, &dw2, &db);
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
