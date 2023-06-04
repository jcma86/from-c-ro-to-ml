#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float training_data[][2] = {
    { 0.0, 0.0 },
    { 1.0, 2.0 },
    { 2.0, 4.0 },
    { 3.0, 6.0 },
    { 4.0, 8.0 },
};

int n_training_examples = sizeof(training_data) / sizeof(training_data[0]);

// y = x * w;
// y = x1 * w1 + x2 * w2 + ... + b;
// y = x * 2 + 0;

float randf(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float cost(float w)
{
    float sum = 0.0;
    for (int i = 0; i < n_training_examples; i++) {
        float x = training_data[i][0];
        float y = x * w;
        float d = y - training_data[i][1];
        sum += d * d;
    }
    sum /= n_training_examples;

    return sum;
}

void evaluate_model(float w)
{
    printf("************************\n");
    for (int i = 0; i < n_training_examples; i++) {
        float x = training_data[i][0];
        float y = x * w;
        printf("%f * %f = %f\n", x, w, y);
    }
    printf("************************\n");
}

int main()
{
    srand(25);
    // srand(time(0));
    float eps = 1e-3;
    float learning_rate = 1e-3;

    float w = randf();

    w = randf() * 20.0;

    evaluate_model(w);
    for (int i = 0; i < 1000000; i++) {
        float c = cost(w);
        float dw = (cost(w + eps) - c) / eps;
        w -= learning_rate * dw;
    }
    evaluate_model(w);

    return 0;
}