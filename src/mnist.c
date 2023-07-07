#include "../libs/nn.h"
#include <assert.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void)
{
    srand(time(0));
    randv();

    FILE* fexamples = NULL;
    FILE* flabels = NULL;

    fexamples = fopen("../mnist/train-images-idx3-ubyte", "rb");
    assert(fexamples != NULL);
    flabels = fopen("../mnist/train-labels-idx1-ubyte", "rb");
    assert(flabels != NULL);

    int head_examples[4];
    int head_labels[2];
    unsigned char pixel;
    unsigned char label;

    fread(&head_examples, 4, 4, fexamples);
    fread(&head_labels, 4, 2, flabels);

    // size_t n_labels = ntohl(head_labels[1]);
    size_t n_samples = ntohl(head_examples[1]);
    size_t rows = ntohl(head_examples[2]);
    size_t cols = ntohl(head_examples[3]);

    Mat training_data = mat_create(n_samples, rows * cols + 10);
    Mat m_in = {
        .rows = training_data.rows,
        .cols = rows * cols,
        .stride = rows * cols + 10,
        .data = &MAT_AT(training_data, 0, 0),
    };

    Mat m_out = {
        .rows = m_in.rows,
        .cols = training_data.cols - m_in.cols,
        .stride = m_in.stride,
        .data = &MAT_AT(training_data, 0, m_in.cols),
    };

    for (size_t i = 0; i < n_samples; i++) {
        printf("\rReading sample %zu/%zu", i + 1, n_samples);
        fflush(0);
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                fread(&pixel, 1, 1, fexamples);
                MAT_AT(training_data, i, r * cols + c) = (int)pixel / 255.0;
            }
        }
        fread(&label, 1, 1, flabels);
        for (size_t c = 0; c < 10; c++)
            MAT_AT(training_data, i, rows * cols + c) = 0;
        MAT_AT(training_data, i, rows * cols + (int)label) = 1;
    }

    mat_shuffle_rows(training_data);

    fclose(fexamples);
    fclose(flabels);

    // for (size_t i = 0; i < n_samples; i++) {
    //     for (size_t r = 0; r < rows; r++) {
    //         for (size_t c = 0; c < cols; c++) {
    //             printf("%3d", (int)(MAT_AT(training_data, i, r * cols + c) * 255));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    size_t layers[] = { m_in.cols, 7, m_out.cols };
    size_t n_layers = ARRAY_SIZE(layers);
    DATA_TYPE learning_rate = 0.5;

    NN nn = nn_create(layers, n_layers, learning_rate);
    NN dnn = nn_create(layers, n_layers, learning_rate);
    nn_rand(nn, -1.0, 1.0);

    nnui_init(nn, m_in.rows, 0, false);
    bool paused = false;
    size_t batch_size = 1;
    nnui_set_status_message(paused ? "Training paused..." : "Training!");

    size_t sample = 0;
    while (!nnui_should_close()) {
        if (!paused) {
            for (size_t ex = 0; ex < m_in.rows / 6; ex += batch_size) {
                nn_backprop(nn, dnn, m_in, m_out, ex, batch_size);
                nn_update_params(nn, dnn);
            }
        }

        if (sample != nnui_get_current_example()) {
            sample = nnui_get_current_example();

            printf("Number ");
            for (size_t c = 0; c < m_out.cols; c++) {
                if (MAT_AT(m_out, sample, c) == 1) {
                    printf("(%zu):\n", c);
                    break;
                }
            }
            for (size_t r = 0; r < rows; r++) {
                for (size_t c = 0; c < cols; c++) {
                    printf("%3d", (int)(MAT_AT(training_data, sample, r * cols + c) * 255));
                }
                printf("\n");
            }
            printf("\n");
        }

        mat_cpy(NN_INPUT(nn), mat_row(m_in, sample));
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

    nn_destroy(nn);
    nn_destroy(dnn);
    mat_destroy(training_data);

    return 0;
}