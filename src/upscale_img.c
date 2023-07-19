#include "../libs/nn.h"
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

Image img_in;
Image img_out;
Texture2D texture_in;
Texture2D texture_out;
float scale_in = 1.0;
bool render_test = true;

void img_render(NN* nn, void* param)
{
    if (!render_test)
        return;
    render_test = false;

    (void)param;

    DrawRectangle(0, 0, NNUI_WIDTH, NNUI_HEIGHT, LIGHTGRAY);

    img_out = GenImageColor(img_in.width * scale_in, img_in.height * scale_in, RED);
    for (size_t r = 0; r < (size_t)img_out.height; r++) {
        for (size_t c = 0; c < (size_t)img_out.width; c++) {
            MAT_AT(NN_INPUT(*nn), 0, 0) = (DATA_TYPE)r / (img_out.height - 1);
            MAT_AT(NN_INPUT(*nn), 0, 1) = (DATA_TYPE)c / (img_out.width - 1);
            nn_forward(*nn);

            Color color = {
                .r = (unsigned char)(MAT_AT(NN_OUTPUT(*nn), 0, 0) * 255.0),
                .g = (unsigned char)(MAT_AT(NN_OUTPUT(*nn), 0, 1) * 255.0),
                .b = (unsigned char)(MAT_AT(NN_OUTPUT(*nn), 0, 2) * 255.0),
                .a = 255,
            };

            ImageDrawPixel(&img_out, c, r, color);
        }
    }

    DrawTextureEx(
        texture_in,
        (Vector2) { .x = 10, .y = 10 },
        0.0,
        1.0,
        WHITE);

    texture_out = LoadTextureFromImage(img_out);
    DrawTextureEx(
        texture_out,
        (Vector2) { .x = 10, .y = 200 },
        0.0,
        1.0,
        WHITE);
}

int main(int argc, char* argv[])
{
    srand(time(0));
    randv();
    img_in = LoadImage("../assets/landscape2.png");

    // Mat will hold row, col, r, g, b  values; row and col are inputs
    Mat m_img = mat_create(img_in.width * img_in.height, 5);
    Mat m_in = {
        .rows = m_img.rows,
        .cols = 2,
        .stride = 5,
        .data = &MAT_AT(m_img, 0, 0),
    };

    Mat m_out = {
        .rows = m_in.rows,
        .cols = m_img.cols - m_in.cols,
        .stride = m_in.stride,
        .data = &MAT_AT(m_img, 0, m_in.cols),
    };

    DATA_TYPE learning_rate = 0.3;

    NN nn;
    if (argc > 1)
        nn = nn_load_from_file(argv[1]);
    else {
        size_t layers[] = { m_in.cols, 70, 100, 100, 20, 10, m_out.cols };
        size_t n_layers = ARRAY_SIZE(layers);
        nn = nn_create(layers, n_layers, learning_rate);
        nn_rand(nn, -1.0, 1.0);
    }

    NN dnn = nn_create(nn.arch, nn.n_layers, *nn.lr);
    nnui_init(nn, m_in.rows, 0, true);

    // nn_set_act_fx(nn, 2, ACTFX_RELU);
    nn_set_act_fx(nn, 3, ACTFX_RELU);
    // nn_set_act_fx(nn, 4, ACTFX_RELU);

    Color* img_in_data = LoadImageColors(img_in);
    size_t i = 0;
    for (size_t r = 0; r < (size_t)img_in.height; r++) {
        for (size_t c = 0; c < (size_t)img_in.width; c++) {
            int ind = (r * img_in.width) + c;
            MAT_AT(m_img, i, 0) = (DATA_TYPE)r / (img_in.height - 1);
            MAT_AT(m_img, i, 1) = (DATA_TYPE)c / (img_in.width - 1);
            DATA_TYPE cr = (DATA_TYPE)img_in_data[ind].r / 255.0;
            DATA_TYPE cg = (DATA_TYPE)img_in_data[ind].g / 255.0;
            DATA_TYPE cb = (DATA_TYPE)img_in_data[ind].b / 255.0;

            // DATA_TYPE gray = (0.3 * cr) + (0.59 * cg) + (0.11 * cb);
            // MAT_AT(m_img, i, 2) = gray;
            // MAT_AT(m_img, i, 3) = gray;
            // MAT_AT(m_img, i, 4) = gray;

            MAT_AT(m_img, i, 2) = cr;
            MAT_AT(m_img, i, 3) = cg;
            MAT_AT(m_img, i, 4) = cb;
            i++;
        }
    }

    // NOTE: Textures MUST be loaded after Window initialization (OpenGL context is required)
    texture_in = LoadTextureFromImage(img_in);

    bool paused = false;
    size_t batch_size = 1; // m_in.rows / 8;

    SetTraceLogLevel(LOG_NONE);
    nnui_set_test_area_render(img_render);
    char buffer[50];

    while (!nnui_should_close()) {
        if (!paused) {
            for (size_t ex = 0; ex < m_in.rows; ex += batch_size) {
                nn_backprop(nn, dnn, m_in, m_out, ex, batch_size);
                nn_update_params(nn, dnn);
            }
        }

        nn_cost(nn, m_in, m_out, 0, m_in.rows);
        nnui_render();
        if (nnui_was_key_pressed(KEY_R))
            nnui_reset_cam();
        if (IsKeyPressed(KEY_T))
            render_test = !render_test;
        if (IsKeyPressed(KEY_P))
            paused = !paused;
        if (IsKeyPressed(KEY_S)) {
            snprintf(buffer, sizeof(buffer), "nn_%.5f.nn", *nn.cost);
            nn_save_to_file(nn, buffer);
        }
        if (IsKeyPressed(KEY_KP_ADD))
            scale_in += 0.5;
        if (IsKeyPressed(KEY_KP_SUBTRACT))
            scale_in = max(1.0, scale_in - 0.5);

        snprintf(buffer, sizeof(buffer), "Training: %s Preview (%.1fx): %s", paused ? "off" : "on", scale_in, render_test ? "on" : "off");
        nnui_set_status_message(buffer);
        if (!paused)
            nnui_add_point_to_chart(*nn.cost);
    }

    nn_destroy(nn);
    nn_destroy(dnn);
    mat_destroy(m_img);

    UnloadImage(img_in);
    UnloadImage(img_out);
    UnloadTexture(texture_in);
    UnloadTexture(texture_out);

    CloseWindow();

    return 0;
}