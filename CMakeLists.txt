cmake_minimum_required(VERSION 3.5)
project(FROM_Cro_TO_ML)

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_C_FLAGS "-Wall -Wextra -Werror")
set(CMAKE_BUILD_TYPE "Debug")

find_package(glfw3 REQUIRED)
find_package(raylib REQUIRED)

set(NN_LIB libs/nn.c libs/nn.h)

add_library(nn STATIC ${NN_LIB})

target_link_libraries(nn m glfw raylib)

add_executable(simple_neuron src/simple_neuron.c)
add_executable(gates src/gates.c)
add_executable(gates_2 src/gates_2.c)
add_executable(nn_example src/nn_example.c)
add_executable(mnist src/mnist.c)
add_executable(upscale_img src/upscale_img.c)

target_link_libraries(simple_neuron m)
target_link_libraries(gates m)
target_link_libraries(gates_2 m)
target_link_libraries(nn_example m nn)
target_link_libraries(mnist m nn)
target_link_libraries(upscale_img m nn raylib)