project(FROM_Cro_TO_ML)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_C_FLAGS "-Wall -Wextra -Werror")

add_executable(simple_neuron src/simple_neuron.c)
add_executable(gates src/gates.c)

target_link_libraries(simple_neuron m)
target_link_libraries(gates m)