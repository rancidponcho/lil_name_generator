cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(lil_name_generator)

set(CMAKE_PREFIX_PATH ~/dev/tools/libtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(lil_name_generator main.cpp lil_name_gen.cpp)
target_link_libraries(lil_name_generator "${TORCH_LIBRARIES}")
set_property(TARGET lil_name_generator PROPERTY CXX_STANDARD 17)