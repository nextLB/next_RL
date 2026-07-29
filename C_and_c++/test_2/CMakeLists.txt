cmake_minimum_required(VERSION 3.10)
project(Example)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)

pybind11_add_module(example example.cpp)   # 去掉 NO_EXTRAS（除非你需要它控制编译选项）

# 强制输出名为 example.so
set_target_properties(example PROPERTIES
    PREFIX ""          # 去掉可能自动添加的 "lib" 前缀
    SUFFIX ".so"       # 强制后缀为 .so
# )
