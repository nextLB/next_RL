(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2$ ls
CMakeLists.txt  example.cpp  pybind11  test.py
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2$ cat ./CMakeLists.txt 
cmake_minimum_required(VERSION 3.10)
project(Example)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)

pybind11_add_module(example example.cpp NO_EXTRAS)
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2$ cat ./example.cpp 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

int add(int i, int j) {
    return i + j;
}

double sum_vector(const std::vector<double> &v) {
    double sum = 0.0;
    for (double val : v) sum += val;
    return sum;
}

class Pet {
public:
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }
private:
    std::string name;
};

// --- pybind11 模块定义 ---
PYBIND11_MODULE(example, m) {
    // 关键：定义 py 为 pybind11 命名空间的别名
    namespace py = pybind11;

    m.doc() = "pybind11 example plugin";

    m.def("add", &add, "A function which adds two numbers");
    m.def("sum_vector", &sum_vector, "Sum all elements in a list of numbers");

    py::class_<Pet>(m, "Pet")
    .def(py::init<const std::string &>())
    .def("setName", &Pet::setName)
    .def("getName", &Pet::getName);
}
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2$ mkdir ./build
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2$ cd ./build/
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2/build$ cmake ..
-- The C compiler identification is GNU 14.2.0
-- The CXX compiler identification is GNU 14.2.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- pybind11 v3.1.0 
-- Found Python: /home/next/anaconda3/envs/WYT_S_S/bin/python3 (found suitable version "3.11.15", minimum required is "3.8") found components: Interpreter Development.Module Development.Embed
Using compatibility mode for Python, set PYBIND11_FINDPYTHON to NEW/OLD to silence this message
-- Performing Test HAS_FLTO_AUTO
-- Performing Test HAS_FLTO_AUTO - Success
-- Configuring done (1.3s)
-- Generating done (0.0s)
-- Build files have been written to: /home/next/桌面/NEXT/C_and_c++/test_2/build
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2/build$ make
[ 50%] Building CXX object CMakeFiles/example.dir/example.cpp.o
[100%] Linking CXX shared module example.cpython-311-x86_64-linux-gnu.so
[100%] Built target example
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2/build$ ls
CMakeCache.txt  CMakeFiles  cmake_install.cmake  example.cpython-311-x86_64-linux-gnu.so  Makefile  pybind11
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2/build$ cp ../test.py ./
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2/build$ ls
CMakeCache.txt  CMakeFiles  cmake_install.cmake  example.cpython-311-x86_64-linux-gnu.so  Makefile  pybind11  test.py
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2/build$ cat ./test.py 
import example

print(example.add(10, 20))
print(example.sum_vector([1.0, 2.5, 3.5]))

p = example.Pet("Mochi")
print(p.getName())
p.setName("Kitty")
print(p.getName())
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2/build$ python ./test.py 
30
7.0
Mochi
Kitty
(WYT_S_S) next@debian:~/桌面/NEXT/C_and_c++/test_2/build$ 
