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
