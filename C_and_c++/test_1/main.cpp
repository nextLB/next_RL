// mylib.cpp
#include <iostream>

// 用 extern "C" 告诉 C++ 编译器不要粉碎名字，这样 Python 可以直接找到
extern "C" {
    int add(int a, int b) {
        return a + b;
    }

    void say_hello(const char* name) {
        std::cout << "Hello, " << name << " from C++!" << std::endl;
    }
}
