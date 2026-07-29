import ctypes

# 加载共享库
lib = ctypes.CDLL("./mymain.so")

# 调用 add 函数（返回 int）
result = lib.add(3, 5)
print(f"3 + 5 = {result}")

# 调用 say_hello，需要传递字符串（bytes 类型）
lib.say_hello(b"Python")
