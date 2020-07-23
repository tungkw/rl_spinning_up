import torch
import torch.nn as nn
import utils

def func1():
    return 1

def func2():
    return 2

def test(func):
    for i in range(10):
        yield func()

if __name__ == "__main__":
    f = func1
    def func():
        return f()
    k = 0
    for i in test(func):
        print(i)
        k += 1
        if k > 5:
            f = func2
