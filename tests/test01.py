import sys
import numpy as np
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import Variable, Square, Exp, numerical_diff


def test01():
    A = Square()
    B = Exp()
    C = Square()
    
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    
    print(type(y))
    print(y.data)
    
def test02():
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)
    
def test03():
    A = Square()
    B = Exp()
    C = Square()
    
    x = Variable(np.array(0.5))

    def f(x):
        return C(B(A(x)))

    dy = numerical_diff(f, x)
    print(dy)

def test04():
    A = Square()
    B = Exp()
    C = Square()
    
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
    
if __name__ == "__main__":
    # test03()
    test04()
    