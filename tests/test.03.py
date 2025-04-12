import sys
import numpy as np
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import Variable, Add, add, square, exp

# xs = [Variable(np.array(2)), Variable(np.array(3))]
# f = Add()
# ys = f(xs)
# y = ys[0]
# print(ys.data)

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)

x3 = Variable(np.array(4))
y = square(x3)
print(y.data)

x4 = Variable(np.array(5))
y = exp(x4)
print(y.data)