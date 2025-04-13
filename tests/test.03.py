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

# x0 = Variable(np.array(2))
# x1 = Variable(np.array(3))
# y = add(x0, x1)
# print(y.data)

# x3 = Variable(np.array(4))
# y = square(x3)
# print(y.data)

# x4 = Variable(np.array(5))
# y = exp(x4)
# print(y.data)

# x = Variable(np.array(2.0))
# y = Variable(np.array(3.0))

# z = add(square(x), square(y))
# z.backward()
# print(z.data)
# print(x.grad)
# print(y.grad)
# x = Variable(np.array(3.0))
# y = add(x, x)
# y.backward()
# print(x.grad)

# # x = Variable(np.array(3))
# x.cleargrad()
# y = add(add(x, x), x)
# y.backward()

# print(y.grad, id(y.grad))
# print(x.grad, id(x.grad))

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)