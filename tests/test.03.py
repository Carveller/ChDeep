import sys
import numpy as np
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import Variable, Add

xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)