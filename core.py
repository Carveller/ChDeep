import numpy as np
import weakref

from utils import as_array

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        
        self.data = data
        self.grad = None
        self.creator = None
        # 增加辈分变量，实现复杂计算图
        self.generation = 0
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        
    # def backward(self):
    #     f = self.creator    # 获取函数
    #     if f is not None:
    #         x = f.input     # 获取函数的输入
    #         x.grad = f.backward(self.grad)      # 调用函数的backward方法
    #         x.backward()    # 调用自己前面那个变量的backward方法（递归）
    
    def backward(self):
        # 最后的输出是没有梯度的，需要指定梯度
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # 保存函数，后面需要调用函数的backward方法
        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)
        
        while funcs:
            # 获得后面的func
            f = funcs.pop()
            # 获取输出
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            # 调用函数的backward，获得输入的梯度，并作为下一个函数backward的输入
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            # 赋值输入的梯度
            for x, gx in zip(f.inputs, gxs):
                # 如果x.grad是None，说明是第一次遇到这个变量
                if x.grad is None:
                    x.grad = gx
                else:   # 否则说明这个变量被重复使用，需要在已经有的梯度上相加
                    '''
                    注意如果add.backward()中直接将gy返回, 那么这里不能是x.grad += gx, 
                    而应该是x.grad = x.grad + gx
                    这是因为在add的backward中直接将gy的引用传播下来了, 
                    因此x.grad其实和y.grad是一个数组, 如果使用 +=, 是就地操作会影响y.grad
                    但其实可以修改 add.backward(), 使其不要传播gy的引用
                    '''
                    x.grad += gx
                # 保存每个输入的创造者
                if x.creator is not None:
                    add_func(x.creator)
    
    def cleargrad(self):
        self.grad = None

        
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)      # 使用星号进行解包
        if not isinstance(ys, tuple):       # 对非元组情况的额外处理
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        self.generation = max([x.generation for x in inputs])
        
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        
        # 如果列表只有一个元素，则返回第一个元素
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        # x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        # 这里可以直接return gy
        # 这样的话返回的就是输入的引用
        # 在Variable的backword中就需要更改，详细见注释
        gx0 = 1 * gy
        gx1 = 1 * gy
        return gx0, gx1

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)



    
    