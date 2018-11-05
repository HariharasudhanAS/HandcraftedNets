import network
import numpy as np

def binary(x):
    get_bin = '{:032b}'.format(x)
    lst = np.asarray([int(i) for i in str(get_bin)])
    lst = np.reshape(lst, (32,1))
    e = np.zeros((2,1))
    e[lst[-1]] = 1.0
    a = (lst,e)
    return a

def binary_1(x):
    get_bin = '{:032b}'.format(x)
    lst = np.asarray([int(i) for i in str(get_bin)])
    a = (lst,lst[-1])
    return a

train_data = [binary(x) for x in range(0,50000)]

test_data = [binary_1(x) for x in range(50000,60000)]

validation_data = [binary_1(x) for x in range(60000,70000)]

net = network.Network([32, 2])
net.SGD(train_data, 100, 10, 10.0, test_data=test_data)
