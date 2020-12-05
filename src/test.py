from bloom import Bloom
from model import Net, train
from random import random
import torch as T

def make_exs(n, m):
    """
    n is dims for x
    m is number of exs
    """
    present = {T.rand(1,n) for _ in range(m//2)}    
    absent = {T.rand(1,n) for _ in range(m//2)}
    exs = ([(x, T.tensor([[1]], dtype=T.bool)) for x in present] + 
           [(x, T.tensor([[0]], dtype=T.bool)) for x in absent])

    # TODO: clean up to avoid iterating repeatedly
    return ([ex[0] for ex in exs], [ex[1] for ex in exs])
    
def model_test(n, m):
    net = Net(n)
    xs, ys = make_exs(n, m)
    train(net, xs, ys, 1)

    # TODO

def bloom_test(n, e, test):

    bloom = Bloom(n, e)

    print("Size of bit array: {}".format(bloom.m))
    print("False positive probability: {}".format(bloom.e))
    print("Number of hash functions: {}".format(bloom.k))
 
    present = {str(random()) for _ in range(n)}
    absent = {str(random()) for _ in range(n)}

    for item in present:
        bloom.add(item)

    # Count false pos/neg
    f_pos = f_neg = t_pos = t_neg = 0
    for x in present:
        if bloom.contains(x):
            t_pos += 1
        else:
            f_neg += 1
    for x in absent:
        if bloom.contains(x):
            f_pos += 1
        else:
            t_neg += 1

    f_pos_rate = f_pos/(t_pos + f_pos)
    f_neg_rate = f_neg/(t_neg + f_neg)

    return f_pos_rate, f_neg_rate

 
if __name__ == '__main__':
    n = 10000 # number of items to add
    e = 0.01 # false positive probability
    # fpr, fnr = bloom_test(n, e, [])
    # print("f_pos rate={}, f_neg rate={}".format(fpr, fnr))
    model_test(n, 5)

