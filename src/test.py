from model import Net
from bloom import Bloom
from toast import Toast
from random import random
import torch as T

def make_exs(m, n):
    """
    n is dims for x
    m is number of exs
    """
    return (T.randint(10, (m,n), dtype=T.float)/10,
            T.randint(2,(m,1), dtype=T.float))
    
def toast_test(m, n, c, err, epochs):
    toast = Toast(n, c, err)
    xs, ys = make_exs(m, n*c)
    # pos_indices = T.nonzero(T.reshape(ys, (-1,))).squeeze()
    # neg_indices = T.nonzero(T.reshape(~ys.bool(), (-1, ))).squeeze()
    # positives = xs[pos_indices]
    # negatives = xs[neg_indices]

    # toast.train(positives, negatives, epochs)
    toast.train(xs, ys, epochs)

    correct = 0
    for x,y in zip(xs,ys):
        correct += int(toast.contains(x) == bool(y))
    print("correct: {}, incorrect: {}, correct%: {}"
          .format(correct, m-correct, correct/m))

def model_test(m, n, epochs):
    net = Net(n)
    xs, ys = make_exs(m, n)
    net.train(xs, ys, epochs)

    correct = 0
    for x,y in zip(xs,ys):
        print(x, net(x) > 0.5, y)
        correct += int((net(x) > 0.5) == y)
    print("correct: {}, incorrect: {}, correct%: {}"
          .format(correct, m-correct, correct/m))

def bloom_test(m, n, e):
    xs, ys = make_exs(m, n)
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
    # n = 10000 # number of items to add
    # e = 0.01 # false positive probability
    # fpr, fnr = bloom_test(n, e, [])
    # print("f_pos rate={}, f_neg rate={}".format(fpr, fnr))
    # model_test(n, m=10, epochs=1)
    toast_test(m=10, n=5, c=10, err=0.01, epochs=5)


