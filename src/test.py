from word import Word
from model import WordNet
from bloom import WordBloom, Bloom
from toast import Toast
from sandwich import Sandwich
from util import shuffled, ilen
import torch as T

def make_uniform_exs(num_exs, n, c):
    """
    Generate examples that are uniformly randomly labeled
    """
    xs = [Word(n, c) for _ in range(num_exs)]
    ys = ([0 for _ in range(num_exs/2)] + 
          [1 for _ in range(num_exs/2)])

    return xs, shuffled(ys)
    
def make_sum_exs(num_exs, n, c):
    """
    Generate examples that are linearly separable based on element-wise sum
    """
    xs = [Word(n, c) for _ in range(num_exs)]
    ys = []
    for x in xs:
        tensor = x.data
        label = float(T.sum(tensor)) < (n*c/2)
        # print(tensor, label)
        ys.append(label)

    return xs, ys

def make_parity_exs(num_exs, n, c):
    """
    Generate examples that are checkerboard-like based on parity
    """
    xs = [Word(n, c) for _ in range(num_exs)]
    ys = []
    for x in xs:
        tensor = x.data
        label = int(T.sum(tensor)) % 2 == 0
        # print(tensor, label)
        ys.append(label)

    return xs, ys

def make_circle_exs(num_exs, n, c):
    """
    Generate examples that are linearly separable based on a circle
    """
    xs = [Word(n, c) for _ in range(num_exs)]
    ys = []
    half_len = n*c/2
    center = T.tensor([half_len, half_len])
    for x in xs:
        tensor = x.data
        label = float(T.dist(T.sum(tensor), center)) > half_len/4
        ys.append(label)

    return xs, ys

def bloom_test(xs, ys, num_pos, num_neg, n, c, e):
    bloom = WordBloom(Bloom.init_ne(num_pos, e))

    positives = [x for x,y in zip(xs,ys) if y]
    bloom.add_set(positives)

    false_pos = false_neg = 0
    for x,y in zip(xs,ys):
        filter_contains = bloom.contains(x)
        false_pos += not y and filter_contains
        false_neg += y and not filter_contains

    print(bloom)
    print("fpr: {}, fnr: {}, correct%: {}"
          .format(false_pos/num_neg, 
                  false_neg/num_pos, 
                  1 - (false_pos + false_neg)/(num_pos + num_neg)))
 
def model_test(xs, ys, num_pos, num_neg, n, c, epochs):
    net = WordNet(n, c)
    net.train(xs, ys, epochs)

    false_pos = false_neg = 0
    for x,y in zip(xs,ys):
        model_contains = bool(net(x) > 0.5) #FIXME
        false_pos += not y and model_contains
        false_neg += y and not model_contains

    print("fpr: {}, fnr: {}, correct%: {}"
          .format(false_pos/num_neg, 
                  false_neg/num_pos, 
                  1 - (false_pos + false_neg)/(num_pos + num_neg)))

def toast_test(xs, ys, num_pos, num_neg, n, c, err, epochs):
    toast = Toast(n, c, err)
    print(toast)
    
    print("Training toast...")
    toast.train(xs, ys, epochs)

    print("Testing toast...")
    false_pos = false_neg = 0
    for x,y in zip(xs,ys):
        filter_contains = bool(toast.contains(x))
        false_pos += not y and filter_contains
        false_neg += y and not filter_contains

    print("fpr: {}, fnr: {}, correct%: {}"
          .format(false_pos/num_neg, 
                  false_neg/num_pos, 
                  1 - (false_pos + false_neg)/(num_pos + num_neg)))

def sandwich_test(xs, ys, num_pos, num_neg, n, c, err, b_1, epochs):
    sandwich = Sandwich(n, c, err, b_1)
    print(sandwich)

    print("Training sandwich...")
    sandwich.train(xs, ys, epochs)

    print("Testing sandwich...")
    false_pos = false_neg = 0
    for x,y in zip(xs,ys):
        filter_contains = bool(sandwich.contains(x))
        false_pos += not y and filter_contains
        false_neg += y and not filter_contains

    print("fpr: {}, fnr: {}, correct%: {}"
          .format(false_pos/num_neg, 
                  false_neg/num_pos, 
                  1 - (false_pos + false_neg)/(num_pos + num_neg)))


if __name__ == '__main__':
    num_exs = 1000
    epochs = 50
    n=5
    c=10

    # xs, ys = make_uniform_exs(num_exs, n, c)
    # xs, ys = make_sum_exs(num_exs, n, c)
    # xs, ys = make_parity_exs(num_exs, n, c)
    xs, ys = make_circle_exs(num_exs, n, c)
    num_pos = ilen(x for x,y in zip(xs,ys) if y)
    num_neg = ilen(x for x,y in zip(xs,ys) if not y)

    print("pos: {}, neg: {}".format(num_pos, num_neg))

    print("Running Bloom test...")
    bloom_test(xs, ys, num_pos, num_neg, n=n, c=c, e=0.01)
    print("Done.")
    # print("Running model test...")
    # model_test(num_pos, num_neg, n=5, c=10, epochs=1000)
    # print("Done.")
    # print("Running toast test...")
    # toast_test(xs, ys, num_pos, num_neg, n=n, c=c, err=0.01, epochs=epochs)
    # print("Done")
    print("Running sandwich test...")
    sandwich_test(xs, ys, num_pos, num_neg, n=n, c=c, err=0.01, b_1=4, epochs=epochs)
    print("Done")
