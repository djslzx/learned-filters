from word import Word
from model import WordNet
from bloom import WordBloom
from toast import Toast
from util import shuffled
import torch as T

def make_exs(num_pos, num_neg, n, c):
    """
    m is number of exs
    """
    xs = [Word(n, c) for _ in range(num_pos + num_neg)]
    ys = ([0 for _ in range(num_pos)] + 
          [1 for _ in range(num_neg)])

    return xs, shuffled(ys)
    
def toast_test(num_pos, num_neg, n, c, err, epochs):
    toast = Toast(n, c, err)
    xs, ys = make_exs(num_pos, num_neg, n, c)
    toast.train(xs, ys, epochs)

    false_pos = false_neg = 0
    for x,y in zip(xs,ys):
        filter_contains = bool(toast.contains(x))
        false_pos += not y and filter_contains
        false_neg += y and not filter_contains

    print(toast)
    print("fpr: {}, fnr: {}, correct%: {}"
          .format(false_pos/num_neg, 
                  false_neg/num_pos, 
                  1 - (false_pos + false_neg)/(num_pos + num_neg)))

def model_test(num_pos, num_neg, n, c, epochs):
    net = WordNet(n, c)
    xs, ys = make_exs(num_pos, num_neg, n, c)
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

def bloom_test(num_pos, num_neg, n, c, e):
    bloom = WordBloom(num_pos, e) 
    xs, ys = make_exs(num_pos, num_neg, n, c)

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
 
if __name__ == '__main__':
    num_pos = 100
    num_neg = 100

    print("Running Bloom test...")
    bloom_test(num_pos, num_neg, n=5, c=10, e=0.01)
    print("Done.")
    # print("Running model test...")
    # model_test(num_pos, num_neg, n=5, c=10, epochs=1000)
    # print("Done.")
    print("Running toast test...")
    toast_test(num_pos, num_neg, n=5, c=10, err=0.01, epochs=500)
    print("Done")
