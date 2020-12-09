from word import Word
from model import WordNet
from bloom import WordBloom, Bloom
from toast import Toast
from sandwich import Sandwich
from util import ilen
import torch as T
import argparse
from generator import make_uniform_exs, make_polynomial_exs, make_circle_exs, make_line_exs, make_parity_exs

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
    print(toast)

def sandwich_test(xs, ys, num_pos, num_neg, n, c, err, err1k, epochs):
    sandwich = Sandwich(n, c, err, num_pos, err1k)
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
    print(sandwich)

def quick_test():
    num_exs = 1000
    epochs = 100
    n=5
    c=10

    # xs, ys = make_uniform_exs(num_exs, n, c)
    # xs, ys = make_line_exs(num_exs, n, c)
    # xs, ys = make_parity_exs(num_exs, n, c)
    xs, ys = make_circle_exs(num_exs, n, c)
    # xs, ys = make_polynomial_exs(num_exs, n, c)
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
    sandwich_test(xs, ys, num_pos, num_neg, n=n, c=c, err=0.01, err1k=5, epochs=epochs)
    print("Done")

def command_line():
    p = argparse.ArgumentParser(description='Construct ')
    p.add_argument('-f', '--filter', type=str, default="bloom", 
                   help='filter type (bloom, toast, sandwich')
    p.add_argument('-g', '--generator', type=str, default="uniform", 
                   help='example generator type (uniform, line, parity, circle, polynomial')
    p.add_argument('-s', '--size', type=int, default=1000,
                   help='number of examples')
    p.add_argument('-n', '--length',type=int, default = 5,
                   help='length of example strings')
    p.add_argument('-c', '--alphabet', type=int, default = 10,
                   help='size of example alphabet')
    p.add_argument('-e', '--error', type=int, default = 10,
                   help='error rate')
    p.add_argument('-ep', '--epochs', type=int, default = 100,
                   help='number of epochs')
    a = p.parse_args()
    
    num_exs = a.size
    epochs = a.epoch
    n = a.length
    c = a.alphabet
    err = a.error
    xs, ys = [],[]

    if a.generator is "uniform":
        xs, ys = make_uniform_exs(num_exs, n, c)
    elif a.generator is "line":
        xs, ys = make_line_exs(num_exs, n, c)
    elif a.generator is "parity":
        xs, ys = make_parity_exs(num_exs, n, c)
    elif a.generator is "circle":
        xs, ys = make_circle_exs(num_exs, n, c)
    elif a.generator is "polynomial":
        xs, ys = make_polynomial_exs(num_exs, n, c)
    else:
        raise Exception("Not a valid example generator")

    num_pos = ilen(x for x,y in zip(xs,ys) if y)
    num_neg = ilen(x for x,y in zip(xs,ys) if not y)

    print("pos: {}, neg: {}".format(num_pos, num_neg))
    
    if a.filter is "bloom":
        print("Running Bloom test...")
        bloom_test(xs, ys, num_pos, num_neg, n=n, c=c, e=err)
        print("Running toast test...")
    elif a.filter is "toast":
        toast_test(xs, ys, num_pos, num_neg, n=n, c=c, err=err, epochs=epochs)
    elif a.filter is "sandwich":
        print("Running sandwich test...")
        sandwich_test(xs, ys, num_pos, num_neg, n=n, c=c, err=err, err1k=5, epochs=epochs)
    else:
        raise Exception("Not a valid filter")
    print("Done")

if __name__ == '__main__':
    quick_test()
    #command_line()
