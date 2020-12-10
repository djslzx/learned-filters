import torch as T
from model import WordNet
from bloom import WordBloom, Bloom
from toast import Toast
from sandwich import Sandwich
from util import ilen
import argparse
import generator as gen 

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

    print(net)
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

    print(toast)
    print("fpr: {}, fnr: {}, correct%: {}"
          .format(false_pos/num_neg, 
                  false_neg/num_pos, 
                  1 - (false_pos + false_neg)/(num_pos + num_neg)))

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

    print(sandwich)
    print("fpr: {}, fnr: {}, correct%: {}"
          .format(false_pos/num_neg, 
                  false_neg/num_pos, 
                  1 - (false_pos + false_neg)/(num_pos + num_neg)))

if __name__ == '__main__':

    p = argparse.ArgumentParser(description='Construct ')
    p.add_argument('-f', '--filter', type=str, default="bloom", 
                   help='filter type (bloom, toast, sandwich)')
    p.add_argument('-l', '--generator', type=str, default="circle", 
                   help='example generator type (uniform, line, parity, circle, polynomial)')
    p.add_argument('-s', '--size', type=int, default=1000,
                   help='number of examples')
    p.add_argument('-n', '--length',type=int, default = 5,
                   help='length of example strings')
    p.add_argument('-c', '--alphabet', type=int, default = 10,
                   help='size of example alphabet')
    p.add_argument('-e', '--error', type=int, default = 0.01,
                   help='error rate')
    p.add_argument('-ep', '--epochs', type=int, default = 100,
                   help='number of epochs')
    a = p.parse_args()
    
    num_exs = a.size
    n = a.length
    c = a.alphabet
    u_size = c ** n
    err = a.error
    epochs = a.epochs

    xs = gen.make_uniform_words(num_exs, n, c)
    ys = {
        "uniform": lambda xs: gen.label_uniform(xs),
        "line": lambda xs: gen.label_line(xs, n, c),
        "polynomial": lambda xs: gen.label_polynomial(xs, n),
        "parity": lambda xs: gen.label_parity(xs),
        "circle": lambda xs: gen.label_circle(xs, n, c),
    }[a.generator](xs)

    num_pos = ilen(x for x,y in zip(xs,ys) if y)
    num_neg = ilen(x for x,y in zip(xs,ys) if not y)

    print("n={}, c={}, |U|={}, |K|={}, |T|={}"
          .format(n, c, u_size, num_pos, num_neg))

    if a.filter == "bloom":
        print("Running Bloom test...")
        bloom_test(xs, ys, num_pos, num_neg, n=n, c=c, e=err)
    elif a.filter == "toast":
        print("Running toast test...")
        toast_test(xs, ys, num_pos, num_neg, n=n, c=c, err=err, epochs=epochs)
    elif a.filter == "sandwich":
        print("Running sandwich test...")
        sandwich_test(xs, ys, num_pos, num_neg, n=n, c=c, err=err, err1k=5, epochs=epochs)
    else:
        print(a.filter)
        raise Exception("Not a valid filter")
    print("Done")

