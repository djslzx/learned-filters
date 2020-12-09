from word import Word
from model import WordNet
from bloom import WordBloom, Bloom
from toast import Toast
from sandwich import Sandwich
from util import shuffled, ilen
import torch as T

def make_uniform_words(num_exs, n, c):
    """
    Generate words uniformly randomly
    """
    return [Word(n, c) for _ in range(num_exs)]
    
def label_uniform(words):
    m = len(words)
    return ([0 for _ in range(m//2)] + 
            [1 for _ in range(m//2)])

def label_line(words, n, c):
    """
    Label examples by separating linearly based on word magnitude
    """
    labels = []
    for w in words:
        tensor = w.data
        label = float(T.sum(tensor)) < (n*c/2)
        labels.append(label)
    return labels

def label_polynomial(words, n):
    """
    Label examples by separating with a randomly generated polynomial
    Note: Doesn't work too well; examples are split lopsidedly,
    sometimes resulting in empty pos or neg sets
    """
    coeffs = T.rand(size=(n,), dtype=T.float) - 0.5

    def g(tensor):
        """
        Compute g(x) for a point x and polynomial discriminator
        defined by coeffs, and check if g(x) > 0
        """
        powers = T.pow(tensor, T.arange(start=0, end=n, dtype=T.float))
        out = float(T.dot(powers, coeffs))
        return out > 0
    
    labels = []
    for w in words:
        tensor = w.data
        label = g(tensor)
        labels.append(label)
    return labels

def label_parity(words):
    """
    Generate examples that are checkerboard-like based on parity
    """
    labels = []
    for w in words:
        tensor = w.data
        label = int(T.sum(tensor)) % 2 == 0
        labels.append(label)
    return labels

def label_circle(words, n, c):
    """
    Generate examples that are linearly separable based on a circle
    """
    labels = []
    z = n*c/2
    center = T.tensor([z, z])
    for w in words:
        tensor = w.data
        label = float(T.dist(T.sum(tensor), center)) > z/4
        labels.append(label)
    return labels

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


if __name__ == '__main__':
    num_exs = 1000
    epochs = 100
    n=5                         # 5-letter words
    c=10                        # c-letter alphabet

    xs = make_uniform_words(num_exs, n, c)
    # ys = label_uniform(xs)
    # ys = label_line(xs, n, c)
    # ys = label_polynomial(xs, n)
    # ys = label_parity(xs)
    ys = label_circle(xs, n, c)

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
