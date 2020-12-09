from word import Word
from util import shuffled
import torch as T

def make_uniform_exs(num_exs, n, c):
    """
    Generate examples that are uniformly randomly labeled
    """
    xs = [Word(n, c) for _ in range(num_exs)]
    ys = ([0 for _ in range(num_exs//2)] + 
          [1 for _ in range(num_exs//2)])

    return xs, shuffled(ys)
    
def make_line_exs(num_exs, n, c):
    """
    Generate examples that are linearly separable based on element-wise sum
    """
    words = [Word(n, c) for _ in range(num_exs)]
    labels = []
    for w in words:
        label = float(T.sum(w.data)) < (n*c/2)
        # print(tensor, label)
        labels.append(label)
    return words, labels

def make_polynomial_exs(num_exs, n, c):
    """
    Generate examples that are separated by a randomly generated polynomial
    """
    # Generate coefficients in [0,c)
    d = 3
    coeffs = T.rand(size=(n,), dtype=T.float) - 0.5

    def g(tensor):
        """Discriminating polynomial"""
        powers = T.pow(tensor,
                       T.arange(start=0, end=n, dtype=T.float))
        out = float(T.dot(powers, coeffs))
        return out > 0
    
    words = [Word(n, c) for _ in range(num_exs)]
    labels = []
    for w in words:
        label = g(w.data)
        labels.append(label)
    return words, labels

def make_parity_exs(num_exs, n, c):
    """
    Generate examples that are checkerboard-like based on parity
    """
    words = [Word(n, c) for _ in range(num_exs)]
    labels = []
    for w in words:
        tensor = w.data
        label = int(T.sum(tensor)) % 2 == 0
        labels.append(label)
    return words, labels

def make_circle_exs(num_exs, n, c):
    """
    Generate examples that are linearly separable based on a circle
    """
    words = [Word(n, c) for _ in range(num_exs)]
    labels = []
    half_len = n*c/2
    center = T.tensor([half_len, half_len])
    for w in words:
        tensor = w.data
        label = float(T.dist(T.sum(tensor), center)) > half_len/4
        labels.append(label)
    return words, labels