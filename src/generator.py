from word import Word
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
