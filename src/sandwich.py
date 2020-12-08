import torch as T
from bloom import WordBloom
from model import WordNet
from math import log

class Sandwich:

    def __init__(self, n, c, b, err):
        """
        n: number of letters in string
        c: size of alphabet
        b: use b*n bits for amqs
        err: total error rate of sandwich
        """
        self.model = WordNet(n, c)
        self.tau = 0.5 # default value, adjust by tuning later
        self.alpha = 2 ** -log(2)

        # AMQs can only be set up after training model
        self.amq1 = None
        self.amq2 = None
        self.err = err

    def train(self, xs, ys, epochs):
        """
        Train the model and setup the two amqs.
        """
        

    def contains(self, x):
        pass

    def __str__(self):
        pass
