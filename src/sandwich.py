import torch as T
from bloom import WordBloom
from model import WordNet

class Sandwich:

    def __init__(self, n, c, err, tau=0.5):
        """
        n: number of letters in string
        c: size of alphabet
        """
        self.model = WordNet(n, c)
        self.tau = tau # TODO: tune

        # AMQs can only be set up after training model
        self.amq1 = None
        self.amq2 = None
        self.err = err
        
    def train(self, xs, ys, epochs):
        pass

    def contains(self, x):
        pass

    def __str__(self):
        pass
