"""
-- > model -yes-> user
      |
      no
      |
      v
     filter -yes-> user
      |
      no-> user
"""

from bloom import Bloom
from model import Net
import torch as T

class Toast(object):

    def __init__(self, n, c, err, tau=0.5):
        """
        n: number of letters in string
        c: size of alphabet
        """
        self.model = Net(n*c)
        self.tau = tau # TODO: tune

        # AMQ can only be set up after training model
        self.amq = None
        self.err = err

    def train(self, xs, ys, epochs):
        # Train neural net
        # Note: torch dataloader takes care of shuffling
        # print("pos:", positives)
        # print("neg:", negatives)

        # xs = T.cat((positives, negatives))
        # ys = T.cat((T.ones(positives.size()),
        #             T.zeros(negatives.size())))

        print("xs:", xs)
        print("ys:", ys)
        self.model.train(xs, ys, epochs)

        # Get false negatives
        positives = [x for x,y in zip(xs,ys) if y]
        false_negs = [x for x in positives
                      if not self.model(x) > self.tau]
        
        # Build filter for negatives
        self.amq = Bloom(len(false_negs), self.err)
        self.amq.add_set(false_negs)

    def contains(self, x):
        # Uses logical short-circuiting to check
        # amq only if model reports negative
        return (self.model(x) > self.tau or
                self.amq.contains(x))

    def __str__(self):
        pass
