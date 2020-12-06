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

from bloom import WordBloom
from model import WordNet
import torch as T

class Toast(object):

    def __init__(self, n, c, err, tau=0.5):
        """
        n: number of letters in string
        c: size of alphabet
        """
        self.model = WordNet(n, c)
        self.tau = tau # TODO: tune

        # AMQ can only be set up after training model
        self.amq = None
        self.err = err

    def train(self, xs, ys, epochs):
        # Train neural net
        # Note: torch dataloader takes care of shuffling
        self.model.train(xs, ys, epochs)

        # Get false negatives
        positives = [x for x,y in zip(xs,ys) if y]
        false_negs = [x for x in positives
                      if not self.model(x) > self.tau]
        
        # Build filter for negatives
        if len(false_negs) > 0:
            self.amq = WordBloom(len(false_negs), self.err)
            self.amq.add_set(false_negs)

    def contains(self, x):
        # Check amq only if model reports negative and model has false negatives
        model_ans = self.model(x) > self.tau
        if model_ans or self.amq is None:
            return model_ans
        else:
            return self.amq.contains(x)

    def __str__(self):
        return "amq: {}, e={}".format(self.amq, self.err)
