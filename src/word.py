"""
Define the input type that will be used for both filters and ML models
"""

import torch as T

class Word(object):
    """
    A representation for a word of n letters
    where letters are from an alphabet of size c
    """

    def __init__(self, n, c):
        self.n = n
        self.c = c
        self.data = T.randint(c, (n,))
    
    def _val_to_tensor(self, x):
        """
        Maps the value x to a 1xc tensor where
        all values are 0 except for the x-th value,
        which is 1
        """
        t = T.zeros(self.c)
        t[x] = 1
        return t

    @property
    def model_type(self):
        """
        Returns the word as a 1x(nc) tensor
        """
        return T.cat(tuple(self._val_to_tensor(x) 
                           for x in self.data))

    @property
    def filter_type(self):
        """
        Returns the word as a string of n letters
        """
        return ''.join(str(x.item()) for x in self.data)

    def __str__(self):
        return "data: {}, n={}, c={}".format(self.data, 
                                             self.n, 
                                             self.c)

if __name__ == '__main__':
    w = Word(3, 2)
    print(w)
    print(w.model_type)
    print(w.filter_type)
