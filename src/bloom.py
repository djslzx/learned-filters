from bitarray import bitarray
from math import log, log2
import mmh3 

class Bloom(object):
    """
    A Bloom filter holding 
      - `n` elements,
      - `e` false positive rate,
      - `m` bit array size, and
      - `k` hash functions
    """

    @classmethod
    def get_m(self, n, e):
        """
        Compute m, the size of the bit array

        m = (n * lg(1/e))/ln(2)
        """
        m = (n * -log2(e))/log(2)
        return int(m)
    
    @classmethod
    def get_k(self, n, e):
        """
        Compute k, the number of hash functions to use

        k = lg(1/e)
        """
        k = -log2(e)
        return int(k)

    def __init__(self, n, e):
        self.n = n
        self.e = e
        self.m = Bloom.get_m(n, e)
        self.k = Bloom.get_k(n, e)
        self.bits = bitarray(self.m)
        self.bits.setall(0)

    def add(self, x):
        """
        Add x to the filter
        """
        for i in range(self.k):
            self.bits[mmh3.hash(x,i) % self.m] = True

    def contains(self, x):
        """
        Check if x is in the filter
        """
        for i in range(self.k):
            if self.bits[mmh3.hash(x,i) % self.m] == False:
                return False
        return True

    def __str__(self):
        return str(self.bits)