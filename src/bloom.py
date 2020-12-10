from bitarray import bitarray
from math import log, log2, exp
import mmh3 

class Bloom:
    """
    A Bloom filter holding 
      - `n` elements,
      - `e` false positive rate,
      - `m` bit array size, and
      - `k` hash functions
    """

    # TODO: change to take in given size,
    # figure out other params based on that

    def __init__(self, n):
        self.n = n

    @classmethod
    def init_ne(cls, n, err):
        """
        Initialize using number of elements and error 
        """
        b = Bloom(n)
        b.err = err
        b.m = Bloom.get_m(n,err)
        b.k = Bloom.get_k(n, b.m)
        b.bits = bitarray(b.m)
        b.bits.setall(0)
        return b

    @classmethod
    def init_nm(cls, n, m): 
        """
        Initialize using number of elemnts and size of bitarray 
        """
        b = Bloom(n)
        b.m = m
        b.k = Bloom.get_k(n, m)
        b.err = (1 - exp(-b.k * b.n/b.m)) ** b.k
        b.bits = bitarray(b.m)
        b.bits.setall(0)
        return b

    def add(self, x):
        """
        Add x to the filter
        """
        for i in range(self.k):
            self.bits[mmh3.hash(x,i) % self.m] = True

    def add_set(self, elts):
        for elt in elts:
            self.add(elt)

    def contains(self, x):
        """
        Check if x is in the filter
        """
        for i in range(self.k):
            if self.bits[mmh3.hash(x,i) % self.m] == False:
                return False
        return True

    @classmethod
    def get_m(self, n, err):
        """
        Compute m, the size of the bit array

        m = (n * lg(1/e))/ln(2)
        """
        m = (n * -log2(err))/log(2)
        return int(m)
    
    @classmethod
    def get_k(self, n, m):
        """
        Compute k, the number of hash functions to use

        k = m/n * ln(2)
        """
        k = m/n * log(2)
        return int(k)

    def __len__(self):
        return self.m

    def __str__(self):
        return ("[Bloom] size={}, n={}, err={}, m={}, k={}"
                .format(len(self), self.n, self.err, self.m, self.k))

class WordBloom:
    
    def __init__(self, bloom):
        self.bloom = bloom

    def add_set(self, elts):
        """
        Add set of elements to the filter
        """
        self.bloom.add_set([x.filter_type for x in elts])

    def contains(self, x):
        """
        Check if x is in the filter
        """
        return self.bloom.contains(x.filter_type)

    def __len__(self):
        return len(self.bloom)

    def __str__(self):
        return "[WordBloom] {}".format(self.bloom)
