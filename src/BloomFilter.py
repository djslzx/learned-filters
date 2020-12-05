  
# https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/
import math
import mmh3
from bitarray import bitarray
 
 
class BloomFilter(object):
 
    '''
    Class for Bloom filter, using murmur3 hash function
    '''
 
    def __init__(self, num_items, fp_prob):
        '''
        num_items : int
            Number of items expected to be stored in bloom filter
        fp_prob : float
            False Positive probability in decimal
        '''
        # False posible probability
        self.fp_prob = fp_prob
        self.size = self.get_size(num_items, fp_prob)
        self.hash_count = self.get_hash_count(self.size, num_items)
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)
 
    def add(self, item):
        '''
        Add an item in the filter
        '''
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            self.bit_array[digest] = True
 
    def contains(self, item):
        '''
        Check for existence of an item in filter
        '''
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            if self.bit_array[digest] == False:
                return False
        return True
 
    @classmethod
    def get_size(self, n, p):
        '''
        Return the size of bit array(m) to used using
        following formula
        m = -(n * lg(p)) / (lg(2)^2)
        n : int
            number of items expected to be stored in filter
        p : float
            False Positive probability in decimal
        '''
        m = -(n * math.log(p))/(math.log(2)**2)
        return int(m)
 
    @classmethod
    def get_hash_count(self, m, n):
        '''
        Return the hash function(k) to be used using
        following formula
        k = (m/n) * lg(2)
 
        m : int
            size of bit array
        n : int
            number of items expected to be stored in filter
        '''
        k = (m/n) * math.log(2)
        return int(k)