"""
Defines common utility functions used across files.
"""
from random import sample

def ilen(iterable):
    return sum(1 for _ in iterable)

def shuffled(l):
    """Return shuffled version of a list instead of shuffling in place"""
    return sample(l, k=len(l))
