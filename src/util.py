"""
Defines common utility functions used across files.
"""
from random import sample

def shuffled(l):
    """Return shuffled version of a list instead of shuffling in place"""
    return sample(l, k=len(l))
