import random
from itertools import combinations
# subset_index = sum([list(map(list, combinations(non_zero_index, i))) for i in range(len(non_zero_index) + 1)], [])
list(map(list, combinations(non_zero_index, i))) for i in range(len(non_zero_index) + 1)

def random_combination(self, iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)