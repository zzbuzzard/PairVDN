import numpy as np
from typing import List, Tuple, Dict


class ExperienceBuffer:
    """A simple, circular buffer containing several named arrays."""
    def __init__(self, length: int, keys: List[str], key_shapes: List[Tuple], key_dtypes: List[type]):
        self.keys = keys
        self.data = []
        for key, shape, dtype in zip(keys, key_shapes, key_dtypes):
            shape = (length,) + shape
            empty = np.zeros(shape, dtype=dtype)
            self.data.append(empty)

        self.index = 0
        self.length = length
        self.full = False

    def __len__(self):
        if self.full:
            return self.length
        return self.index

    def add_experiences(self, **kwargs):
        """Pass data in format {key1: batched data1, key2: batched data2, ...} with equal batch dim"""
        for key, buf in zip(self.keys, self.data):
            d = kwargs[key]
            n = d.shape[0]

            if self.index + n <= self.length:
                buf[self.index: self.index + n] = d
            else:
                # handle wraparound
                count1 = self.length - self.index
                count2 = n - count1
                buf[self.index:] = d[:count1]
                buf[:count2] = d[count1:]

        self.index += n
        if self.index >= self.length:
            self.full = True
            self.index %= self.length


def batched_dataloader(buffer: ExperienceBuffer, batch_size: int, shuffle: bool = True, drop_last: bool = False):
    """
    This generator shuffles and loads experiences from an ExperienceBuffer. Shuffling is naive and not at all cache
    friendly.
    """
    nkeys = len(buffer.keys)
    while True:
        max_ind = len(buffer)  # exclusive
        indices = np.random.permutation(max_ind)

        for i in range(0, max_ind, batch_size):
            j = min(i + batch_size, max_ind)

            if drop_last and i + batch_size > max_ind:
                break

            inds = indices[i:j]

            yield {buffer.keys[i]: buffer.data[i][inds] for i in range(nkeys)}
