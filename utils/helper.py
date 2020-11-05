import time

from typing import Union

Number = Union[float, int]


def timing(f):
    def wrap(*args, **kwargs):
        since = time.perf_counter()
        ret = f(*args, **kwargs)
        print(f'{f.__name__:s} function took {time.perf_counter() - since:.3f} s')

        return ret

    return wrap
