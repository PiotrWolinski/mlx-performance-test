import time

import mlx.core as mx
import numpy as np


def dot_np(a_size: list[int], b_size: list[int]) -> float:
    assert a_size[1] == b_size[0]

    start = time.time()

    a = np.random.uniform(size=a_size)
    b = np.random.uniform(size=b_size)

    # Numpy has eager evaluation, so nothing to do apart from
    # running the computations
    _ = a @ b

    elapsed_time = time.time() - start

    return elapsed_time


def dot_mlx(a_size: list[int], b_size: list[int]) -> float:
    assert a_size[1] == b_size[0]

    start = time.time()

    a = mx.random.uniform(shape=a_size)
    b = mx.random.uniform(shape=b_size)

    # MLX offers lazy evaluation, so in order to actually compute
    # results, we need to explicetely say to evaluate the result
    c = a @ b
    mx.eval(c)

    elapsed_time = time.time() - start

    return elapsed_time


def main():
    size = [10_000, 10_000]
    size = [10_000, 10_000]
    repeat = 10
    mlx_time_avg = sum([dot_mlx(size, size) for _ in range(repeat)]) / repeat
    np_time_avg = sum([dot_np(size, size) for _ in range(repeat)]) / repeat


    print(f"Average mlx time = {mlx_time_avg}")
    print(f"Average np time = {np_time_avg}")
    print(f"mlx is on average {np_time_avg / mlx_time_avg:2f} times faster")


if __name__ == "__main__":
    main()
