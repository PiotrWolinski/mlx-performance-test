# MLX performance comparison

## Overview

Sole purpose of this repository is to play around with [MLX](https://ml-explore.github.io/mlx/build/html/index.html), to see what are the possible speed ups on the Apple silicon.  
So far it only contains simple matrix multiplication, but I have plans to grow this collection further.

## Setup

Those simple tests are run on the 2021 14" MacBook Pro with M1 Pro (8 cores) processor and 32GB of memory.

## Results

### Dot product

Dot product using mlx turns out to be on the average (over 10 runs) **12.5 times faster** than numpy implementation.

## Test it yourself

Simply install dependencies using `poetry` and run chosen test.
