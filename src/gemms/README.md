# GEMMS

## Naive

Single-threaded. Uses naive index ordering (completes each element in a column of the output, then moves to the next column). Very slow.

## Reordered

Single-threaded. Uses optimal index ordering: j, k, then i.

## Tiled_a

Single-threaded. Divides the left, right and output matrices into tiles of some size and performs the computation for each tile sequentially.

The size of the tile is a tunable parameter.

## Tiled_b

Single-threaded. Recursively divides the left, right and output matrices into smaller tiles until some size is reached.

The size of the tile is a tunable parameter.

## Multithreaded_a

The same as Tiled_a, except that tiles are computed concurrently over different tile indices. No attempt is made to parallelize over tiles split by k.

The size of the tile and number of threads are tunable parameters.

## Multithreaded_b

The same as Tiled_b, except that tiles are computed concurrently over different recursive calls. Again, no attempt is made to parallelize over tiles split by k.

The size of the tile and number of threads are tunable parameters.

## Multithreaded_c

The same as Reordered. However, the k index is moved to the inner loop to avoid atomic operations. In practice, this massively slows down the implementation.

The number of threads is a tunable parameter.

### [Main README](../../README.md)
### [src README](../README.md)
### [Results](../../RESULTS.md)