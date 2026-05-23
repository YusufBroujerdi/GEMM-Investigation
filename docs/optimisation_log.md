### 22-05-26

* I will begin by benchmarking GEMM with naive indexing vs. reordered.

* Reording the indexes to access adjacent memory addresses in succession should make better use of the processor's cache, which is typically the main bottleneck.

* However, the compiler may reorder the indexes for me.