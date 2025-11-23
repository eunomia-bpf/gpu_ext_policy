# results

sudo /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src/prefetch_always_max

```
$ /home/yunwei37/workspace/gpu/co-processor-demo/memory/micro/uvmbench  --kernel=rand_stream --mode=uvm --size_factor=1.2 --iterations=10
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.2 (oversubscription)
Total Working Set: 38531 MB
Stride Bytes: 4096 (page-level)
Kernel: rand_stream
Mode: uvm
Iterations: 10


Results:
  Kernel: rand_stream
  Mode: uvm
  Working Set: 38531 MB
  Bytes Accessed: 38531 MB
  Median time: 1161.96 ms
  Min time: 841.629 ms
  Max time: 1217.33 ms
  Bandwidth: 34.7715 GB/s
  Results written to: results.csv
```

default: no custom policy

```
yunwei37@lab:~/workspace/gpu$ /home/yunwei37/workspace/gpu/co-processor-demo/memory/micro/uvmbench  --kernel=rand_stream --mode=uvm --size_factor=1.2 --iterations=10
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.2 (oversubscription)
Total Working Set: 38531 MB
Stride Bytes: 4096 (page-level)
Kernel: rand_stream
Mode: uvm
Iterations: 10


Results:
  Kernel: rand_stream
  Mode: uvm
  Working Set: 38531 MB
  Bytes Accessed: 38531 MB
  Median time: 1805.74 ms
  Min time: 795.411 ms
  Max time: 1890.62 ms
  Bandwidth: 22.3748 GB/s
  Results written to: results.csv
```

 sudo /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src/prefetch_none 

