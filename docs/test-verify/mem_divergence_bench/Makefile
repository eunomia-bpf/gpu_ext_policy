# GPU Memory Access & Thread Divergence Benchmark Makefile

NVCC = nvcc
NVCC_FLAGS = -O3 -Wno-deprecated-gpu-targets

# Default target
all: bench

bench: bench.cu
	$(NVCC) $(NVCC_FLAGS) -o bench bench.cu -lm

run: bench
	./bench

analyze: results.csv
	python3 analyze.py

clean:
	rm -f bench results.csv results.png

.PHONY: all run analyze clean
