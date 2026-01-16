# Makefile for GPU eBPF Verification Gap Examples

NVCC := nvcc
NVCC_FLAGS := -O2 -arch=sm_61 -Wno-deprecated-gpu-targets

# Detect CUDA installation
CUDA_PATH ?= /usr/local/cuda
ifneq ($(wildcard $(CUDA_PATH)/bin/nvcc),)
    NVCC := $(CUDA_PATH)/bin/nvcc
endif

EXAMPLES := 01_thread_divergence \
            02_memory_access \
            03_atomic_contention \
            04_deadlock \
            05_map_bandwidth_contention

all: check_cuda $(EXAMPLES)

check_cuda:
	@which nvcc > /dev/null 2>&1 || (echo "Error: nvcc not found" && exit 1)
	@echo "NVCC: $$(which nvcc)"
	@echo ""

01_thread_divergence: 01_thread_divergence.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

02_memory_access: 02_memory_access.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

03_atomic_contention: 03_atomic_contention.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

04_deadlock: 04_deadlock.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

05_map_bandwidth_contention: 05_map_bandwidth_contention.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: all
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	./01_thread_divergence
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	./02_memory_access
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	./03_atomic_contention
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	./04_deadlock
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	./05_map_bandwidth_contention

run1: 01_thread_divergence
	./01_thread_divergence

run2: 02_memory_access
	./02_memory_access

run3: 03_atomic_contention
	./03_atomic_contention

run4: 04_deadlock
	./04_deadlock

run5: 05_map_bandwidth_contention
	./05_map_bandwidth_contention

clean:
	rm -f $(EXAMPLES)

.PHONY: all check_cuda run run1 run2 run3 run4 run5 clean
