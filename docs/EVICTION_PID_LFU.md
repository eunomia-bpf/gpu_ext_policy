# PID-based Quota Eviction Policy

## Overview

`eviction_pid_lfu` implements a quota-based GPU memory eviction policy using BPF struct_ops. It allows differentiated memory management for different processes based on configurable quota percentages.

## Strategy

Each process is assigned a quota (percentage of total active chunks):
- **Within quota**: Chunk is moved to tail of LRU list (protected from eviction)
- **Over quota**: Chunk is not moved (easier to evict)

This creates differentiated memory residency based on quota allocation, giving high-priority processes more protection from eviction.

## How It Works

### Key Callbacks

1. **`uvm_pmm_chunk_activate`**: Called when a new chunk is activated
   - Increments `current_count` for the owning PID
   - Tracks total activations

2. **`uvm_pmm_chunk_used`**: Called when a chunk is accessed
   - Checks if current active chunks for this PID are within quota
   - If within quota: calls `bpf_uvm_pmm_chunk_move_tail()` to protect chunk
   - If over quota: does not move chunk (easier to evict)

3. **`uvm_pmm_eviction_prepare`**: Called when eviction is about to happen
   - Decrements `current_count` for the chunk being evicted
   - Maintains accurate count of current active chunks

### Quota Calculation

```
quota_chunks = total_active_chunks * quota_percent / 100
```

Where:
- `total_active_chunks` = sum of `current_count` for high priority + low priority PIDs
- `quota_percent` = configured percentage (0-100)

## Usage

```bash
sudo ./eviction_pid_lfu [options]

Options:
  -p PID     Set high priority PID
  -P PERCENT Set high priority quota percentage (0-100, 0=unlimited)
  -l PID     Set low priority PID
  -L PERCENT Set low priority quota percentage (0-100)
  -d PERCENT Set default quota percentage for other PIDs (0=unlimited)
  -h         Show help
```

### Example

```bash
# High priority process gets 80% quota, low priority gets 20%
sudo ./eviction_pid_lfu -p 12345 -P 80 -l 67890 -L 20
```

## Statistics

The policy tracks per-PID statistics:

| Metric | Description |
|--------|-------------|
| `current_count` | Current number of active chunks for this PID |
| `total_activate` | Total chunks activated (cumulative) |
| `total_used` | Total chunk_used calls (cumulative) |
| `in_quota` | Times chunk was within quota (moved to tail) |
| `over_quota` | Times chunk was over quota (not moved) |

Statistics are printed every 5 seconds and on exit.

## Test Results

Example test with two competing processes (size_factor=0.6, rand_stream kernel):

### Configuration
- High priority: 80% quota
- Low priority: 20% quota

### Results

| Metric | High Priority (80%) | Low Priority (20%) |
|--------|--------------------|--------------------|
| Median time | 723 ms | 1229 ms |
| In quota (moved) | 96.5% | 0.7% |
| Over quota | 3.5% | 99.3% |
| Speedup | 1.7x faster | baseline |

### Key Observations

1. High priority process has 96.5% of its chunk accesses protected
2. Low priority process has 99.3% of accesses unprotected
3. High priority achieves 1.7x better performance
4. `current_count` correctly tracks active chunks (~15K total)

## Testing

Use the test script:

```bash
cd /home/yunwei37/workspace/gpu/co-processor-demo/memory/micro
python3 test_pid_lfu.py -P 80 -L 20 -k rand_stream
```

Options:
- `-P/--high-quota`: High priority quota percentage (default: 80)
- `-L/--low-quota`: Low priority quota percentage (default: 20)
- `-k/--kernel`: Access pattern kernel (seq_stream, rand_stream, pointer_chase)

## Files

| File | Description |
|------|-------------|
| `src/eviction_pid_lfu.bpf.c` | BPF program implementing the policy |
| `src/eviction_pid_lfu.c` | Userspace loader and statistics |
| `memory/micro/test_pid_lfu.py` | Test script for benchmarking |

## Data Structures

### `pid_chunk_stats` (per-PID tracking)

```c
struct pid_chunk_stats {
    u64 current_count;      /* Current active chunk count */
    u64 total_activate;     /* Total chunks activated */
    u64 total_used;         /* Total chunk_used calls */
    u64 in_quota;           /* Times within quota (moved) */
    u64 over_quota;         /* Times over quota (not moved) */
};
```

### BPF Maps

| Map | Type | Description |
|-----|------|-------------|
| `config` | ARRAY | Configuration (PIDs, quotas) |
| `pid_chunk_count` | HASH | Per-PID statistics (key: PID) |

## Design Notes

1. **Dynamic quota**: Quota is calculated based on current active chunks, not cumulative. This adapts to changing workload.

2. **Eviction tracking**: `eviction_prepare` decrements `current_count` to maintain accurate active chunk count.

3. **Total calculation**: `get_total_active_chunks()` sums only the configured PIDs' counts for quota calculation.

4. **Move semantics**: `bpf_uvm_pmm_chunk_move_tail()` moves chunk to tail of LRU list, making it the "most recently used" and least likely to be evicted.
