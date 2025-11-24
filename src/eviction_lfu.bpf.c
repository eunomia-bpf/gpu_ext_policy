/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"
#include "bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* LFU (Least Frequently Used) Eviction Policy
 *
 * 经典 LFU Cache 算法改编到 GPU chunk 管理：
 *
 * 数据结构（对应经典实现）：
 * 1. chunk_freq: chunk_addr -> freq（类似 key_table）
 * 2. freq_to_chunk: freq -> chunk_addr（类似 freq_table，但简化为只记录一个代表）
 * 3. min_freq: 全局最小频率
 *
 * 核心操作：
 * - activate: 插入新 chunk，freq=1，min_freq=1
 * - chunk_used: increase_freq()，freq++，更新 min_freq
 * - eviction: 优先 evict freq=min_freq 的 chunks
 *
 * 限制：
 * - BPF 无法维护双向链表（freq_table 中的 DoublyLinkedList）
 * - 用 freq_to_chunk[freq] 只记录该频率的"某个" chunk（而不是全部）
 * - 同频率下的 LRU 顺序靠 UVM list 的位置维护
 */

#define MAX_FREQ 255

/* Map 1: chunk -> 频率 (对应 key_table) */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 100000);
    __type(key, u64);    // chunk address
    __type(value, u32);  // frequency
} chunk_freq SEC(".maps");

/* Map 2: freq -> 某个该频率的 chunk (简化的 freq_table) */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, MAX_FREQ + 1);
    __type(key, u32);    // frequency (0-255)
    __type(value, u64);  // chunk address (0 = empty)
} freq_to_chunk SEC(".maps");

/* Map 3: 全局状态（min_freq） */
struct lfu_state {
    u32 min_freq;        // 当前最小频率（对应经典 LFU 的 min_freq）
    u32 total_chunks;    // 统计信息
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct lfu_state);
} lfu_global SEC(".maps");

/* 获取全局状态 */
static __always_inline struct lfu_state *get_state(void)
{
    u32 key = 0;
    struct lfu_state *state = bpf_map_lookup_elem(&lfu_global, &key);
    if (!state) {
        struct lfu_state init = { .min_freq = 1, .total_chunks = 0 };
        bpf_map_update_elem(&lfu_global, &key, &init, BPF_ANY);
        state = bpf_map_lookup_elem(&lfu_global, &key);
    }
    return state;
}

/* 清理旧频率桶（如果该 chunk 是这个桶的代表） */
static __always_inline void clean_old_freq_bucket(u64 chunk_addr, u32 old_freq)
{
    u64 *bucket_chunk = bpf_map_lookup_elem(&freq_to_chunk, &old_freq);
    if (bucket_chunk && *bucket_chunk == chunk_addr) {
        // 这个 chunk 是旧频率桶的代表，清空它
        u64 zero = 0;
        bpf_map_update_elem(&freq_to_chunk, &old_freq, &zero, BPF_ANY);
    }
}

/* 增加 chunk 的频率（对应经典 LFU 的 _increase_freq）*/
static __always_inline void increase_freq(u64 chunk_addr, uvm_gpu_chunk_t *chunk, struct list_head *list)
{
    u32 *freq = bpf_map_lookup_elem(&chunk_freq, &chunk_addr);
    if (!freq)
        return;

    u32 old_freq = *freq;
    u32 new_freq = old_freq + 1;

    if (new_freq > MAX_FREQ)
        new_freq = MAX_FREQ;

    // 1. 从旧频率桶中移除（如果是代表，则清空桶）
    clean_old_freq_bucket(chunk_addr, old_freq);

    // 2. 更新频率
    bpf_map_update_elem(&chunk_freq, &chunk_addr, &new_freq, BPF_ANY);

    // 3. 添加到新频率桶
    bpf_map_update_elem(&freq_to_chunk, &new_freq, &chunk_addr, BPF_ANY);

    // 4. 更新 min_freq（对应经典 LFU 逻辑）
    struct lfu_state *state = get_state();
    if (state && old_freq == state->min_freq) {
        // 检查旧频率桶是否空了
        u64 *check = bpf_map_lookup_elem(&freq_to_chunk, &old_freq);
        if (!check || *check == 0) {
            // 桶空了，min_freq++
            state->min_freq = old_freq + 1;
        }
    }

    // 5. 根据新频率调整在 UVM list 中的位置
    //
    // 理想情况：list 应该严格按频率排序 (HEAD=freq低, TAIL=freq高)
    // 问题：BPF 只有 move_head/move_tail，没有 insert_after/insert_before
    //
    // Workaround: 所有 freq > old_freq 的 chunk 都移到 TAIL
    // 这样能保证大致的顺序：低频在 HEAD，高频在 TAIL
    // 但同频率内的顺序是 LRU（最近访问的在后面）
    //
    // TODO: 需要新 kfunc bpf_uvm_pmm_chunk_insert_after() 才能做严格排序

    if (new_freq > old_freq) {
        // 频率增加了，移到 TAIL（让更高频的有机会在前面）
        bpf_uvm_pmm_chunk_move_tail(chunk, list);
    }
    // 频率没变 -> 不移动（保持原位）
}

SEC("struct_ops/uvm_pmm_chunk_activate")
int BPF_PROG(uvm_pmm_chunk_activate,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    u64 addr = (u64)chunk;
    u32 freq = 1;
    struct lfu_state *state = get_state();

    // 对应经典 LFU 的 put() 插入新元素：freq = 1
    bpf_map_update_elem(&chunk_freq, &addr, &freq, BPF_ANY);
    bpf_map_update_elem(&freq_to_chunk, &freq, &addr, BPF_ANY);

    if (state) {
        state->min_freq = 1;  // 插入新元素必定 min_freq = 1
        state->total_chunks++;
    }

    // 新 chunk 是最低频，放在 HEAD（容易被 evict）
    bpf_uvm_pmm_chunk_move_head(chunk, list);

    return 1; // BYPASS
}

SEC("struct_ops/uvm_pmm_chunk_used")
int BPF_PROG(uvm_pmm_chunk_used,
             uvm_pmm_gpu_t *pmm,
             uvm_gpu_chunk_t *chunk,
             struct list_head *list)
{
    u64 addr = (u64)chunk;

    // 对应经典 LFU 的 get() 或 put() 已存在的 key：调用 _increase_freq
    increase_freq(addr, chunk, list);

    return 1; // BYPASS
}

SEC("struct_ops/uvm_pmm_eviction_prepare")
int BPF_PROG(uvm_pmm_eviction_prepare,
             uvm_pmm_gpu_t *pmm,
             struct list_head *va_block_used,
             struct list_head *va_block_unused)
{
    struct lfu_state *state = get_state();
    if (!state)
        return 0;

    u32 min_freq = state->min_freq;

    /* 对应经典 LFU 的淘汰逻辑：
     *
     * 经典实现：
     *   min_list = freq_table[min_freq]
     *   node_to_evict = min_list.pop_right()  // 尾部 = 最久未用
     *
     * 我们的实现：
     *   - freq_to_chunk[min_freq] 存了该频率的某个 chunk
     *   - UVM list 上，低频 chunks 已经在 HEAD 附近（increase_freq 调整的）
     *   - 内核从 HEAD evict，自然选中低频的
     *   - 同频率下，HEAD->TAIL 相当于 LRU（老的在 HEAD，新的在 TAIL）
     *
     * 这里只做 logging 验证逻辑
     */

    bpf_printk("BPF LFU: eviction, min_freq=%u\n", min_freq);

    // 查找 min_freq 的代表 chunk（debug 用）
    u64 *victim_addr = bpf_map_lookup_elem(&freq_to_chunk, &min_freq);
    if (victim_addr && *victim_addr != 0) {
        bpf_printk("BPF LFU: victim at freq=%u, addr=%llx\n", min_freq, *victim_addr);
    }

    return 0;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_lfu_clean = {
    .uvm_bpf_test_trigger_kfunc = (void *)NULL,
    .uvm_prefetch_before_compute = (void *)NULL,
    .uvm_prefetch_on_tree_iter = (void *)NULL,
    .uvm_pmm_chunk_activate = (void *)uvm_pmm_chunk_activate,
    .uvm_pmm_chunk_used = (void *)uvm_pmm_chunk_used,
    .uvm_pmm_eviction_prepare = (void *)uvm_pmm_eviction_prepare,
};
