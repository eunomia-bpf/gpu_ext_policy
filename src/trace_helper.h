/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright (c) 2025 */
/*
 * Trace Helper - Helper functions for extracting process information from UVM chunks
 *
 * Data structure path:
 *   chunk -> va_block -> managed_range -> va_range.va_space
 *         -> va_space_mm.mm -> owner (task_struct*) -> tgid
 */

#ifndef __TRACE_HELPER_H__
#define __TRACE_HELPER_H__

#include <bpf/bpf_core_read.h>
#include "uvm_types.h"

/*
 * get_owner_pid_from_chunk - Get owner PID from a GPU chunk
 * @chunk: pointer to uvm_gpu_chunk_struct
 *
 * Returns: owner process PID (tgid), or 0 if not available
 */
static __always_inline u32 get_owner_pid_from_chunk(uvm_gpu_chunk_t *chunk)
{
    uvm_va_block_t *va_block;
    uvm_va_range_managed_t *managed_range;
    uvm_va_space_t *va_space;
    struct mm_struct *mm;
    struct task_struct *owner;

    if (!chunk)
        return 0;

    va_block = BPF_CORE_READ(chunk, va_block);
    if (!va_block)
        return 0;

    managed_range = BPF_CORE_READ(va_block, managed_range);
    if (!managed_range)
        return 0;

    va_space = BPF_CORE_READ(managed_range, va_range.va_space);
    if (!va_space)
        return 0;

    mm = BPF_CORE_READ(va_space, va_space_mm.mm);
    if (!mm)
        return 0;

    owner = BPF_CORE_READ(mm, owner);
    if (!owner)
        return 0;

    return BPF_CORE_READ(owner, tgid);
}

#endif /* __TRACE_HELPER_H__ */
