# GPU Memory Access Pattern Analysis

本文档分析了四种不同 workload 的内存访问模式，并给出最佳 eviction 策略建议。

## 总览对比

| Workload | 访问模式 | Chunk 复用次数 | Chunk 生命周期 | Eviction 压力 | 推荐策略 |
|----------|---------|---------------|---------------|--------------|---------|
| llama.cpp-prefill | 顺序扫描，周期性 | 中 (46) | 长 (12.4s) | 低 (0) | FIFO |
| llama.cpp-decode | 随机散点 | 低 (5) | 中 (7.5s) | 低 (0) | LRU |
| faiss-build | 多条顺序流 | 低 (10) | 长 (22s) | 低 (0) | FIFO |
| faiss-query | 全随机，高密度 | 高 (95) | 长 (28s) | 高 (~8000/s) | LFU |

---

## 1. llama.cpp-prefill (LLM Prefill 阶段)

### 访问模式特征
```
Virtual Address Space Access Pattern:
- 清晰的斜线模式 → 顺序扫描
- 5-6 个周期性的扫描（每个周期约 3000ms）
- 热点集中在固定 VA 范围 (0x7e0e...到 0x7e1c...)
```

### 关键指标
- **VA Block 访问频率**: 中位数 36 次
- **Chunk 复用次数**: 中位数 46 次
- **Chunk 生命周期**: 中位数 12416ms
- **Eviction 压力**: 0（无 eviction）
- **Avg VAs/Chunk**: 5.2
- **Avg Chunks/VA**: 3.8

### Hook Event Timeline
- 0-3000ms: 高 ACTIVATE（初始加载）
- 3000ms+: EVICTION_PREPARE 开始，稳定在 ~400/interval
- POPULATE 和 ACTIVATE 有周期性波动

### 推荐策略: **FIFO**
```
原因:
1. 顺序扫描模式 → 最早进入的最先不需要
2. 周期性访问 → 每个周期扫描相同区域
3. 无 eviction 压力 → 策略影响有限
4. FIFO 开销最低（不需要在 chunk_used 时移动）
```

---

## 2. llama.cpp-decode (LLM Decode 阶段)

### 访问模式特征
```
Virtual Address Space Access Pattern:
- 前 7000ms: 有一定顺序性（斜线）
- 7000ms+: 变成随机散点
- 整体稀疏，VA 范围分散
```

### 关键指标
- **VA Block 访问频率**: 中位数 3 次（很低！）
- **Chunk 复用次数**: 中位数 5 次
- **Chunk 生命周期**: 双峰分布（~0ms 和 ~7500ms）
- **Eviction 压力**: 0
- **Avg VAs/Chunk**: 1.8
- **Avg Chunks/VA**: 1.4

### 特殊观察
- Reuse vs Lifetime 图显示两个 cluster：
  - 短生命周期 + 高访问（左上）
  - 长生命周期 + 中等访问（右侧）
- 这表明有"临时"和"持久"两类 chunk

### 推荐策略: **LRU**
```
原因:
1. 随机访问模式 → 需要保护最近访问的数据
2. 低复用次数 → 不需要 LFU 的频率追踪
3. 两类 chunk → LRU 能自然区分（短命的快速 evict）
4. 访问频率低 → chunk_used 调用少，LRU 开销可接受
```

---

## 3. faiss-build (向量索引构建)

### 访问模式特征
```
Virtual Address Space Access Pattern:
- 多条平行斜线 → 多个顺序流同时进行
- 随时间推移，访问范围扩大
- 热力图显示条带状模式
```

### 关键指标
- **VA Block 访问频率**: 中位数 6 次
- **Chunk 复用次数**: 中位数 10 次（但有长尾到 600+）
- **Chunk 生命周期**: 中位数 22088ms
- **Eviction 压力**: 0
- **Avg VAs/Chunk**: 4.1
- **Avg Chunks/VA**: 2.8

### 特殊观察
- Reuse vs Lifetime 图呈现正相关：生命周期越长，访问次数越多
- 有少量"超级热点"chunk（访问 400-600 次）
- Hook Timeline 显示初始阶段 ACTIVATE 很高，然后稳定

### 推荐策略: **FIFO** 或 **LFU**
```
原因:
1. 顺序流模式 → FIFO 适合
2. 有热点 chunk → LFU 能保护它们
3. 无 eviction 压力 → 策略影响有限

建议:
- 如果内存充足: FIFO（低开销）
- 如果内存紧张: LFU（保护热点）
```

---

## 4. faiss-query (向量查询)

### 访问模式特征
```
Virtual Address Space Access Pattern:
- 完全随机！密密麻麻的散点
- 整个 VA 空间被均匀访问
- 热力图显示全域高热度
```

### 关键指标
- **VA Block 访问频率**: 中位数 54 次
- **Chunk 复用次数**: 中位数 95 次（非常高！）
- **Chunk 生命周期**: 中位数 28249ms
- **Eviction 压力**: 极高！~8000 events/sec，累计 220000+
- **Avg VAs/Chunk**: 15.2
- **Avg Chunks/VA**: 9.1

### 特殊观察
- 这是唯一有显著 eviction 压力的 workload
- Eviction Rate 稳定在 8000/sec
- 所有 chunk 访问次数都很高（80-120 次）
- 生命周期分布紧凑（26000-30000ms）

### 推荐策略: **LFU**
```
原因:
1. 全随机访问 → FIFO/MRU 无法预测
2. 高复用次数 → 频率信息有价值
3. 高 eviction 压力 → 策略选择很重要！
4. 均匀访问模式 → LRU 效果有限

LFU 优势:
- 保护高频访问的 chunk
- 在均匀访问下退化为 FIFO（可接受）
```

---

## 策略选择决策树

```
                    访问模式?
                       │
           ┌──────────┼──────────┐
           │          │          │
        顺序扫描    随机但有热点   完全随机
           │          │          │
           ↓          ↓          ↓
         FIFO       LRU        LFU
           │          │          │
    (低开销，顺序     (保护最近    (保护高频
     数据用完即弃)    访问的数据)   访问的数据)


    特殊情况:
    ─────────────────────────────────────────
    如果 eviction 压力 = 0:
    → 任何策略效果相似，选 FIFO（最低开销）

    如果访问顺序 = 进入顺序:
    → FIFO = MRU，选 FIFO

    如果有明显的 hot/cold 分离:
    → LFU 或 Frequency-threshold 策略
```

---

## MRU 适用场景分析

基于以上分析，**MRU 在这四个 workload 中都不是最优选择**：

| Workload | MRU 效果 | 原因 |
|----------|---------|------|
| llama-prefill | ≈ FIFO | 顺序访问，但 FIFO 开销更低 |
| llama-decode | 差 | 随机访问，会误杀需要的数据 |
| faiss-build | ≈ FIFO | 顺序流，但 FIFO 更简单 |
| faiss-query | 差 | 全随机，会误杀热点数据 |

### MRU 真正适合的场景
```
1. 严格单 pass 处理（每个数据只处理一次）
2. 处理完立即写回（写回 = 最后一次访问）
3. 并行处理多个 chunk，完成顺序不确定
4. 不需要保护任何热点数据

例如:
- 视频帧处理（每帧独立，处理完就丢）
- 日志扫描（扫一遍不回头）
- ETL 单 pass 转换
```

---

## 实现建议

### 1. 默认策略选择
```
if (workload == "llm_inference") {
    if (phase == "prefill") return FIFO;
    else return LRU;
} else if (workload == "vector_search") {
    if (operation == "build") return FIFO;
    else return LFU;  // query
} else {
    return LRU;  // 保守默认
}
```

### 2. 自适应策略
```
监控指标:
- eviction_rate: 高 → 策略选择重要
- access_pattern: 顺序 vs 随机
- chunk_reuse_count: 高 → LFU 有价值

动态切换:
- 检测到顺序模式 → FIFO
- 检测到随机 + 高复用 → LFU
- 检测到随机 + 低复用 → LRU
```

### 3. Prefetch 策略配合
```
所有 workload 都显示 prefetch_always_max 效果好：
- prefill: 顺序访问，prefetch 命中率高
- decode: 虽然随机，但热点数据 prefetch 后能重复使用
- faiss: 高复用意味着 prefetch 的数据会被多次访问

结论: prefetch_always_max + 合适的 eviction 策略
```

---

## 总结

| 策略 | 最佳场景 | 开销 | 复杂度 |
|------|---------|------|--------|
| FIFO | 顺序扫描 | 最低 | 最简单 |
| LRU | 随机 + 时间局部性 | 中 | 简单 |
| LFU | 随机 + 频率局部性 | 高 | 复杂 |
| MRU | 单 pass streaming | 中 | 简单 |

**关键洞察**:
1. 大多数 workload 的 eviction 压力为 0，策略选择影响有限
2. faiss-query 是唯一高 eviction 压力的场景，策略选择关键
3. prefetch 策略比 eviction 策略对性能影响更大
4. 简单策略（FIFO）往往足够好，除非有明确的热点模式
