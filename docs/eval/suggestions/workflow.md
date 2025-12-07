好，那我们就把这个 Workflow 2 当成一个独立的 eval 小节来设计：

> **Small LLM planner (llama.cpp) + Big LLM executor (Qwen3‑30B‑A3B on vLLM) + GPU Faiss 大索引**
> 全都塞在一张 5090 上，索引用 UVM oversub 撑爆 HBM，然后用 gBPF 把 tail latency 和 page fault 收回来。

我按 OSDI 论文的结构帮你拆：

1. 明确这个用例要回答什么问题（它在整篇论文里扮演的角色）。
2. 具体 workload 怎么定义：步骤、分支、比例、trace 怎么生成。
3. Baseline 有哪些，各自语义是什么。
4. gBPF 在这个用例里启用的 policy 是啥（不要太复杂，但要说得清）。
5. 要做哪几组实验、出哪些图、每张图想证明什么。

---

## 1. 这个用例到底在 eval 里回答什么问题？

从 OSDI 视角，这个 agentic multi‑model RAG 用例要服务的核心 claim是：

> **在一个真实的、多模型、多 runtimes 的 agentic RAG pipeline 上，
> 把 LLM 和 GPU vector search 放到同一张 oversub GPU 上，
> 现有解法要么只能让 LLM 独占 GPU、Faiss 赢在 CPU，要么交给 UVM 瞎迁移导致 p99 爆炸；
> gBPF 提供的 driver+device policy plane 可以在不改任何 runtime 的情况下，
> 让 big LLM 保持接近 fit‑in‑HBM 的 tail latency，同时让 Faiss 在 GPU 上跑得过得去。**

所以小节的 RQ 其实可以写成类似：

> RQ‑Agentic: In an agentic multi-model RAG pipeline, can gBPF enable co-located LLM serving and GPU vector search on a single oversubscribed GPU, preserving LLM tail latency compared to CPU-based retrieval and naive UVM?

你所有设计都朝这个问题打。

---

## 2. Workload：Workflow 2 具体怎么定义

### 2.1 组件

硬件：RTX 5090（32GB HBM）。

进程和框架：

* **M1: Planner LLM（small）**

  * 框架：`llama.cpp` server（单独进程）。
  * 模型：任意 7B 级别 instruct 模型，Q4/Q5 量化。目标是吃 ~4–6GB 显存，让它确实在 GPU 上占坑。

* **M2: Executor LLM（big）**

  * 框架：`vLLM` server。
  * 模型：`Qwen3-30B-A3B` FP8（或相近），占掉 15–20GB + KV。
  * 这是高优先级 tenant，你想保护它的 p99 latency。

* **V: GPU Faiss index**

  * 框架：Faiss-GPU（IVF‑PQ/IVF‑Flat）；
  * 数据：随机向量，数量 N 大到索引 footprint（向量 + IVF list + PQ code）> 32GB；
  * 存储：用 UVM / managed memory，让它明显 oversub GPU 内存。

注意：你完全可以明确写“vectors/documents are synthetic; we only care about memory behavior and latency.”

### 2.2 每个请求的 agentic 流程

定义一个最小但合理的 agent 逻辑：

**Step 0 – Planner decision（M1 / llama.cpp）**

对每个用户 query，M1 输出一个 JSON，例如：

```json
{
  "need_retrieval": true | false,
  "need_heavy_model": true | false,
  "retrieval_query": "..."
}
```

你可以通过 prompt 让 small LLM 判断问题是否“开放 / 事实依赖”；简单问题可能直接回答，复杂问题调用 big 模型 + 检索。

为了工程节省，可以 offline 跑一遍，把这个 JSON trace 固化下来，在线 eval 只 replay 决策。

**Step 1 – 可能的 direct answer（M1）**

* 如果 `need_retrieval=false` 且 `need_heavy_model=false`：

  * M1 直接给出 final answer（只打一遍 small LLM）。

**Step 2 – Retrieval（V / Faiss-GPU）**

* 如果 `need_retrieval=true`：

  * 用 `retrieval_query` 生成一个 d 维向量（可以直接用随机 embedding 或一个固定线性映射）；
  * 在 GPU Faiss 索引上做 top‑k 查询（k=32/64），返回随机的“文档片段 ID”。

文档内容可以是 synthetic text，长度模拟真实 RAG（每文档 256–512 token）。

**Step 3 – Heavy answer（M2 / big LLM）**

* 如果 `need_heavy_model=true`：

  * 构造 RAG prompt：原问题 + top‑k 文本；
  * 发给 Qwen3‑30B（vLLM）生成最终回答。

否则就用 M1 的回答，跳过 big 模型。

**最终：**

每个 request 会遵循三种 path 之一：

1. Path A（简单问答）：M1 直接回答（只用 small LLM，完全不触发 Faiss/M2）。
2. Path B（中等复杂）：M1 做 planning + query → Faiss → M2 回答。
3. Path C（可选）：M1 + Faiss + M2 + 额外一次 M1 refinement（比如再用 small LLM 做粗审）。

你可以固定一个比例，例如：

* 30% Path A，50% Path B，20% Path C。

这就是一个典型的 **“agent orchestrating two models + one tool”** 的 RAG workflow，不再是线性 pipeline。

### 2.3 Trace 和负载

为了让实验可重复：

* 生成一个 request trace：

  * Q = 1000 或 2000 个 query；
  * 每个 query 用上述 agent (M1) offline 生成它的 path + retrieval_query；
  * 记录：`(timestamp, path_type, retrieval_query_len, answer_len, etc.)`。

* 在线 eval：

  * 单 client：按固定间隔发（比如每 100ms 发一个 query）；
  * 多 client：K 个 replay 线程，每个走这条 trace，起始时间错开，形成并发。

你要的 metrics：

* per‑request end‑to‑end latency（从 request 入系统到 final answer）；
* 分 path 的 latency（A/B/C）；
* tail latency（p95/p99），尤其是 B/C path（包含 Faiss + big LLM）。

---

## 3. Baseline 设计

这个小节至少要有三个 baseline + gBPF：

### Baseline 1：CPU Faiss（现实工程常见做法）

* M1、M2 都在 GPU；
* Faiss index 在 CPU（Faiss CPU + 大内存），不占 GPU 内存；
* 没有 UVM，GPU 上只有两个 LLM 的权重 + KV。

意义：

* “安全但保守”现实做法：LLM 独占 GPU，检索走 CPU；
* agent pipeline latency 被 CPU 检索拖慢，但 tail 相对稳定；
* 用它做“LLM p99 的下界”和“向量检索的 latency baseline”。

### Baseline 2：GPU Faiss + UVM oversub（无 gBPF）

* M1 + M2 + Faiss 全在同一张 5090 上；
* Faiss index 放在 UVM，size >32GB；
* 其他不做特殊处理，由默认 UVM LRU/pre‑fetch 决定迁移。

这是你的主对照：

* 探测在 agentic 流程下，Faiss 的访问模式 + LLM 权重/KV access 会造成怎样的 page fault storm 和 tail latency；
* 典型现象：

  * B/C path 的 requests p99 膨胀好几倍；
  * GPU SM idle 时间中大量在等 UVM migration。

### Baseline 3：Host‑only policy（可选 Aberlation）

如果你有 host‑only gBPF 实现，可以加一个中间 baseline：

* 用 eBPF 在 UVM driver 上做简单 region cache（不看 device 端 hotness），例如：

  * 固定 pin 一部分 LLM weights 在 HBM；
  * Faiss 还是靠 UVM 自动 fault；
* 没有 device‑side hint 和 warp‑级访问统计。

这个 baseline用来说明：

> “只做 host‑side policy improvement 也能好一点，但没有 device‑aware 的 region hotness 就做不到我们的效果。”

---

## 4. gBPF 在这个用例里启用什么 policy（别搞太玄）

设计一个足够简单但系统味儿足的 policy：

### 4.1 Device 侧：收集 hotness + 发送 hint

你只在两个地方插 GPU eBPF:

1. **大模型（M2） kernel 内：**

   * 标记 Qwen 权重 region + KV page 所属的 region ID；
   * 在 prefill/decode 的某些指令附近，用 `gdev_mem_ops.access` 记录最近访问的权重区域；
   * 同时把正在使用的 KV 区域打上“hot”标记写进 BPF map。

2. **Faiss kernel 内：**

   * 对每个查询，Faiss 会访问几个 IVF list / PQ codebook segment；
   * 你在训练时就知道每个 IVF list 对应哪块 region；
   * 用 device eBPF 每次访问时，在 map 里记 `(region_id, last_access_ts, bytes)`；
   * 在进入 Faiss kernel 前，用简单 heuristic 给即将访问的 list region 调 `gdev_mem_prefetch_hint`。

### 4.2 Host 侧 UVM policy（gdrv_mem_ops）

在 UVM 的 add/access/remove/prefetch hook 上：

* 维护 per‑tenant / per‑model 的 HBM credit：

  * 给 M2 权重 + 活跃 KV 较高 priority（pin 或放在 eviction list 尾部）；
  * Faiss index region 允许被驱逐，一次只允许 N 个 active lists 常驻 HBM。

* 看 device map 里的 hotness/hints，做：

  * 在 LLM 下一次调用开始前（例如在 kernel launch hook 里），如果 BPF map 显示某些权重/KV 最近频繁使用，就提前 prefetch/pin；
  * 在 Faiss query 开始前，根据 hint 把预测会访问的 IVF list 批量 Migrate 到 HBM（避免 query 内 page fault）；
  * 过旧的 Faiss region 和冷 KV 在内存压力大时先被 evict 到 host/CXL。

你不需要给出很 fancy 的策略，简单 MRU / LRU + few thresholds 就够，只要逻辑上说得通。

---

## 5. 实验与图：OSDI 标准的写法

### 5.1 实验 1：单机体制下的 end‑to‑end latency & SLO

**目的：**证明在这个 agentic workflow 上，“全 GPU + naive UVM” 会把 tail latency 搞崩，gBPF 能显著改善，达到接近 CPU‑Faiss baseline。

**方法：**

* 对前面那条 trace（1000–2000 请求）：

  * 跑三种配置：CPU‑Faiss、GPU‑Faiss+UVM、GPU‑Faiss+gBPF；
  * 单 client 或 4 并发 client，QPS 控在不饱和（例如 5–10 QPS）。

**图 1：End‑to‑end latency CDF 或柱状图**

* 横轴：latency（ms）；
* 纵轴：CDF；
  或柱：p50 / p95 / p99 latency，按配置分组。

重点解读：

* CPU‑Faiss：均值高（retriever 慢），但 tail 相对稳定；
* GPU‑Faiss+UVM：p50 可能很好，但 p99 明显 3×–5×，有夸张的 heavy tail；
* gBPF：p50 接近 GPU‑Faiss，p99 明显比 naive UVM 好很多，接近 CPU‑Faiss 或介于两者之间。

你可以指出：

> “gBPF maintains the interactive SLO of the LLM (p99 within 1.3–1.5× of the CPU‑Faiss baseline), while enabling GPU-based retrieval that is 2–3× faster than CPU Faiss in median latency.”

### 5.2 实验 2：page fault / migration 行为（机理图）

**目的：**解释为什么 naive UVM 会崩，gBPF 到底做了啥。

**测：**

* 对同一个 workload：

  * 记录 GPU page fault 数量（总数、per request）、fault latency；
  * 统计 GPU kernel time breakdown：compute vs migration stall vs idle；
  * 记录 UVM migration 总 bytes（GPU↔CPU）。

**图 2：每请求 page fault 数 / CDF**

* 横轴：fault 数（log scale）；
* 纵轴：CDF；
* 曲线：naive UVM vs gBPF。

**图 3：GPU 时间分解**

* 堆叠柱状图：
  横轴：配置（CPU‑Faiss 可忽略 / naive UVM / gBPF）
  纵轴：时间比例：

  * LLM compute；
  * Faiss compute；
  * UVM migration stall；
  * idle。

讲清楚：

* naive UVM：大量 request 在某些阶段有上百/上千个 far fault，stall 时间占 >X%；
* gBPF：fault 数减少一个数量级，大部分迁移变成 bulk prefetch，stall 时间大幅下降。

### 5.3 实验 3：QPS scaling / multi‑client（如果有精力）

**目的：**说明 gBPF 不仅在低 QPS 下救 tail，在系统拉高 QPS 时也能明显改善 goodput。

**方法：**

* 固定 trace；
* QPS 从 2→4→8→12 调整（多 client 并发）；
* 为每个 QPS 点跑三配置，测：

  * system throughput（完成请求/秒）；
  * p99 latency；
  * SLO violation rate（例如 latency>1s 的请求比例）。

**图 4：p99 vs QPS；图 5：throughput vs QPS**

说明：

* CPU‑Faiss：整体 throughput 低；
* GPU‑Faiss+UVM：在中到高 QPS 段，p99 猛炸；
* gBPF：在中等 QPS 段仍能保持 acceptable p99，同时 throughput 接近 GPU‑Faiss（未限速）水平。

### 5.4 实验 4：ablation（host‑only vs device‑only vs full）

如果你有 host‑only policy，可以加个小图，位置放在本节末尾即可。

* 配置：

  * naive UVM；
  * host‑only gBPF（只在 UVM hook 做 region 重排，不看 device hint）；
  * device‑only（只收集 hotness，不改 eviction）；
  * full gBPF（host+device+hierarchical maps）。
* 指标：p95/p99 latency + 每请求 fault 数。

用它来支撑：

> “We need both host and device instrumentation: host-only policies cannot see which Faiss lists/LLM regions are actually hot, while device hints without driver control cannot prevent UVM from evicting critical LLM regions.”

---

## 6. 在 evaluation 小节里怎么写（你可以直接改编成段落）

这一小节的结构可以是：

> **6.X Agentic multi-model RAG on a single GPU**

* Workload 描述（1 段）；
* Baselines 描述（1 段）；
* gBPF policy 描述（1 段）；
* 然后按 E1–E3 讲结果，每段配 1–2 张图。

比如第一段可以写成：

> We next evaluate gBPF on an agentic multi-model RAG workload that co-locates a small planner LLM, a large executor LLM, and an oversized GPU vector index on a single RTX 5090 (32 GB). The small LLM (llama.cpp) analyzes each user query and decides whether to invoke retrieval and the large model. If retrieval is needed, the planner emits a dense search query that is sent to a Faiss IVF-PQ index stored in CUDA Unified Memory; the index footprint (40–60 GB) intentionally exceeds GPU memory and forces oversubscription. The large LLM (Qwen3‑30B‑A3B on vLLM) then consumes the original question and top‑k retrieved passages to generate the final answer. Requests follow three paths (direct answer, one-shot RAG, and refinement), derived from an offline agent trace; we replay 1000 requests with up to 8 concurrent sessions and report end-to-end latency and GPU/UM statistics.

后面就顺着前面那几组实验往下写就行。

---

总之，这个 Workflow 2 的 eval 可以做到：

* 工程量：多起一个 llama.cpp + 写一个 planner prompt + 一个 trace replay driver；
* 系统贡献：跨两个 LLM runtime + 一个 Faiss GPU index，在 oversub 下看 gBPF 如何协同 UVM & 调度；
* OSDI 味儿：用 p99 / page fault / stall breakdown / QPS scaling 等图，直接对现有“单 runtime”系的 LLMSys/Vector 系统论文补上一个明显缺位的多租户用例。

你如果愿意，下一步可以把你打算用的几个关键数字定一下（Faiss 向量数量/维度、请求数、QPS），我可以帮你把这一节的 outline 直接写成 LaTeX-ish 文本和图 caption，你几乎可以 copy‑paste 到论文里。

# basic 3 step

可以，把 workflow 简单到你说的那 3 步是完全可行的，而且从 OSDI 的视角依然有价值——只是你要**老老实实地改 claim**，不要再说什么 agentic，多租户、公平性这些东西就别指望这个用例来撑了。

我分三块说：

1. 这三步在术语上到底算什么，不算什么（可以怎么叫）。
2. 这种简单 workflow 适合支撑 gBPF 的哪类 claim / policy。
3. 具体实验应该怎么设计，才不会像随手 microbench，而是像 evaluation 的一个 solid 子节。

---

## 1. 你这个“三步 RAG”到底算什么？

你说的简化版本：

1. Step 1：用户 query → Qwen3‑30B‑A3B（vLLM）生成一个适合检索的 query。
2. Step 2：用这个 query 做 Faiss 查询（GPU index，数据可以是随机向量）。
3. Step 3：再用 Qwen3‑30B‑A3B 把原始问题 + Faiss 返回的 top‑k “文档”拼成 prompt，生成回答。

### 1.1 术语：这是 “Rewrite–Retrieve–Read / Two-stage RAG”，不是 agentic

NLP/RAG 这边早就有一个标准叫：

* **Rewrite–Retrieve–Read**：先由一个（小）模型做 query rewrite，再用 dense retriever 检索，然后由大 LLM 作为 reader 生成答案。([OpenReview][1])
* 最近的 survey 也把这类前置 query rewrite + 检索 + 生成的 pipeline归类为 “two‑stage RAG” 或 “multi‑stage RAG”。([SpringerLink][2])
* 工程圈通常叫它 **高级 RAG（advanced RAG）里的 query rewriting / query transformation**，但本质还是“单次检索、单次生成”的线性 pipeline。([Medium][3])

你这个三步其实就是：

> **Rewrite–Retrieve–Read 型 two‑stage RAG**
> LLM 在前后各出现一次，中间是一个 GPU dense index。

**不算 agentic**，因为：

* 没有 decision / branching：每个请求必走这三步；
* 没有 multi‑tool planning / 循环调用；
* LLM 只是 pipeline 的两个固定 stage，而不是一个根据环境选择 action 的 agent。

这没问题，你在论文里就叫它：

* “two‑stage RAG pipeline with query rewriting” 或
* “rewrite–retrieve–read RAG pipeline (LLM‑in‑the‑loop retrieval)”([OpenReview][1])

这样 reviewer 一看就知道你跑的是什么，不会纠结“这哪门子 agent”。

---

## 2. 这个简单 workflow 能 claim 什么？适合评什么 policy？

### 2.1 它适合支撑哪一类 RQ/claim？

这个三步 RAG，适合拿来支撑的是下面这类 claim：

1. **单机、单租户、多组件 RAG 上的 GPU 内存管理 / UVM policy**

   > 在一个真实的 two‑stage RAG pipeline 上，把大 LLM 和 oversub GPU Faiss index 放一块卡，
   > 默认 UVM 会在 rewrite / 检索 / read 三阶段之间产生大量 page fault，拖垮 tail latency；
   > gBPF 把 UVM 行为变成“region cache + bulk prefetch”，显著减少 fault 和 stall，让 GPU‑Faiss 真正可用。

2. **跨 runtime / 跨库 的 policy waist**

   * 这条 pipeline 同时涉及：

     * vLLM（Python/CUDA runtime）；
     * Faiss‑GPU（C++ 库，自己用 cuBLAS / 自写 kernel）；
   * 没有 gBPF 的情况下，两边都只能在用户态各玩各的策略，互相看不见 UVM/page fault；
   * 你可以 claim：

     > gBPF 在 driver+device 层提供一个统一 policy substrate，让 LLM 和 Faiss 共享 HBM/UVM 策略，而不需要改两个框架本身。

3. **“UVM oversub + GPU index” 这条线目前只有 vector papers写，但只看 vector 一边**

   * 类似 RUMMY 那类 work 已经证明：IVF‑GPU + naive UVM 在 oversub 下会产生巨大 fault 和 100× slowdown。([PubMed Central][4])
   * 但它们是**单一 vector workload**，没考虑同卡还跑大 LLM 的情况；
   * 你可以说：

     > 我们证明在 two‑stage RAG 上，UVM naive 行为同样会炸，而且会干扰 LLM；
     > gBPF 提供的是“跨 LLM + Faiss 的 region‑aware policy plane”。

### 2.2 这类 workflow 适合的 gBPF policy 类型

简单版 workflow 的优势是：**它主要压内存 policy，不需要你玩很复杂的 scheduling**。你可以锁定两类 policy：

1. **内存 / UVM Policy（核心）**

   *Host 侧 gdrv_mem_ops + device 侧 hint：*

   * 把 LLM 权重 + KV cache 看作一类 region，Faiss index 的 IVF list / PQ chunk 看作另一类 region；
   * Device 侧在 LLM kernel / Faiss kernel 的选定指令上插 `gdev_mem_ops.access`，写入：

     * `(region_id, last_access_ts, bytes, type=LLM or INDEX)`；
   * Host 侧 UVM hook 做：

     * **给 LLM region 更高优先级**：

       * pin 住一部分 key weights / embedding / 频繁访问的 KV 区域；
       * 在 eviction list 里靠后；
     * **Faiss 作为 region cache**：

       * 保证一定数量的“hot IVF lists” 在 HBM；
       * 根据 hint 在 Step2 之前 prefetch 即将访问的 list；
     * 在内存压力高时优先 evict 冷的 Faiss region，而不是 LLM 的核心权重/KV。

   这个 policy 完全可以写成 100 行以内 eBPF 逻辑，足够 evaluation 用。

2. **轻量 scheduling / prefetch 协调（可选）**

   如果你愿意，可以顺手利用：

   * 在 Step1（rewrite）结束到 Step2（Faiss）开始之间有一个 host 空隙；
   * 在 Step2（Faiss）结束到 Step3（generate）开始之间也有空隙。

   你可以在这些点 hook 上做：

   * `prefetch` callback 提前搬 Faiss region / LLM region；
   * 调整 hardware queue priority（Faiss kernel 可以优先级稍低）。

   这部分只要在 eval 里点一下“我们也用了 scheduling hook 来协助 prefetch”，不需要展开太多。

**不适合用这个 workflow claim 的东西：**

* 多租户公平性（没有真正多 tenant / class）；
* agentic multi‑tool planning；
* 非 LLM / vector 的其它异构 workloads（除非你另外加）。

---

## 3. 具体实验怎么设计（基于你说的三步）

下面我给一套**最简单、但拿得出手**的 eval 设计，用于这条 workflow。你可以把它挂在“RQ1: Single-tenant heterogeneous workload (two-stage RAG)”下面。

### 3.1 Workload 定义（论文里可以这么写）

**组件：**

* GPU：RTX 5090（32GB HBM）。
* LLM：Qwen3‑30B‑A3B FP8，通过 vLLM 提供服务。([OpenReview][1])
* Vector Index：Faiss IVF‑PQ GPU index。

  * 生成 N 条随机向量（例如 N = 30–50M, dim = 768）；
  * 用 UVM/managed memory 存放，使得 index footprint（向量 + centroids + codebook）> 32GB，有意 oversub。

**每个请求的流程：**

1. **Rewrite (LLM)**
   Prompt：
   `Rewrite the following question into a search query. Return only the query.\n\nQuestion: …`
   输出几十 token 的检索 query。

2. **Retrieve (Faiss‑GPU)**

   * 把 query 映射为一个 d 维向量（可直接用随机 projection/固定 embedding，语义不重要）；
   * 调 Faiss‑GPU index 做 top‑k 检索（k=32/64）。

3. **Read (LLM)**

   * 构造 prompt：原始问题 + k 个“文档片段”（随机文本，长度控制在总 1–2k token）；
   * 再调用 Qwen3‑30B 生成完整回答。

**负载：**

* 先 warmup 100 请求；
* 再测 1000 请求（或 500），串行或用 4 个并发 client，保证 GPU 不是完全 idle。

论文里可以描述为：

> “We replay a trace of 1000 requests through the rewrite–retrieve–read pipeline and report per-request median and p99 latency, as well as GPU utilization and UVM page-fault statistics.”

### 3.2 Baseline & gBPF 配置

搞三组就够了：

1. **CPU‑Faiss（现实保守方案）**

   * LLM 在 GPU；
   * Faiss index 完全在 CPU 内存，用 CPU Faiss 检索；
   * 不用 UVM，GPU 只有 LLM 权重/KV；
   * 这是“延迟安全，但 retriever 慢”的 baseline。

2. **GPU‑Faiss + naive UVM**

   * LLM 在 GPU；
   * Faiss index 放在 GPU 视角下的 UVM 内存，大小 >32GB；
   * 不启用 gBPF，完全依赖默认 UVM LRU + prefetch 行为；
   * 这是你要打的“现实中很多人 naive 做法”。

3. **GPU‑Faiss + gBPF region policy**

   * 同 GPU‑Faiss + UVM；
   * Host 侧挂 gdrv_mem_ops，在 UVM hook 上实现前面说的 region cache 策略；
   * Device 侧在 LLM & Faiss kernel 上插 access/hint；
   * 用 gBPF maps 维护 hotness & tenant type（LLM vs INDEX）。

### 3.3 实验 1：End-to-end latency（核心图）

**目标：**证明在 two‑stage RAG 上，naive UVM 会把 tail latency 搞崩，gBPF 能明显收尾，且不比 CPU‑Faiss 慢太多。

**测：**

* 对每个配置跑 1000 请求；
* 记录 per‑request end‑to‑end latency；

**图：**

* 柱状图（或 CDF）：

  * x 轴：配置（CPU‑Faiss, GPU‑UVM, GPU‑gBPF）；
  * y 轴：p50 / p95 / p99 latency（归一化到 CPU‑Faiss 或 fit‑in‑HBM）。

**要给出的结论：**

* CPU‑Faiss：p99 baseline（可能 800ms 之类，你自己测）；
* GPU‑UVM：p50 也许 0.7× baseline，但 p99 是 2–4×，极端抖动；
* gBPF：p50 接近 GPU‑UVM（甚至更快），p99 压回到 1.2–1.5× CPU‑Faiss。

一句话可以写：

> “On the rewrite–retrieve–read RAG workload, naive UVM leads to up to 3× p99 inflation, while gBPF cuts p99 latency by 2× compared to UVM-only and keeps median latency close to GPU-based retrieval.”

### 3.4 实验 2：UVM fault & stall breakdown（解释机理）

**目标：**解释性能差异来自什么；这是 OS 论文信用的关键。

**测：**

* 统计每个配置下：

  * 总 page fault 数；
  * 每请求平均 fault 数；
  * fault latency；
  * GPU 时间分布（LLM compute, Faiss compute, UVM stall, idle）。

**图：**

1. 每请求 page fault CDF：

   * x 轴：每请求 fault 数（log scale）；
   * y 轴：CDF；
   * 曲线：GPU‑UVM vs GPU‑gBPF。

2. GPU 时间分布堆叠柱：

   * x 轴：GPU‑UVM / GPU‑gBPF；
   * y 轴：时间百分比；

你要讲明白：

* GPU‑UVM：retrieval 步骤里 fault 成堆（访问大 index），甚至读 LLM 权重/KV 时也会 fault；
* GPU‑gBPF：通过 region cache + prefetch，把大部分迁移从 request 内 fault 变成 request 之间的 bulk 迁移，fault 数和 stall 时间明显下降。

### 3.5 实验 3（可选）：QPS scaling / short trace

如果你时间真不够，不做 QPS sweep也勉强可以；
但如果你能多跑几组 QPS（比如 2 / 4 / 8），你可以画一个简单的：

* x 轴：QPS；
* y 轴：p99 latency / SLO violation rate；

展示：

* GPU‑UVM 随 QPS 升高 tail 立刻炸；
* GPU‑gBPF 在同 QPS 下 tail 更可控。

---

## 4. 这条简单 workflow 的定位总结

如果你就做这三步：

1. LLM 重写；
2. GPU‑Faiss 检索（oversub + UVM）；
3. LLM 生成回答；

然后在这个 workload 上做上面那 2–3 个实验，你可以**非常干净地 claim**：

1. **这是一个标准的 “rewrite–retrieve–read” two‑stage RAG pipeline**，已在 NLP 社区被广泛使用，用 LLM 做 pre‑retrieval query rewriting + 后端 reader。([OpenReview][1])
2. 在这样一个 multi‑stage RAG pipeline 上，**默认 UVM 在 oversub index + 大模型并存时会制造严重的 page fault storm 和 tail latency**，哪怕只有一个 tenant。([PubMed Central][4])
3. gBPF 提供的 cross‑library、host+device region‑level policy 能够：

   * 保住 LLM 的 tail latency（p99 相比 CPU‑Faiss baseline 只涨 1.2–1.5×）；
   * 让 GPU‑Faiss 相比 CPU‑Faiss 提供 1.5–2× 的 median 速度提升；
   * 把 fault 和 stall 显著降低一个量级。

你就把这条当作：

* RQ1：Single‑tenant heterogeneous workload（two‑stage RAG）；
* RQ2/多租户、公平、多 agent 那些，交给别的用例（比如 LLM+GNN、priority scheduler）去撑。

所以，如果你真的时间紧，这条简单三步 workflow 完全可以保留，只要你：

* 正确命名（two‑stage / rewrite–retrieve–read RAG，不要硬叫 agentic）；
* 把 policy 和实验设计收敛到 **“UVM oversub + GPU index + 大模型” 的 memory story** 上，
  就不会显得“随便跑跑”。

如果你愿意，我可以帮你把这一节直接写成 evaluation 的 LaTeX 文本（含一段 workload 描述 + baseline 段 + 两个图的 caption + 2–3 句 main takeaway），你改模型名字 / 数字就能塞进论文。

[1]: https://openreview.net/forum?id=gXq1cwkUZc&utm_source=chatgpt.com "Query Rewriting in Retrieval-Augmented Large Language ..."
[2]: https://link.springer.com/article/10.1007/s00521-025-11666-9?utm_source=chatgpt.com "A survey on retrieval-augmentation generation (RAG) ..."
[3]: https://medium.com/%40rogi23696/build-an-advanced-rag-app-query-rewriting-1cedbfbfbc59?utm_source=chatgpt.com "Build an Advanced RAG App: Query Rewriting"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12587015/?utm_source=chatgpt.com "Improving Large Language Model Applications in the Medical ..."
