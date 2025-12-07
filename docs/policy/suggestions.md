先上表，再补一点解释。

---

## 1. Microbenchmark 工作负载 ⇨ 适合的 prefetch / eviction 策略

### 1.1 Tier‑0 / Synthetic + Scientific + Graph + DL Kernels

| Workload               | 访存模式                                                              | Prefetch 策略（谁来 prefetch）                                                                                                                                 | Eviction / Placement 策略                                                                                                              | 典型用途                                                                                                  |
| ---------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **seq_stream**         | 单块大 region 顺序单遍扫描，几乎无重用                                           | **Aggressive 顺序 prefetch（host）**：根据 fault 地址顺序拉下一批 page（多页一批），或直接在 region_add 时一次性 bulk prefetch；可以在 gdrv_mem_ops.prefetch 里实现“从当前 fault region 向前看 N 页” | **FIFO / bypass**：把这类 region 标记为 “streaming / non‑cacheable”，在 region_add 时设置 flag；eviction 优先从 streaming list 头部驱逐，不维持复杂 hotness 统计 | 纯带宽测试 / worst‑case streaming，验证 “prefetch 不会把 cache 用烂”                                               |
| **rand_stream**        | 大 region，page 级完全随机访问，几乎无重用                                       | **Disable / conservative prefetch**：在 prefetch hook 中检测高 miss rate+低重用，直接禁止预取，避免白搬                                                                       | **Random / LRU + thrash detection**：如果检测到某 region 访问 pattern 是“每个 page 只用一次”，可以直接 mark 为 bypass，不进入 region cache 列表；否则 fall back LRU | 作为 “无时间局部性” 的负例，保证策略不会恶化性能                                                                            |
| **pointer_chase**      | 链式指针追踪，随机、小步长、不规则；单线程/单 warp，序列化严重                                | 一般不做通用 prefetch；如果你能在 device eBPF 里看懂 node 结构，可以**warp‑local lookahead prefetch**：对 `next` 指针多走一两步，发出 gdev_mem_prefetch_hint                             | eviction 用 **简单 LRU/FIFO** 即可；更重要的是避免过度 prefetch，把带宽留给其他 workload                                                                    | 测 UVM 本身 latency & 你的 handler 开销，不是策略收益主战场                                                            |
| **Hotspot / Jacobi2D** | 固定大小 2D 网格（约 100–200MB），每轮迭代扫一遍，stencil 局部性强，迭代次数多                | **Phase‑wise bulk prefetch（host）**：在每轮 kernel launch 前（region_add/prefetch hook）直接把网格对应 region prefetch 到 GPU；必要时 tile‑based prefetch：提前拉下一个 tile 行块     | **Pin‑for‑phase / MRU**：在整个迭代 phase 内将网格 region 视为 pinned（禁止 eviction 或放在 MRU 一端）；phase 结束时统一 demote                                 | 代表 CFD / PDE 等科学计算，有明显时间+空间局部性                                                                        |
| **Kmeans**             | 每轮：顺序扫所有 points，随机访问少量 centroids；多轮迭代                             | 对 data points：**顺序 prefetch（host）**，像 seq_stream；对 centroids：**常驻 / 小对象一次性 prefetch**                                                                    | **Points：streaming；centroids：pinned**。points 可以 FIFO/bypass；centroids 在 region_add 时标记为长期常驻，不参与普通 eviction                           | 典型 “顺序 + 热点小对象” 混合模式，检验你策略能不能区分 hot/cold                                                              |
| **GEMM（LLM 权重复用）**     | weights 巨大（几十 GB），按 layer 顺序访问；每个 token 重复扫所有 layer；activation 很小 | **Lookahead prefetch（host+device）**：host 在 prefetch hook 按 layer id 顺序预取 L+1/L+2；device 在每个 GEMM 调用前，用 gdev_mem_prefetch_hint 标记下一层或下一批 experts          | **Hot‑set pinning + age‑based eviction**：对频繁访问的 layer / expert 维持 LFU/LRU 计数，把 top‑K region pin 在 GPU；冷 layer 按 age/FIFO 驱逐          | 主要代表 LLM decode 阶段的权重复用。gBPF 里 llama.cpp 的 expert cache policy 就是 max‑prefetch + hot‑set eviction 的组合 |
| **Conv2D（多层）**         | 特征图较小，filter 权重按 layer 顺序扫描，多层卷积；权重可重用                            | 对 feature map：一次性 prefetch+pin；对 filters：**layer 顺序 prefetch**，类似 GEMM，但每层更小，可以一次预取多层                                                                    | **Filters 用 hot‑set / age eviction**；feature map pinned；如果 layer 数很多，可以给最近几层更高优先级，远期 layer 允许被淘汰                                     | 代表 CNN 推理；策略与 GEMM 很像，只是 region 粒度更细                                                                  |
| **BFS（单源）**            | CSR 图结构，frontier 扩展，访问边几乎随机，一次遍历就结束                               | Prefetch 很难有收益：可以在 host 上对 CSR 的 edgeList 做**小步 sequential prefetch**（按 chunk），但不能 aggressive                                                            | **简单 LRU/FIFO**；更重要的是把 BFS 标成低优先级 workload，避免它把 HBM 刷爆；如果检测到 working‑set≫HBM，可直接 bypass UVM 改走 host‑side compute                     | 用作几乎无重用的图负例；PageRank 才是有时间局部性的版本                                                                      |
| **PageRank**           | 固定图（~150MB），每轮遍历所有边 + 更新度量，多轮迭代，有时间局部性但空间访问随机                     | **Chunk‑based prefetch（host）**：按照 CSR edge blocks 顺序预取；可在每轮开始 bulk prefetch 图结构；device 侧根据 warp 访问统计，用 hint 提示哪些 block 很热                                | **Graph structure pinned / long‑lived；residual/label 数组 MRU**：尽量不在迭代中驱逐图；如果 oversub 很高，可以用“冷 block 优先 evict”（LFU on block）           | 模拟图分析 / GNN 中 adjacency 重用；检验策略能否在“空间随机+时间重用”下工作                                                      |

---

## 2. 真实应用工作负载 ⇨ 对应的 UVM Policy

结合你 paper 里的几个真实 workload（llama.cpp、vLLM、GNN、Faiss），以及 gBPF 的 region cache 接口（region_add / access / remove / prefetch + gdev_mem_prefetch_hint / pin_hint 等），可以给一个更“应用级”的表：

| 应用 Workload                             | 关键数据结构 / 访存模式                                            | Prefetch 策略                                                                                                                                          | Eviction / Placement 策略                                                                                                                                                                                            |
| --------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **LLM prefill（大 batch GEMM）**           | 单次大 tokens×weights GEMM，权重一次扫完；prefill 通常没太多重用           | 对权重做 **bulk prefetch on launch**（prefill 开始时直接迁到 GPU）；对 activations 用缺省 UVM                                                                          | Eviction 可以简单 FIFO；prefill 本身是 “one‑shot streaming” 阶段，不要过于复杂；关键是不要让 prefill 把 decode 的 hot 集合挤掉                                                                                                                   |
| **LLM decode（权重 + KV cache）**           | 每轮 decode：重复访问全部权重；KV cache 访问局部强（最近几 token 热）           | **双通道 prefetch**：① 对权重按层顺序预取；② 对 KV‑cache segment：按时间顺序预取下一轮会访问的 segment（可由 device 侧根据 head/tail offset 发 hint）                                      | 权重：hot‑set pinning（几乎全部常驻或按专家/层热度管理）；KV‑cache：**age‑based eviction**，最老 segments 优先 spill；必要时为 KV 和权重分开担保 HBM 配额（避免互相抢）                                                                                            |
| **MoE Experts（llama.cpp GPT‑OSS‑120B）** | 一大堆 expert 权重，gating 决定小子集；某些 experts 长期 hot             | Device 侧在 gating 完成后，用 **expert‑id 驱动 prefetch_hint**，对即将访问的 experts 做 bulk prefetch；host 侧 prefetch hook 收到 hint 后在 region_add/prefetch 里执行真实迁移     | Eviction 用 **per‑expert LFU/LRU**：维护每个 expert 的 access count / timestamp，把冷专家从 HBM 驱逐；hot 专家 pin。gBPF 里 max‑prefetch + expert‑level hotness 就是这个模式                                                                 |
| **vLLM KV‑cache（Qwen‑30B MoE）**         | 模型本体几乎填满 HBM，KV‑cache 增长溢出；访问以“最近几 token 的 KV segment”为主 | host 侧 **sequential prefetch by segment**：根据 PCIe/region_access 流，按 segment id 顺序预取接下来将扫描的一小段 KV；device 侧可以在 decode loop 中为将来的 step 发出 will‑use hint | Eviction：**KV‑segment age eviction**（近似 LRU，但避免 thrash），优先驱逐最老 request / 最老 segment；必要时对 MoE 权重和 KV 分出不同 eviction 列表，保证 KV 不被权重完全挤出。gBPF 实验里就是用 KV‑aware sequential prefetch + UVM eviction 替换 vLLM 自己的 offload 策略 |
| **GNN Training（PyTorch）**               | 邻接表 + feature block：每轮 epoch 多次访问局部子图，1.5× oversub       | **Block‑wise prefetch**：按 adjacency/feature block 顺序预取，尤其是在 minibatch/graph‑partition 边界处对下一块做 prefetch；可以在 kernel launch 前一次拉入当前批次所有 blocks         | Eviction：**FIFO within partition + hot partition pinning**。当前 epoch 里的活跃 partition 尽量常驻，结束后批量 demote；gBPF 的实验里就是 “block prefetch + FIFO eviction”，把 epoch 时间从 71s 拉回 26s                                           |
| **Faiss / Vector DB – IVF index**       | IVF/Flat 下：遍历部分 lists 扫描 posting，某些 centroids / lists 热  | Prefetch：根据 query stream 在 host 侧维护“下一个 query 会访问哪些 lists”，对这些 lists 做 **segment‑级 prefetch**；也可以用 PCIe 统计对热点 list 做提前拉取                             | Eviction：**segment‑hotness‑based eviction**。对 posting list segment 做 LFU/2Q；hot centroids 和其附近 lists pinned，冷 segment 在内存压力下先驱逐。gBPF 中 adaptive prefetch policy 就是观察 PCIe 负载+访问流量来调 prefetch 和 eviction            |
| **Faiss / HNSW‑like index**             | 图式搜索，几乎随机访问，部分节点/层热                                      | Prefetch 只能做**浅层邻域 prefetch**：当访问 hub 节点时多拉几条邻接；不要全局 aggressive prefetch                                                                             | Eviction：在节点级做 hot/cold；把经常作为 entry 的 hub 节点所在 region pin，其他按 LRU 驱逐                                                                                                                                               |
| **ResNet / CNN 推理**                     | 多层 Conv/FC，权重固定、按层顺序访问，batch 小                           | 类似 GEMM：在 host prefetch hook 中按 layer id 做顺序预取；或在模型加载时 bulk prefetch + pin                                                                           | Eviction：大部分 inference 模式下，**直接 pin 全模型** 最省心；如果 oversub，只能对 tail 部分 layer 做 hotness 管理                                                                                                                            |

---

## 3. 怎么用这张表

简单说，这个表就是帮你回答两个问题：

1. **这个 workload 适合“要不要 prefetch”，如果要，是哪一类：**

   * 完全不预取 / 禁用（rand_stream、BFS negative case）；
   * 顺序/块级 prefetch（seq_stream、Hotspot、GNN、Faiss IVF）；
   * 由 device hint 驱动的语义 prefetch（MoE expert、LLM decode、GEMM‑reuse）。

2. **这个 workload 的 eviction 应该是“谁来保护、谁先牺牲”：**

   * streaming 数据：FIFO 或 bypass（不占据宝贵 HBM 热集，如 seq_stream / points）；
   * 热集：pin 或 hot‑set eviction（权重、KV 热段、centroids/hubs、邻接表、图结构）；
   * 真正没有局部性的场景：尽量保持简单（LRU/FIFO），更多是 sanity check，保证你的 policy 不把它搞得更糟。

如果你愿意，我可以帮你把这些策略直接翻译成 gBPF 的伪代码 skeleton：对应到 `region_add/region_access/region_remove/prefetch` 四个 hook + `gdev_mem_prefetch_hint/gdev_mem_pin_hint` 两个 helper，各给一个最小可运行的 eBPF 例子。
