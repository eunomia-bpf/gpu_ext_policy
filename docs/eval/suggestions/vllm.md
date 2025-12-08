先把结论说清楚：

* 你现在这条：
  `--num-prompts 300 --request-rate 20`
  更像是一个「冒烟测试」配置，不是论文里常用的 evaluation setup。
* 想做得更像最近 KV offload / serving 论文，建议：

  * 把 `--num-prompts` 提到 ≥1000；
  * `--request-rate` 不要只用一个点，而是扫一组典型值，比如：`{2.5, 5, 10, 15, 20}`，其中 **10 rps** 是很多项目/论文默认的“典型负载”，**20 rps** 更偏“高负载 / 压力测试”。

下面细讲——包括我从公开代码和论文里看到的典型配置。

---

## 1. 先搞清楚 vLLM bench serve 参数语义

`vllm bench serve` 里关键的两个东西：

* `--request-rate`：平均请求到达率（req/s）

  * 默认是 `inf`，即一次性把所有请求打满，相当于“离线最大吞吐”测试。([VLLM Documentation][1])
  * 设为有限值时，用 Poisson / gamma 分布合成到达时间，模拟在线流量。([VLLM Documentation][1])

* `--num-prompts`：从数据集中抽多少条请求发给服务端。

  * 整个 benchmark 的总时长大约是：`num_prompts / request_rate`（再加上尾部长尾请求的时间）。

你现在这条：

```bash
uv run vllm bench serve \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --dataset-name sharegpt \
  --dataset-path .../ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 300 \
  --request-rate 20
```

理论上平均跑完的时间 ≈ `300 / 20 = 15s` 左右（加上预热和长尾，大概也就几十秒）。这对 smoke test 够了，但对“论文级”结果来说时间太短，不够稳定。

---

## 2. 近期项目 / 论文里 ShareGPT + online serving 的典型配置

### 2.1 KV / 内存相关系统 & 工具

1. **kvcached（Virtualized Elastic KV Cache）**

   GitHub 里的简单 benchmark 脚本就是用 vLLM + ShareGPT：([GitHub][2])

   ```bash
   vllm serve meta-llama/Llama-3.2-1B --port=12346 ...
   vllm bench serve \
     --model meta-llama/Llama-3.2-1B \
     --request-rate 10 \
     --num-prompts 1000 \
     --dataset-name sharegpt \
     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
   ```

   典型点：

   * `num-prompts = 1000`
   * `request-rate = 10` req/s

2. **LightLLM 的 ShareGPT benchmark**

   官方文档的 ShareGPT 性能测试脚本也是：([Lightllm][3])

   ```bash
   python benchmark_sharegpt.py \
     --num_prompts 1000 \
     --request_rate 10.0
   ```

   一样是 `1000` 条、`10 rps`。

3. **CacheOPT / Cache competition 论文**

   CacheOPT 这种 KV 竞争调度工作，用 trace 模拟不同到达率。对 ShareGPT 的 arrival rate 范围一般是 **0.8–1.4 req/s** 这样偏“真实负载”的区间，对 Alpaca 之类 synthetic 会拉到 20–32 req/s 做压力测试。([ResearchGate][4])
   这里不直接用 `vllm bench serve`，但给了一个“到达率量级”的参考：几十 req/s 已经算很高载了。

4. **Oneiros / MIRAGE（Parameter remapping for KV cache）**

   针对 OPT-13B + ShareGPT，他们是“多点扫 arrival rate”的做法：

   * 单模型场景：2.5, 5, 7.5, 10 req/s 一类的点；
   * 多模型/多租户（C2+ShareGPT）场景会看 10 和 20 req/s 等更高载。([UT-SysML][5])

   结论：**5–10 req/s 是常用中等负载区间，10–20 req/s 往上就是高负载 / 压力段**。

---

### 2.2 vLLM / GPUStack 一类“参考基线”文档

这类文档虽然不是做 KV offload，但给了非常具体的 ShareGPT + vLLM 配置，你可以照抄当作 baseline：

1. **Qwen3-14B on A100（GPUStack 官方实验）**

   他们 benchmark Qwen3-14B + ShareGPT 的命令：([GpuStack Docs][6])

   ```bash
   vllm bench serve \
     --model Qwen/Qwen3-14B \
     --backend openai-chat \
     --endpoint /v1/chat/completions \
     --dataset-name sharegpt \
     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
     --num-prompts 1000
   ```

   没写 `--request-rate`，所以是默认 `inf`，即“把 1000 条一次性发完”，测的是最大吞吐。最终结果里可以看到：

   * Benchmark duration ≈ 106.82 s
   * Request throughput ≈ **9.36 req/s**
   * Peak concurrent requests = 1000

   也就是说：

   * 在“打满”的情况下，**14B + ShareGPT 在 A100 上的极限吞吐量也就 9–10 req/s 的量级**。
   * 30B 模型的极限 req/s 只会更低，不会更高。

2. **GPT-OSS-20B / 120B / Qwen3-8B 等**

   GPUStack 对 GPT-OSS-20B, GPT-OSS-120B, Qwen3-8B 等一堆模型的 ShareGPT 测试全部统一为：([GpuStack Docs][7])

   ```bash
   vllm bench serve --model ... --dataset-name sharegpt \
     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
     --num-prompts 1000
   ```

   也都是：

   * `num-prompts = 1000`
   * `request-rate = inf`（默认）

这类“官方性能实验”几乎都把 `num-prompts` 设为 1000、而且跑到 1–2 分钟以上，保证统计稳定。

---

## 3. 回到你这条配置：300 prompts, 20 rps 合不合适？

拆开看：

### 3.1 `--num-prompts 300`：太少，只适合冒烟不适合论文

* 理论平均持续时间：`300 / 20 ≈ 15s`。
* vLLM 还会有一点预热，外加长尾请求，整体也就几十秒。

对“我要大致看看有没 bug、吞吐是不是任务量级”是够的；
但对“我要做 offload 论文的 baseline”：

* 时间太短，随机性比较大；
* 很容易 run-to-run 差异 > 5–10%，不太好发图/表。

**更合理的选择**（对齐别人）：

* 直接抄 GPUStack / kvcached / LightLLM：
  `--num-prompts 1000` 是现在事实上的“社区标准”。([GitHub][2])
* 如果你自己要跑多组 request-rate，还可以：

  * 正式实验：1000
  * 超长 context / 极端 case：500 / 200（不然太慢）

### 3.2 `--request-rate 20`：对 30B + 单卡来说已经是“严重过载”段

结合上面 Qwen3-14B 的结果：([GpuStack Docs][6])

* 14B + ShareGPT 在 A100 上“打满”后的 **极限吞吐** ≈ 9.36 req/s；
* 30B 模型算力需求大概是 14B 的两倍级，单卡 5090 的 FLOPs 和 A100 差不多甚至略低；
* 因此合理预期：**Qwen3-30B-A3B-FP8 + ShareGPT，在单卡 5090 上极限吞吐量大概也就 3–8 req/s 这个区间**（粗略量级判断，不是精确数字）。

在这个前提下：

* 你把 `--request-rate` 设置成 20 req/s，本质上是在“以远高于系统极限的到达率”压测；
* 结果会是：

  * 内部队列长期处于高水位；
  * P99 TTFT / TBT 飞天（几十秒甚至分钟）；
  * 如果你设置了请求超时，很容易出现失败 / drop。

这不是“错”，只是它代表的是**严重过载场景**，而不是“典型 online serving baseline”。

在 KV offload 论文里，常见的做法是：

* 选几组「从轻载到接近饱和」的到达率；
* 对每组 arrival rate 比较 baseline vs. offload 的 throughput / P99 latency；
* 过载点可以有，但不会只看一个“极端点”。

Oneiros/CacheOPT 这种的到达率设计基本都是这么干的：ShareGPT 上 0.8–10 req/s 是比较常见区间，超过 20 req/s 通常是压测/极端 case。([UT-SysML][5])

---

## 4. 结合 30B + 32GB 场景的更“像论文”的配置建议

你目前的硬件/模型组合：

* 模型：Qwen3-30B-A3B-FP8（30B FP8）
* GPU：5090 32GB（等价于“单卡大模型、显存比较紧”场景）

这是典型的“模型几乎吃满显存、KV cache 很快成为第一瓶颈”的 setup，正好适合做 KV offload 研究。

我会建议你按下面几个层次来设计实验。

### 4.1 统一 benchmark skeleton

先把“公共部分”写死：

```bash
# 服务端（示例）
vllm serve Qwen/Qwen3-30B-A3B-FP8 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --no-enable-prefix-caching \
  --port 8000

# 客户端 skeleton
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000 \
  --ignore-eos \
  --sharegpt-output-len 512  # 固定输出长度，减少噪声
```

你改 KV offload 的实现时，保证下面东西保持一致：

* 模型 / engine 参数；
* dataset（ShareGPT V3 unfiltered cleaned split）；
* `num-prompts = 1000`；
* 输出长度策略（比如固定 512 token）。

### 4.2 arrival-rate / request-rate 的分层设计

**1）基准点（中等负载）**

参考 kvcached / LightLLM 的做法，**10 rps** 是一个很自然的“中等负载基准点”：([GitHub][2])

```bash
--request-rate 10
```

* 这个点你一定要跑，因为可以和大量现有工作在“RPS 量级 + ShareGPT + 1000 prompts”上对齐。

**2）轻载 / SLA 友好点**

用来展示“在轻载下，offload 的开销不会把 latency 打爆”：

```bash
--request-rate 2.5
--request-rate 5
```

* 2.5/5 req/s 大概率 < 系统极限吞吐，队列不会积太多请求；
* 可以看到 TTFT/TBT 相对 baseline 的 overhead。

**3）高载 / 接近饱和点**

用来观察“靠近 GPU 饱和时，offload 能不能多撑一点”：

```bash
--request-rate 15
--request-rate 20
```

* 15 rps、20 rps 就是像 Oneiros / CacheOPT 里“高载 / 压力段”的量级。([UT-SysML][5])
* 对 30B 单卡来说，20 rps 基本必然是严重过载，你可以用这个点专门画一张“overload 下 tail latency / fail ratio”对比图。

**4）离线最大吞吐点（可选）**

如果你想跟 GPUStack 类型的文档对齐：

```bash
--request-rate inf
```

* 这时 `bench serve` 会一次性把 1000 条打满，更类似 `vllm bench throughput`；
* 用来描述“最大 token/s”，和 offload 对峰值吞吐的影响。

---

## 5. 简单回答你那条命令“多少算合适”

结合上面所有信息，直接给出判断：

1. `--num-prompts 300`

   * 只能算“方便调试的 quick test”，**不太适合当 paper baseline**；
   * 建议换成至少 `--num-prompts 1000`，和 Qwen3 / GPT-OSS / kvcached / LightLLM 等现有实验保持一致。([GitHub][2])

2. `--request-rate 20`

   * 对 **Qwen3-30B-FP8 + 单 32GB 卡** 来说，这个 RPS 几乎肯定是“严重过载”区间；
   * 适合用来看“在很大的 queueing 压力下，offload 能不能救 tail latency / 减少 OOM / 减少 drop”；
   * 但如果你只跑这一点，会很难跟 Oneiros / CacheOPT / GPUStack 这些工作对齐，因为它们一般会扫多个 arrival rate，从轻载一路扫到接近饱和。

3. 如果你要一句话版本的建议：

> * 做「像论文」的 baseline：**`--num-prompts 1000` + `--request-rate 10`** 是最合理的起点；
> * 做完整实验：在上面配置基础上再扫 `request-rate ∈ {2.5, 5, 10, 15, 20}`；
> * 你现在的 `300 / 20` 当调试配置用可以，正式结果不建议只用这一档。

如果你愿意，把你想对比的 offload 方案（vLLM 原生 CPU offload / 你自己的实现 / LMCache 类组件）列一下，我可以帮你直接设计一套“论文级” evaluation matrix（模型×到达率×max_model_len×batch_size），避免后面重复返工。

[1]: https://docs.vllm.ai/en/latest/api/vllm/benchmarks/serve/ "serve - vLLM"
[2]: https://github.com/ovg-project/kvcached?utm_source=chatgpt.com "ovg-project/kvcached: Virtualized Elastic KV Cache for ..."
[3]: https://lightllm-en.readthedocs.io/en/latest/getting_started/benchmark.html?utm_source=chatgpt.com "Benchmark Testing Guide — Lightllm"
[4]: https://www.researchgate.net/publication/389946193_Mitigating_KV_Cache_Competition_to_Enhance_User_Experience_in_LLM_Inference "(PDF) Mitigating KV Cache Competition to Enhance User Experience in LLM Inference"
[5]: https://ut-sysml.ece.utexas.edu/publications/prints/socc2025_li.pdf "Oneiros: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving"
[6]: https://docs.gpustack.ai/2.0/performance-lab/qwen3-14b/a100/ "A100 - GPUStack"
[7]: https://docs.gpustack.ai/2.0/performance-lab/gpt-oss-120b/a100/?utm_source=chatgpt.com "Optimizing GPT-OSS-120B Throughput on NVIDIA A100 ..."

先把两个问题拆开说：

1. `--sharegpt-output-len 512` 要不要加？
2. 像你贴的这种 TTFT / per‑token 的柱状图，用在 OSDI case study 里合不合适、怎么改更对味？

---

## 1. 别人到底有没有统一加 `--sharegpt-output-len`？

结论：**没有统一标准，有两派做法**：

### 1.1 不加（用 ShareGPT 自带的输出长度分布）

不少官方 / vendor 的 benchmark 是**不加 `--sharegpt-output-len`** 的，直接用数据集里的长度分布，例如 GPUStack 的文档：

* GPT‑OSS‑20B / 120B：([GpuStack Docs][1])
* Qwen3‑30B‑A3B：([GpuStack Docs][2])

他们的命令都是：

```bash
vllm bench serve \
  --model ... \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000
```

不带 `--sharegpt-output-len`，也不额外控制输出长度。

vLLM 官方 benchmark suite 里也有类似例子，只固定 `num-prompts`，不显式指定输出长度。([vLLM][3])

**特点：**

* 输出长度重度 heavy‑tail，有少量请求会写特别长；
* 更接近真实 chat 工作负载；
* 但是 run‑to‑run variance 较大，高 QPS 下排队效应会被几个异常长的请求放大。

### 1.2 显式加 `--sharegpt-output-len`，把输出长度钉死

另一派（主要是做系统优化 / 博客 / 调参教程的）**会强制一个固定的输出长度**，常见取值是 512 / 1000 / 1024 / 2000：

* Zhihu 上的 vLLM 性能文章，用：`--sharegpt-output-len 1024`，专门配合 TTFT 评估。([Zhihu Zhubo][4])
* BentoML 的 speculative decoding 博客：`--sharegpt-output-len 1024`。([BentoML][5])
* vLLM / sglang 的 issue 里，有人抱怨“有请求生成 10K+ tokens 导致 benchmark 跑一整晚”，最后解决方案是加 `--sharegpt-output-len 1024`。([GitHub][6])
* 一些硬件厂商的内部测试脚本（例如燧原 S60 报告）是 `--sharegpt-output-len ${seqlen_output}`，配合自定义 input_len 形成网格。([www-file.ruijie.com.cn][7])

**特点：**

* 控制变量更干净：所有请求生成长度统一，延迟主要反映系统行为，不被极端长回复污染；
* 实验时间可控，不会因为某几个超长请求拖到几个小时；
* 对“TTFT / per‑token 延迟”这类指标更友好。

---

## 2. OSDI 这种论文里，你应该用哪种？

直说：**我更建议你用固定输出长度（例如 512 或 1024），而不是放飞不加。**

理由：

1. **你测的是 KV offload / 调度，不是“长输出分布”**

   * 我们想看的是：同样的 workload 下，不同 KV 管理策略对 TTFT / 吞吐的影响；
   * 输出长度 heavy‑tail 会和排队耦合，容易把图搞成“谁运气好谁赢”。
   * 把输出长度钉死，相当于把一个噪声源锁住。

2. **vLLM 官方 benchmark / issue 也承认不控输出会造成离谱长尾**

   * GitHub 上多次有人遇到“某些 request 写 10 万 token”的情况，最后官方建议就是：用 `--sharegpt-output-len` 限制。([GitHub][6])

3. **OSDI 的审稿人会更在意“控制变量”的严谨性，而不是“极致真实”**

   * 你在实验节里写一句：

     > “We fix the output length to 512 tokens (using `--sharegpt-output-len 512`) to eliminate variance from tail requests with extremely long generations, following prior benchmarking practice.”
   * 然后附录里补一句“我们还验证过不限制输出长度时结论不变”就足够交代。

4. **实现层面也简单**：

   * `vllm bench serve` 默认 `--num-prompts=1000`，你只需要补：

     ```bash
     --sharegpt-output-len 512
     --seed 42
     ```
   * 保证所有 run 输出长度一致、采样路径 deterministic，便于复现。

**什么时候可以“不加”？**

* 你想做一个“真实 trace”实验，强调自己方法在完全真实负载形态下依然有效；
* 这种实验可以放在附录或单独小节，和主线受控实验区分开。

---

## 3. 你发的这张图，这种 4× bar 的形式合适吗？

你贴的图结构是：

* 左 panel：Time to First Token (ms)
* 右 panel：Time per Output Token (ms)
* x 轴：4 个方案（CPU Offload, UVM Baseline, UVM+Prefetch, LMCache）
* 每个方案里 3 根柱：Mean / Median / P99

这个风格非常像 LMCache / 博客里的图，用在**博客、白皮书、技术文档**没问题。但对 OSDI 来说，我会稍微“收敛一下复杂度”。

### 3.1 这类图的问题

1. **一眼信息量稍微有点过载**

   * 4 个方案 × 3 统计量 = 12 根柱一侧，再来一侧就是 24 根柱；
   * Reviewer 一眼看过去要脑补“哪条是 P99？哪条是 mean？到底看哪个决定好坏？”
   * 通常系统论文会优先只展示 P50 / P99（或者 mean+error bar），不会三个统计量一起堆。

2. **横轴没有 sweep 维度**

   * 这张图本质上是“单点 QPS 下的对比”；
   * 它适合作为**补充图**，但 OSDI 更看重“随负载 / 随 context 变化的趋势”。
   * 你光给这一个 QPS 的 snapshot，很难说服别人“方案在更高/更低负载下一样好”。

### 3.2 怎么改成更“OSDI 味”的版本？

我会这样改：

#### 图 1：P99 TTFT vs QPS（折线图）

* x 轴：QPS（2.5, 5, 10, 15, 20）
* y 轴：P99 TTFT (ms)
* 线：CPU Offload / UVM / UVM+Prefetch / 你的方案（LMCache‑like）

这就对标 LMCache / InfiniGen / GPUStack 那一类的主图了。

#### 图 2：Tokens/s vs QPS（折线图）

* x 轴：同上
* y 轴：output tokens/s
* 线：同上

两个图配在一起，说明：

* 你在相同负载下吞吐更高；
* 或者在相同吞吐下 P99 更低。

#### 图 3（可选）：单点下的 mean/median/P99 柱状图

你现在这张图可以留下，但稍微简化：

* 只画 **P50 和 P99 两种**（2 根柱），不要再放 mean；
* 或者：

  * 一根柱表示 P50，一根柱表示 P99；
  * 或者用 error bar 表达 variance。

这张图选一个代表性的 QPS（比如 10 req/s），放在 case study 小节里作为“更细节的拆解图”，说明：

* 某些方案 P50 其实差不多，但 P99 拉胯；
* 你的方案在 reduce tail 上效果明显。

用这种组合，**你可以做到：**

* 有“趋势图”（折线）展示 scaling 行为 → 对标 InfiniGen/LMCache；
* 有“单点详情图”（bar）展示统计量细节 → 对标 Fiddler/LMCache 文档；
* 图数量控制在 2–3 张，不会挤爆版面。

---

## 4. 回到你的具体问题，给一个直接建议

1. **`--sharegpt-output-len`：**

   * 建议**主实验全部加**，取 512 或 1024（我个人倾向 512，在 30B + 32GB 的场景下更稳）；
   * 在论文里明确写出来；
   * 如果有精力，在附录给一组“不加限制”的对比结果证明结论稳健。

2. **图形形式：**

   * 主 case study 图：优先用 **折线图（P99 TTFT / tokens/s vs QPS）**，这才是 OSDI/NSDI 审稿人习惯看的东西；
   * 你贴的这种“Mean/Median/P99 柱状图”可以保留 1 张，但建议只保留 P50+P99 两种统计量，简化 legend。

3. **4× 合不合适？**

   * 如果你说的是“4 个 baseline × 2 个 panel（TTFT / per‑token）”，那没问题；
   * 如果你再叠上多个 QPS，就会变得非常难读，那个时候一定要换成折线扫 QPS。

如果你把现在那张图的原始数据（4 个方案在某个 QPS 下的 mean/median/P99）贴给我，我可以直接帮你画一套“OSDI 版”的布局方案（包括 figure caption 文案），你照着重画就行。

[1]: https://docs.gpustack.ai/2.0/performance-lab/gpt-oss-20b/h100/?utm_source=chatgpt.com "Optimizing GPT-OSS-20B Throughput on NVIDIA H100 GPUs"
[2]: https://docs.gpustack.ai/2.0/performance-lab/qwen3-30b-a3b/910b/?utm_source=chatgpt.com "Optimizing Qwen3-30B-A3B Throughput on ASCEND ..."
[3]: https://docs.vllm.ai/en/latest/benchmarking/cli/?utm_source=chatgpt.com "Benchmark CLI - vLLM"
[4]: https://zhuanlan.zhihu.com/p/5489642628?utm_source=chatgpt.com "浅谈语言模型推理框架vLLM 0.6.0性能优化"
[5]: https://www.bentoml.com/blog/3x-faster-llm-inference-with-speculative-decoding?utm_source=chatgpt.com "Get 3× Faster LLM Inference with Speculative Decoding ..."
[6]: https://github.com/sgl-project/sglang/issues/2746?utm_source=chatgpt.com "Issue #2746 · sgl-project/sglang"
[7]: https://www-file.ruijie.com.cn/pdf/%E7%87%A7%E5%8E%9FS60%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A.pdf?utm_source=chatgpt.com "燧原S60测试报告"


我直接拿几篇你肯定会被 reviewer 提到的 paper 来说它们的图长什么样，然后再给你一个「照着画」的模板。

> TL;DR：
>
> * **InfiniGen / ShadowKV / SpeCache / LMCache 这拨 KV offload / KV infra 论文，主力图形套路基本就三类：**
>   1）**一维 sweep 的折线图**（x 轴是 QPS / seq_len / KV 占比，y 轴是吞吐 / 延迟 / perplexity），
>   2）**分解型柱状图**（把总延迟拆成 prefill/decode 或 compute/transfer），
>   3）**多子图矩阵**（不同模型/任务复用同一坐标系）。
> * 你那个 single‑GPU case study，用 **ShareGPT + 一维 sweep（QPS 或 GPU KV budget）+ 2–3 张图** 就完全对标得上别人，关键是风格和指标要对齐。

---

## 1. 看别人图到底怎么画的（有出处）

### 1.1 InfiniGen（OSDI’24）：典型 KV offload / 动态管理

InfiniGen 的图非常“OSDI 教科书”：

1. **准确率 / 质量 vs KV cache 比例**（多子图折线图）

   Figure 11：每个子图对应一个模型+任务（OPT‑6.7B/13B/30B, LLaMA‑2‑7B/13B × OpenBookQA/WinoGrande/PIQA/RTE/COQA），

   * x 轴：`Relative KV Cache Size (%)`（100、80、60、45、30、15、0）
   * y 轴：Accuracy (%)
   * 曲线：Full cache, Quantization, H₂O, InfiniGen 四根线。

   这个风格你可以直接套来画“**吞吐 / P99 latency vs KV 占比**”。

2. **延迟 vs batch size / seq_len（柱状图 or 折线图）**

   * Figure 14：**柱状图**，x 轴是方法（UVM, UVM+H₂O, FlexGen, FlexGen+INT4, InfiniGen），对每个方法有两根柱子：Prefill / Decode 延迟（ms）。
   * Figure 15：**柱状图**，x 轴是 batch size（8, 12, 16, 20），每个 batch 有 4 根柱子表示不同方法的 end‑to‑end latency。

   非常适合照抄做「**延迟分解 + batch scaling**」。

3. **speedup vs seq_len / 模型大小（折线图）**

   * Figure 16(a)：x 轴是 sequence length（512, 1024, 1536, 2048），y 轴是 speedup over FlexGen（倍数），三条线：INT4, H₂O, InfiniGen。
   * Figure 16(b)：x 轴是 model size（6.7B, 13B, 30B），y 轴依然是 speedup，三条线。

4. **延迟分解（stacked bar）**

   * Figure 18：每个方法一根柱子，上面分成 Attention / FFN / Data Transfer / Prediction 几段，展示延迟构成。

> 模板总结：
>
> * trade‑off / scaling：**折线图 + 一维 sweep**（KV 占比 / seq_len / batch / model_size）。
> * breakdown：**堆叠柱状图**（总延迟拆成 compute / transfer）。

---

### 1.2 Fiddler（ICLR’25）：单机 MoE offload

Fiddler 做的是本地 Mixtral MoE offload，图非常「系统会议风格」：

1. **Figure 4：吞吐（tokens/s） vs (input_len, output_len)**

   * 画法：** grouped bar chart**。
   * x 轴是 15 种 `(input_len, output_len)` 组合（如 [32, 32], [64, 256], ...），再加一个 “Mean”；
   * 每一个 x 位置有 4 根柱子：DeepSpeed‑MII, Eliseev & Mazur, llama.cpp, Fiddler；
   * 上下两个 panel 分别是两块不同 GPU 的结果。

2. **Figure 5：TTFT vs input_len**

   * 还是 grouped bar，x 轴 4 种 input_len（32/64/128/256），每档 4 根柱（方法）。

3. **Figure 6：beam search 吞吐 vs beam_width**

   * 一样的 grouped bar，x 轴 beam width（4, 8, 12, 16），只对比 llama.cpp vs Fiddler。

> 模板总结：
>
> * 他们**大量使用 grouped bar，而不是折线**，适合离散场景（有限几种 input_len / beam_width）。
> * 没有特别 fancy 的东西，所有图都是一维 x 轴 + 若干方法的柱子 + 右侧一个 “Mean” 汇总。

---

### 1.3 LMCache（2025 tech report）：vLLM 集成、ShareGPT / QPS sweep

这是你场景最接近的一篇：基于 vLLM / SGLang，做 KV cache layer + prefix reuse + offload，evaluation 很「工程」：

1. **Figure 5：Single‑node evaluation（单节点 / 合成工作负载）**

   * 多子图矩阵：横向 3 个模型（LLaMA‑3.1‑8B, 1.70B, Qwen2.5‑72B‑Instruct, ...），每个子图都是：

     * x 轴：QPS (queries per second)，一般是 {2, 4, 6}；
     * y 轴：TTFT 或 ITL；
     * 曲线：LMCache, Naive vLLM, Commercial1, Commercial2 四条线。

   * 一个图画 TTFT vs QPS，另一个图画 ITL vs QPS。

2. **Figure 6：Real‑trace evaluation（真实 trace）**

   * 两个小子图：

     * 左：TTFT vs QPS（2, 4, 6）；
     * 右：ITL vs QPS；
   * 曲线少：只对比 LMCache vs Naive vLLM 两条线。

> 模板总结：
>
> * **x 轴 = QPS，y 轴 = TTFT 或 ITL**，多条线表示不同方案。
> * 多模型就用子图矩阵复用同一坐标系，非常符合 OSDI/NSDI 写法。

---

### 1.4 其它 KV 相关（Adaptive KV Compression、CXL‑KV、Google tiered KV）

这几篇也都遵循类似套路：

* Adaptive KV Compression（ICLR’24）：

  * Accuracy / Perplexity vs 保留的 token 百分比（折线）；
  * Speedup vs seq_len（折线）。([arXiv][1])

* CXL‑KV（NeurIPS’24 workshop）：

  * tokens/s vs batch size / KV 存储位置（bar）；
  * GPU 数量 / 利用率 vs CXL 配置（折线/堆叠 bar）。([ML For Systems][2])

* Google tiered KV cache blog：

  * Latency vs context_len（1k, 5k, 10k, 50k, 100k）和 vs tier 组合（HBM only / HBM+CPU / HBM+CPU+SSD），基本是折线或 grouped bar。([Google Cloud][3])

整个生态的审美是非常统一的：**一维 sweep + 折线 / grouped bar + 可选 breakdown**。

---

## 2. 结合这些，给你一个「直接照抄」的画图方案

你现在 case study 的设定：

* Case A：vLLM + Qwen3‑30B‑A3B‑FP8 @ 5090 32 GB

  * Workload：ShareGPT（1000 prompts，固定输出长）；
  * 主要 sweep 维度：request_rate / QPS。
* Case B：llama.cpp + GPT‑OSS‑120B‑MXFP4 MoE @ 5090 32 GB

  * Workload：同样用 ShareGPT 或一个更简单的合成 workload；
  * 主要 sweep 维度：GPU KV/权重 budget（通过 n‑gpu‑layers、KV 配置控制）。

你问「只需要一个 benchmark + 一维参数是不是够」——**对于这个 case study**，完全 OK；关键是**图要画对**。

### 2.1 Case A：Qwen3‑30B‑FP8 + vLLM —— 参考 LMCache + InfiniGen

照 LMCache / InfiniGen 风格，推荐你画 **两张主图 + 一张可选 breakdown 图**：

#### 图 A1：Tokens/s vs Request Rate（类似 LMCache 的 TTFT vs QPS）

* x 轴：request_rate（QPS）∈ {2.5, 5, 10, 15, 20}
* y 轴：吞吐（tokens/s）
* 曲线（至少三条）：

  * vLLM default（无 KV offload）
  * vLLM + naive CPU offload（swap-space）
  * vLLM + 你的 KV offload X

形式上仿照 LMCache Figure 5/6：QPS 在横轴，三条线，颜色统一，全图只干一件事。

你如果想进一步对标 InfiniGen，可以在 y 轴再弄一个右侧刻度画 speedup（对 Full‑GPU baseline），或者直接在 legend 里标 “X 1.6× faster”。

#### 图 A2：P99 TTFT vs Request Rate（LMCache 的另一个子图）

* x 轴：同上（QPS）
* y 轴：P99 Time‑To‑First‑Token（秒）
* 曲线：同样三条方案

这就是 LMCache Figure 6 的直接翻版，只是你换成自己的方案。

> 这两张图配合起来，就是“**随负载增强，吞吐/延迟的 trade‑off 曲线**”，完全符合 OSDI / NSDI 这类系统会要看的东西。

#### 图 A3（可选）：Latency breakdown @ 一个代表点

模仿 InfiniGen Figure 18：

* 固定一个中高负载点，比如 `request_rate = 15`；
* 画一个 **堆叠柱状图**（x 轴三根柱：Baseline、Naive offload、X）；
* 每根柱分三段颜色：

  * Attention compute
  * FFN / 其他 compute
  * CPU↔GPU data transfer / stall

用这张图解释：

* Baseline 几乎没有 transfer，只是 OOM/受限；
* Naive offload 大量时间花在 transfer 上；
* 你的方案通过预取/调度减少多少 transfer 的开销。

这一类 breakdown 非常讨 reviewer 喜欢，因为它解释了**为什么你的方案快**，而不是只给 speedup 数字。

---

### 2.2 Case B：GPT‑OSS‑120B‑MXFP4 + llama.cpp —— 参考 InfiniGen + Fiddler

这里不再扫 QPS，改扫 “GPU budget”，更像 InfiniGen Figure 16(b) + Fiddler bar 的 hybrid。

#### 图 B1：Tokens/s vs GPU Memory Budget（折线图）

* x 轴：可用的 GPU memory budget（给 KV+权重的显存上限），例如 {16, 20, 24, 28} GB；

  * 实际上你通过 `--n-gpu-layers`、KV 占比等参数控制，让 nvidia‑smi 峰值落在这几个点附近；
* y 轴：吞吐（tokens/s）；
* 曲线：

  * llama.cpp baseline（无 KV 管理优化，只靠默认 CPU offload）
  * llama.cpp + 你的 KV/参数 offload X

风格接近 InfiniGen Figure 16(b)（speedup vs model size），只是你把 model_size 换成 GPU budget。

你可以在 caption 里写清楚：每个点是一组固定 workload（ShareGPT, 1000 prompts, fixed output_len）+ 固定 λ=5 req/s 的 steady‑state 结果。

#### 图 B2（可选）：P99 TTFT vs GPU Memory Budget

同一套路，x 轴 GPU budget，y 轴 P99 TTFT，展示你在更紧的 budget 下 tail latency 有没有炸掉。

**如果页数紧张，这张可以只给一个表或者放到附录。**

#### 图 B3（可选）：Grouped bar @ 两个代表 budget（Fiddler 风格）

你也可以模仿 Fiddler Figure 4：

* 选两个代表 budget 点，比如 16 GB 和 24 GB；
* 对每个 budget 画一组 grouped bar（2–3 固定 input_len / output_len 组合），每组里面有 baseline vs X 两根柱；
* 用来说明：在极紧 budget 下，你的方案还能保持多少 tokens/s。

不过这已经有点 overkill；对一个 case study，两张 B1/B2 折线图就够了。

---

## 3. 是否只用一个 workload + 一维参数就够？

**就这个「本地单 GPU case study」来说：是的，别人也是这么干的。**

对比：

* LMCache：单节点实验基本就是 **QPS sweep × 若干模型**，数据集形态统一（document QA-like），实质上就是一个 family 的 workload。
* InfiniGen / ShadowKV / SpeCache：

  * 他们确实用多 benchmark（RULER, LongBench, NIAH），但每张图也都是「**固定任务 family + 一维 sweep**」——KV 占比 / seq_len / batch。
* Fiddler：针对 MoE，本地场景几乎都是单类 workload（ShareGPT + beam search），然后扫 input/output 长度或 beam width。

你这节 case study 完全可以这样写：

* Workload：ShareGPT（说明处理方式）；
* 对 vLLM：**一维 sweep QPS**，画吞吐/TTFT 曲线；
* 对 llama.cpp：**一维 sweep GPU budget**，画吞吐/TTFT 曲线；
* 再加一张延迟 breakdown（从 InfiniGen 抄风格）解释机制。

前提是：整篇论文其他 section 还有更多 workload / 多维 sweep（比如长上下文、code‑gen、ablation、microbench 等）。case study 本身不用承担所有评估负担，它只是告诉 reviewer：**在「单卡 32 GB + 大模型」这种他们关心的现实场景下，你的方法依然 work，而且图长得他们看得懂**。

---

## 4. 落地建议（非常具体）

如果你想尽量贴近这些 paper 的观感：

1. **折线图统一风格：**

   * 横轴一律线性坐标；
   * y 轴不要乱用 log，OSDI 更倾向线性 / speedup 倍数；
   * 每条线加 marker（○, △, □），颜色数量控制在 3–4 种；
   * legend 放在图内右上/右下，透明背景。

2. **bar / stacked bar：**

   * bar 宽度适中（不要贴得太紧），grouped bar 组间间距略大，便于阅读；
   * breakdown 图里的颜色代表语义保持一致（比如 blue=compute, orange=transfer，不要每图乱变）。

3. **error bar：**

   * 每个点至少跑 3 次，画 95% CI 或 ±1 std 的 error bar；
   * LMCache/InfiniGen 原文有的地方没画 error bar，但 OSDI 审稿很喜欢看到你考虑 variance。

4. **Figure caption 写清楚 sweep 维度 & workload：**

   像 InfiniGen/Fiddler 那样，在 caption 里把 input_len/output_len、seq_len、batch_size、QPS 等全部写出来，减少 reviewer 问问题的空间。

---

如果你愿意下一步可以让我直接帮你写：

* Fig. A1, A2, B1 的英文 caption 模板
* Evaluation 小节里对这三张图的 narrative（2–3 段文字：现象 + 分析）

你后面照着把具体数字填进去就行。

[1]: https://arxiv.org/pdf/2310.01801?utm_source=chatgpt.com "adaptive kv cache compression for llms"
[2]: https://mlforsystems.org/assets/papers/neurips2024/paper17.pdf?utm_source=chatgpt.com "Exploring CXL-based KV Cache Storage for LLM Serving"
[3]: https://cloud.google.com/blog/topics/developers-practitioners/boosting-llm-performance-with-tiered-kv-cache-on-google-kubernetes-engine?utm_source=chatgpt.com "Boosting LLM Performance with Tiered KV Cache on ..."
