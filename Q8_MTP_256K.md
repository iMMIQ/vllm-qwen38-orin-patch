# Q8 KV、MTP、prefix cache 与 256K 上下文实验

本地实验分支，基于 vLLM 0.20.0 与原有 Orin W8A8 prefill 补丁。
测量摘要见 [q8-mtp-results.json](verification/q8-mtp-results.json)。仓库已发布版本的 FP16 KV 单并发成绩不能直接代表此配置。

新增行为：

- KV cache 使用原生 `int8_per_token_head`，每个 token、每个 head 使用独立 FP32 scale；不是 FP8。
- Prefill 从分页 INT8 KV 中提取有效 token，临时解量化为 FP16，使用 FlashAttention 2；decode 保留 INT8 分页读取。
- MTP 验证使用分段 attention，依 query 长度选择 32/64 行；单 token decode 采用 16 行、32 列。
- 分段工作区覆盖 `max_num_seqs × (1 + num_speculative_tokens)`；`FULL_DECODE_ONLY` CUDA Graph 覆盖目标验证，草稿路径沿用该模式下的 eager 执行。
- MTP 启动时检查所有 linear/norm 权重已经加载。
- Prefix cache 使用 `mamba-cache-mode=align`，保留 FP32 GDN state。
- 配置上下文上限 262144；该数值表示每条请求的上限，8 条请求可同时驻留的长度仍受缓存容量限制。
- 模型始终只有 1 层 MTP；`--mtp-tokens` 表示复用这一层连续 draft 的 token 数，不是 MTP 层数。

停服后先应用 vLLM 0.20 attention 修改，再使用 runtime 启动器，重启后生效：

```bash
python /path/to/orin_qwen38_patch/apply_attention_patch.py
python /path/to/orin_qwen38_patch/serve.py \
  --model /path/to/awq-model-with-real-mtp-weights \
  --kv-cache-dtype int8_per_token_head \
  --attention-backend TRITON_ATTN \
  --max-model-len 262144 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 --block-size 2048 \
  --kv-cache-memory-gib 12 --mtp-tokens 3
```

通过 pip 安装的环境可在 `site-packages` 目录应用上述 `vllm/...` 路径补丁。
补丁与 runtime 启用需要同时完成。`--baseline` 关闭 runtime hooks，仍保留已应用的 vLLM attention 源码修改。
当前分支的 MTP 动态 batch 优化及其单独对照开关 `--no-fast-draft` 见 [DYNAMIC_BATCH.md](DYNAMIC_BATCH.md)；下面的成绩是加入该优化之前的测量。

原 AWQ safetensors 文件没有 MTP 权重，虽然旧索引列出了相应名称。此次本机测试使用单独的模型目录，保留原 AWQ trunk，并从本机合并 GGUF 恢复 MTP 的 Q4_K/Q6_K 权重为 FP16，恢复 Gemma RMSNorm 的原始参数表示。
这是已量化草稿权重的恢复，不等同于原始 FP16 MTP checkpoint；模型权重不包含在补丁中。
实际部署应使用完整且匹配的 MTP 权重，不能只修改索引或启用参数。

256K 下，16 个目标 full-attention 层的 INT8 KV（包括 scale）约需 8.125 GiB；MTP attention、GDN state、分页 padding 和工作区另计。
Prefill 的临时 FP16 K/V 工作区在单请求 256K 时约为 1 GiB。8 并发测试使用较短输入，不要求 8×256K。

本机还有其他常驻服务。16 GiB KV pool 的长提示批次触发了宿主机 `lzc-earlyoom`：内核信号追踪确认它向实验 engine 发送 SIGTERM，日志记录的可用 RAM 为 2890 MiB，低于约 3.2 GiB 阈值。此配置的中止请求不计入成绩，最终配置降低到 12 GiB。KV pool 的大小不包含权重、CUDA Graph、prefill scratch 或 CUDA allocator 保留的空闲块。

测量定义：冷请求 prefill TPS 是整批未缓存输入 token 除以从最早请求发出到最后一条请求首 token 的时间，包含排队和首 token 开销。客户端 decode TPS 是首个 token chunk 之后实际输出 token 除以首末 chunk 的时间差。E2E 输出 TPS 是所有输出 token 除以完整请求耗时，包含 prefill、排队和 TTFT，不能与 decode TPS 混用。8 并发同时给出每用户中位数和整批吞吐。冷请求的 decode 区间可能包含后续请求的 prefill；另记录所有用户都已开始且未结束输出的共同区间内的 decode 吞吐。
Prefix cache 测试复用原冷请求的完整前缀并追加 1 token，避免把缓存命中后的逻辑输入长度除以 TTFT 当作计算吞吐。MTP 的每个首 chunk 可能包含多个 token，测量会扣除整个首 chunk。

逐 token 一致性检查：本次 128-token greedy 请求的代码与复述输出，在 draft=0/1/2/3/4/8 的单用户及 8 并发中一致。创意写作出现部分第 51 token 分歧；关闭 MTP 的 8 个相同 greedy 请求也产生两种输出。因此，此配置没有逐 token 的批次不变性保证；数值对照通过不等同于完整模型精度评估。

## 短提示 MTP 对比

以下均为客户端 **decode TPS**，每格为「单用户 / 8 并发每用户中位数」。每条固定输出 128 token，temperature=0；本节短提示测试的 KV pool 为 16 GiB，未触发内存保护。模型只有 1 层 MTP。

| Draft token 数 | 创意写作 | 代码 | 复述 |
| ---: | ---: | ---: | ---: |
| 0 | 10.0 / 8.7 | 10.0 / 8.7 | 10.0 / 8.7 |
| 1 | 12.2 / 10.2 | 15.1 / 12.6 | 15.8 / 13.2 |
| 2 | 11.9 / 9.7 | 17.6 / 14.4 | 20.0 / 16.4 |
| 3 | 10.6 / 8.5 | 19.3 / 15.2 | 23.1 / 18.4 |
| 4 | 10.4 / 7.0 | 17.8 / 13.4 | 24.8 / 18.7 |
| 8 | 7.1 / 3.8 | 16.2 / 9.6 | 30.8 / 18.5 |

代码短提示为 78 个输入 token。draft3 下，单用户 decode 为 19.31 TPS，E2E 输出为 18.08 TPS；8 并发每用户中位数分别为 15.17 / 12.64 TPS，整批分别为 120.74 / 100.66 TPS。这些是短提示代码生成，未包含完整 agent 工具循环。

创意写作的 greedy 测试中 draft1 最快；代码短提示中 draft3 最快；单用户复述中 draft8 最快，8 并发复述中 draft4 最快。该结论受提示、输出长度和 draft 接受率影响。

## Prefix cache 开关对照

只比较开启缓存是否影响速度，不调命中率或缓存容量。固定 Q8 KV、draft3、12 GiB KV pool、256K 上下文上限、8K batch、2048 block；两组使用完全相同的 8192-token 输入和 128-token greedy 输出，单用户和 8 并发各测两轮，表中为两轮中位数。服务先预热，正式输入使用不同前缀，并在开启组每批前清空缓存，保证两组计算同样的输入工作量。

关闭使用 `--no-prefix-cache`（`mamba-cache-mode=none`），开启使用默认配置（`mamba-cache-mode=align`）。因此本对照包含 GDN 状态保存和调度变化，不只测 CPU hash 开销。

8 并发的 prefill、decode 和 E2E 列均为**整批吞吐**；decode 使用所有用户均在输出的共同区间，E2E 包含 prefill 和排队。

| 并发 | Prefix cache | Prefill TPS | 稳定 decode TPS/整批 | E2E 输出 TPS/整批 | TTFT 中位数/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 关 | 784.1 | 18.37 | 7.373 | 10.45 |
| 1 | 开 | 773.3 | 18.77 | 7.373 | 10.59 |
| 8 | 关 | 751.4 | 98.20 | 10.505 | 69.89 |
| 8 | 开 | 703.9 | 94.69 | 9.943 | 66.59 |

开启相对关闭的变化：

- 1 并发：prefill -1.4%，稳定 decode +2.2%，E2E 输出 +0.0%。
- 8 并发：prefill -6.3%，稳定 decode -3.6%，E2E 输出 -5.4%。

所有请求完整输出 128 token，preemption 增量均为 0；开关组逐请求输出 token ID 相同的比例为 1/18。Greedy 输出存在分歧，decode 与 E2E 的差异包含输出轨迹和 MTP 接受率变化，不能全部归因于缓存管理开销。两轮和较短 decode 区间不足以判断微小差异的统计显著性，结论只对应此 8K 输入与代码生成负载。

## 12 GiB KV pool 的上下文测试

本节采用 draft3、temperature=0、每条输出 128 token；每种长度两轮冷请求和两轮前缀复用。提示为合成缓存策略长文本及代码生成要求。

表中 decode 为所有请求都已开始且未结束输出的共同区间内的每用户中位数；E2E 是每用户完整请求的输出 TPS 中位数。8 并发的冷 prefill 是整批吞吐，包含排队和交错 decode，不能当作单请求内核速度。

| 并发 | 输入 token/条 | 冷 prefill TPS/整批 | 冷 TTFT 中位数/s | 冷稳定 decode TPS/用户 | 冷 E2E 输出 TPS/用户 | 热 TTFT 中位数/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 744.6 | 2.75 | 17.97 | 13.03 | 0.42 |
| 1 | 8192 | 771.7 | 10.62 | 18.82 | 7.37 | 0.40 |
| 1 | 32768 | 674.0 | 48.62 | 16.98 | 2.28 | 0.44 |
| 8 | 2048 | 749.5 | 21.61 | 14.29 | 4.18 | 10.95 |
| 8 | 4096 | 714.3 | 36.08 | 13.13 | 2.34 | 24.55 |
| 8 | 8192 | 698.0 | 67.18 | 12.27 | 1.24 | 52.75 |

所有已完成的 24 组请求均完整输出，preemption counter 增量为 0。热请求的部分 miss 属于缓存复用效果，不能从零 preemption 推断全部前缀仍驻留。单用户 8K 热请求的 E2E 输出约为 18.69 TPS，冷请求约为 7.37 TPS。

## 实际 256K 请求

单请求冷输入 262016 token，输出 128 token，合计 262144；这轮未命中缓存。合成记录中间位置的访问码正确匹配。结果仅验证这一个检索样例与性能，不能推断完整长上下文任务准确率。

| 测试 | 输入 token | 输出 token | 缓存命中 token | TTFT/s | Decode TPS | E2E 输出 TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 冷请求 | 262016 | 128 | 0 | 893.27 | 10.89 | 0.141 |
| 缓存保留测试 | 262017 | 64 | 258048 | 23.82 | 11.99 | 2.201 |

冷请求 prefill 为 **293.32 TPS**。缓存保留测试复用整个冷输入并追加合成记录的首 token，输出转为记录续写；它只用于测量缓存保留和 TTFT，不能与冷请求的代码输出直接比较 decode TPS。Mamba align checkpoint 命中 258048 token，仍重算末尾 3969 token。两次输出长度也不同，因此不能用两行 E2E TPS 的比值当作固定工作量加速比。
