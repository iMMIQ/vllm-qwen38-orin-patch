# MTP 动态 batch

基于 vLLM 0.20.0、Jetson AGX Orin / SM87。启用 `--mtp-tokens` 时默认接入；
`--no-fast-draft` 使用原 FP16 草稿投影和 LM head，保留其余 Q8 KV、prefill 和 attention 优化。
本次对照的 baseline 就是这一配置，不能称为原版 vLLM。

```bash
python orin_qwen38_patch/serve.py \
  --model /path/to/awq-model-with-real-mtp-weights \
  --mtp-tokens 3 --max-num-seqs 32
```

## 实现

- 草稿 FC、QKV/gate、attention output、FFN gate/up、down 五类投影使用 W8A8。
  从已加载的 FP16 权重派生 per-output-channel INT8 副本，激活逐行量化。
  INT32 累加后应用 FP32 scale，投影输出为 FP16。
- Triton GEMM 的 M 是运行时参数，显式 `do_not_specialize=["M"]`。
  行 tile 选择 `min(128, max(16, next_power_of_two(M)))`，列 tile=64、K tile=128。
  只预编译 16/32/64/128 四种行 tile；更大的 M 增加行方向 grid。
  尾部以 M/N mask 处理，没有 M=8/32 的白名单或超出该范围的 FP16 回退。
  N、K、dtype 和 tile 仍然按投影形状编译，不能把“四种”理解为整个模型只有四个二进制。
- LM head 每行先做 INT8 粗选 top-16，再加入 token 0，以原始 FP16 权重重排候选。
  单行使用 linear，多行使用 batched matmul。支持空 batch，零 logits 保留最小 token ID。
  候选集近似仍可能遗漏原 FP16 的最优 token；它仅影响草稿，目标模型仍执行验证。
- 启动时预编译四种 GEMM tile，并按配置的请求桶预热 head。
  目标验证的 CUDA Graph 桶自动由 `max_num_seqs`、draft 数和 token budget 推导。
  例如 32 请求、draft3 对应 `[4,8,16,32,64,128]`；13 请求对应 `[4,8,16,32,52]`。
  沿用 `FULL_DECODE_ONLY`，并没有把完整 MTP proposer 改成 CUDA Graph。

加载接口仍为原 AWQ target 与完整 FP16 MTP tensors，不引入新的权重量化文件格式。
保留原 FP16 草稿权重及共享 LM head，派生 INT8 副本额外占用约 1.58 GiB。
目标模型 W4、已有 W8A8 prefill、Q8 KV 与 FP32 GDN state 保持原配置。
动态 M 解决算子覆盖和编译问题；实际请求并发仍受 KV/GDN 状态、显存和调度约束。

## 验证与测量边界

GPU 测试覆盖 M=1、2、3、7、8、15、16、17、18、31、32、33、47、64、65、127、128、129、257。
随机矩阵 N=137、K=5120 的量化整数点积与 CPU INT64 参考逐项一致；这些行数只产生四种 GEMM 编译版本。
额外验证奇数 batch 的 CUDA Graph replay、空 batch、零 logits 与奇数容量 graph 桶。
真实 LM head 在保存的 128 条 hidden states 上，以八种 batch 切片共检查 270 行，FP16 argmax 全部一致；
切片有重叠，不能当作 270 个独立精度样本，也不构成完整任务准确率评估。

端到端采用 temperature=0、draft3、prefix cache 开启、12 GiB KV pool、262144 上下文配置、
8192 batch token、FP32 GDN，GPU 1300.5 MHz。每种正式负载测两轮，冷请求前重置 prefix cache。
单用户和短提示并发每条输出 256 token，8K 输入的 8 并发每条输出 128 token。
本次端到端最长实际输入为 8192 token，没有重新跑动态草稿版本的完整 256K 请求。

Decode 吞吐扣除整个首 chunk；整批 decode 使用最早首 chunk 到最晚末 chunk 的时间。
共同区间吞吐只在所有请求都已开始且均未结束时计算，没有共同区间则记为 null。
Prefill 是未缓存输入量除以首 token 等待时间，包含首 token、排队和交错 decode；
E2E 是完整请求的输出吞吐。高提交并发下这些量不能当作纯内核吞吐。
结果与限制见 [dynamic-draft-results.json](verification/dynamic-draft-results.json)。

两轮中位数，baseline 使用 `--no-fast-draft`：

| 负载与指标 | FP16 草稿 | 动态草稿 | 变化 |
| --- | ---: | ---: | ---: |
| 短输入代码，单用户 decode TPS（三种提示） | 18.07 / 21.13 / 19.02 | 22.09 / 24.20 / 22.63 | +22.2% / +14.5% / +19.0% |
| 8K 单用户 prefill TPS | 775.39 | 785.37 | +1.3% |
| 8K 单用户 decode TPS | 18.88 | 22.11 | +17.1% |
| 短输入 8 并发，整批 decode TPS | 115.57 | 133.69 | +15.7% |
| 短输入 8 并发，每用户 decode TPS 中位数 | 15.12 | 17.21 | +13.9% |
| 8K 输入 8 并发，整批 prefill TPS | 705.74 | 717.96 | +1.7% |
| 8K 输入 8 并发，共同区间 decode TPS/整批 | 95.86 | 105.36 | +9.9% |

真实 head 的单行耗时从 14.21 ms 降至 7.28 ms；M=32 从 14.62 降至 8.97 ms。
M=128 时 top-k 和 FP16 重排成本已经明显，完整 head 从 16.72 降至 15.79 ms，
不能把小 batch 的加速倍数外推到任意并发。
两轮短生成测试没有置信区间；输出轨迹和 MTP 接受率也会影响吞吐。

`max_num_seqs=32` 的短输入对照如下，仍然使用同一个 12 GiB KV pool：

| 提交请求数 | FP16 草稿整批 decode TPS | 动态草稿整批 decode TPS | FP16 草稿 E2E 输出 TPS | 动态草稿 E2E 输出 TPS |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 49.73 | 59.94 | 47.46 | 56.70 |
| 7 | 103.15 | 120.30 | 95.76 | 110.51 |
| 16 | 110.31 | 122.49 | 104.27 | 114.95 |
| 32 | 127.77 | 140.28 | 123.65 | 135.49 |

两组服务日志的实际运行峰值都是 13 请求，KV 使用峰值 96%，等待峰值 19 请求。
FP32 GDN 状态、speculative 状态和页面 padding 也占用缓存池；上下文很短仍然需要这些状态。
16/32 提交请求没有所有请求共同的 decode 区间，上表整批 decode 包含交错 prefill 和等待间隙，
不代表 16/32 条请求同时 decode 的纯吞吐。全部正式批次的抢占计数增量为 0。

逐请求完整输出与对应 FP16 草稿对照：单用户和短输入 8 并发全部匹配；
8K 输入 8 并发为 5/16 匹配；32 请求短输入为 44/64 匹配，3/7/16 请求均匹配。
这种轨迹差异尚未通过 teacher-forced 诊断隔离原因，不能据此保证模型逐 token 一致或量化无精度损失。

复验动态 kernel（不需要模型权重，在上述 CUDA/Triton 环境运行）：

```bash
python verification/check-dynamic-draft.py
```
