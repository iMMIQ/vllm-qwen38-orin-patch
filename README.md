# Qwen3.8-27B Orin patch for vLLM 0.20

**v0.2.0 包含 Q8 KV、动态 batch MTP、prefix cache 与 262144 上下文配置。**
新配置见 [Q8/MTP 实验说明](Q8_MTP_256K.md)。下面的旧成绩对应已发布 v0.1.1 的 FP16 KV 单并发配置。
启用 MTP 时默认使用动态行数的草稿投影和多行 LM head，见 [动态 batch 说明](DYNAMIC_BATCH.md)。
完整 256K 请求已在之前的 Q8 配置上验证；动态草稿版本本轮实际输入最长为 8K。

针对 **Jetson AGX Orin / SM87** 的 Qwen3.8-27B AWQ 优化。使用 TileLang 生成的
INT8 内核和进程级 runtime hooks，保留 vLLM 的调度和模型加载，优化 prefill、Q8 attention 和 MTP 验证。

本机实测 2K / 8K prefill 约 **842 / 818 tokens/s**，decode 约 **10 tokens/s**。
这是模型、版本和单并发配置固定的专项补丁。

## 使用

需要已有可运行的 **vLLM 0.20.0 CUDA 环境**和本地模型权重。
模型为 `philbert440/Qwen3.8-27B-W4A16-AWQ`，固定 revision 和 SHA256 见
[model-reference.json](orin_qwen38_patch/model-reference.json)。
权重接口是 **compressed-tensors / pack-quantized / asymmetric W4 / group-128**。

直接使用仓库中的清理版源码和预编译内核：

```bash
git clone https://github.com/iMMIQ/vllm-qwen38-orin-patch.git
cd vllm-qwen38-orin-patch
python orin_qwen38_patch/check.py --model /path/to/awq-model
python orin_qwen38_patch/apply_attention_patch.py
python orin_qwen38_patch/serve.py --model /path/to/awq-model
```

[v0.2.0 Release](https://github.com/iMMIQ/vllm-qwen38-orin-patch/releases/tag/v0.2.0)
提供完整 patch（包含预编译 SM87 cubin）、源码 patch 和从 v0.1.1 更新的增量 patch。
下载说明与验证结果见 Release 附件 `RELEASE.txt`；安装前用附件 `SHA256SUMS` 校验。
完整 patch 在尚无 `orin_qwen38_patch` 的工作目录应用；增量 patch 应用于 v0.1.1 补丁仓库
`07e9898`，并非直接应用于原生 vLLM checkout。上述 clone 命令直接使用当前完整源码。

历史 [v0.1.1 Release](https://github.com/iMMIQ/vllm-qwen38-orin-patch/releases/tag/v0.1.1)
包含旧的 FP16 KV、单并发、关闭 MTP/prefix cache 的完整 patch，可在自己的工作目录应用：

```bash
git apply --check /path/to/vllm-0.20-qwen38-orin-prefill.patch
git apply /path/to/vllm-0.20-qwen38-orin-prefill.patch
python orin_qwen38_patch/serve.py --model /path/to/awq-model
```

默认服务地址 `http://127.0.0.1:8000`，模型名 `qwen38-orin`。
支持 `--host`、`--port`、`--served-model-name`、`--gpu-memory-utilization`；
当前版本默认 Q8 KV、12 GiB cache、prefix cache、8 并发、262144 上下文，MTP 默认关闭。
启用 MTP 需要完整权重并设置 `--mtp-tokens`，详见 [Q8/MTP 说明](Q8_MTP_256K.md)。
`--baseline` 关闭 runtime hooks；已经应用的 vLLM attention 源码修改仍然生效。

```bash
python orin_qwen38_patch/benchmark.py --url http://127.0.0.1:8000
```

## 固定算子流水

- 保留 Marlin Q4 权重布局，LUT 解包成临时 per-channel W8，使用共享 scratch 顺序复用。
- Per-token A8；INT8 GEMM 固定 `128×128×128`、2 stages、128 threads、M 分组为 2。
- 融合 Gemma norm/residual、SiLU/mul/A8、GDN output norm/gate/A8。
- 8K 单序列采用 head-first exp2 WY，其余形状使用原生 WY。
- `1024 <= M <= 8192` 使用优化投影；短输入、decode 和不支持的参数走原生算子。

启动器设置 `QWEN38_ORIN_PREFILL` 与独立的 `QWEN38_ORIN_DRAFT` 开关。
MTP 草稿投影使用 Triton 的运行时行数和四种固定 tile；目标模型 decode 仍使用原有路径。
源码结构和详细约束见 [使用说明](orin_qwen38_patch/README.txt)。

## 支持范围

已验证 AGX Orin 64GB、L4T 36.4.3、CUDA 12.6、Torch 2.11.0、Triton 3.5.1。
单 GPU、TP=1、文本输入、FP16 activation、FP32 GDN state。
测试 Q8 KV、单用户/8 并发与 262144 上下文；每批最多 8192 token。
8 并发使用较短输入，缓存容量决定同时驻留的总上下文长度。

完整 patch 和仓库包含 **24 个预编译 SM87 cubin**。源码版本需要在配置好的
TileLang 0.1.13 环境中运行：

```bash
python orin_qwen38_patch/kernels/build.py
```

## v0.1.1 历史性能与精度

GPU 1300.5 MHz，单并发，每种长度预热后测三次，表中为中位数。
Prefill TPS = 输入 token / 客户端 TTFT，包含首个输出 token 和 HTTP 开销。

| 输入 token | 原生 AWQ prefill TPS | 优化 prefill TPS | 优化 decode TPS |
| ---: | ---: | ---: | ---: |
| 512 | 453.51 | 454.55 | 9.984 |
| 2048 | 466.27 | 841.83 | 9.927 |
| 8192 | 438.31 | 818.42 | 9.707 |

原生数值来自较早的同机对照，输出 64 token；优化数值来自本次清理回归，输出 32 token。
测量条件与结果见 [性能记录](verification/end-to-end.json) 和
[原始对照摘要](orin_qwen38_patch/evidence/performance.json)。

有限 teacher-forced 诊断语料，相对原始 AWQ 的聚合 PPL 约 **+0.923%**。
数据含重叠前缀，不能换算为任务准确率；未完成完整 code-agent 或长输出评估。
具体数据及边界见 [精度记录](orin_qwen38_patch/evidence/ACCURACY.txt)。

## 发布验证

- 完整和纯源码 patch 均通过应用、逐字节恢复和反向应用检查。
- 独立容器仅挂载应用后的 patch 和原始 AWQ 模型，256 个投影接入，融合与 8K WY 执行。
- 512 / 2K / 8K 共 9 次正式请求，输出 token 与清理前完全一致。
- 清理后的源码独立重建 24 个 cubin，SHA256 全部与已验证产物一致。
- 24 个内核数值对照、尾部 padding 和零激活行检查通过。

验证摘要见 [verification/](verification/)。下载后可用 `sha256sum -c SHA256SUMS`
检查 patch 文件。卸载前停止服务，再用对应 patch 执行 `git apply -R`。

## 许可证

与 vLLM 相同，采用 **Apache License 2.0**，见 [LICENSE](LICENSE)。
TileLang 模板的 MIT 声明保留在 [第三方声明](THIRD_PARTY_NOTICES.txt) 和
[TileLang license](orin_qwen38_patch/LICENSES/TileLang-MIT.txt) 中。
模型权重和依赖软件适用各自的许可证。
