# Qwen3.8-27B Orin prefill patch for vLLM 0.20

针对 **Jetson AGX Orin / SM87** 的 Qwen3.8-27B AWQ prefill 优化。使用 TileLang 生成的
INT8 内核和进程级 runtime hooks，保留 vLLM 的调度、模型加载、KV cache 和原生 decode。

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
python orin_qwen38_patch/serve.py --model /path/to/awq-model
```

或者从 [v0.1.1 Release](https://github.com/iMMIQ/vllm-qwen38-orin-patch/releases/tag/v0.1.1)
下载完整 patch，在自己的工作目录应用：

```bash
git apply --check /path/to/vllm-0.20-qwen38-orin-prefill.patch
git apply /path/to/vllm-0.20-qwen38-orin-prefill.patch
python orin_qwen38_patch/serve.py --model /path/to/awq-model
```

默认服务地址 `http://127.0.0.1:8000`，模型名 `qwen38-orin`。
支持 `--host`、`--port`、`--served-model-name`、`--gpu-memory-utilization`；
`--baseline` 用相同服务配置关闭优化，供性能对照。

```bash
python orin_qwen38_patch/benchmark.py --url http://127.0.0.1:8000
```

## 固定算子流水

- 保留 Marlin Q4 权重布局，LUT 解包成临时 per-channel W8，使用共享 scratch 顺序复用。
- Per-token A8；INT8 GEMM 固定 `128×128×128`、2 stages、128 threads、M 分组为 2。
- 融合 Gemma norm/residual、SiLU/mul/A8、GDN output norm/gate/A8。
- 8K 单序列采用 head-first exp2 WY，其余形状使用原生 WY。
- `1024 <= M <= 8192` 使用优化投影；短输入、decode 和不支持的参数走原生算子。

已清理实验路径，仅保留一个 `QWEN38_ORIN_PREFILL` 启用开关，由启动器设置。
源码结构和详细约束见 [使用说明](orin_qwen38_patch/README.txt)。

## 支持范围

已验证 AGX Orin 64GB、L4T 36.4.3、CUDA 12.6、Torch 2.11.0、Triton 3.5.1。
单 GPU、TP=1、单并发、文本输入、FP16 activation/KV、FP32 GDN state。
启动器固定每批最多 8192 token、上下文上限 12288；MTP 和 prefix cache 关闭。

完整 patch 和仓库包含 **24 个预编译 SM87 cubin**。源码版本需要在配置好的
TileLang 0.1.13 环境中运行：

```bash
python orin_qwen38_patch/kernels/build.py
```

## 性能与精度

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
