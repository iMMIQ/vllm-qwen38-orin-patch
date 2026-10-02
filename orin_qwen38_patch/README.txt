Qwen3.8-27B AWQ patch — vLLM 0.20.0 / Orin SM87
Release v0.2.0 — Q8 KV / dynamic-batch MTP / prefix cache

应用与启动
  git apply --check /path/to/vllm-0.20-qwen38-orin-q8-dynamic-mtp-full.patch
  git apply /path/to/vllm-0.20-qwen38-orin-q8-dynamic-mtp-full.patch
  python orin_qwen38_patch/apply_attention_patch.py
  python orin_qwen38_patch/serve.py --model /path/to/model

使用已有、可运行的 vLLM 0.20 Python 环境。先停服，用 apply_attention_patch.py
应用两处 vLLM attention 源码修改，再通过进程限定的 sitecustomize 安装 hook，
并传播到 spawn worker。apply_attention_patch.py 可重复执行。
--host 默认 127.0.0.1，--port 默认 8000，--served-model-name 默认 qwen38-orin。
--baseline 关闭 runtime hooks，已应用的 vLLM attention 源码修改仍保留。
可先运行 check.py --model ...；启用 MTP 时加 --require-mtp 检查实际 tensors。
默认 Q8 KV、12 GiB cache、prefix cache、8 并发、262144 上下文，MTP 默认关闭。
启用 MTP 的命令、权重要求与测量定义见仓库根目录 Q8_MTP_256K.md。
MTP 默认启用动态行数草稿投影与多行 LM head，见 DYNAMIC_BATCH.md。
--no-fast-draft 仅关闭这两项，保留 Q8 KV、prefill 和分段 attention 优化。

支持范围
- Jetson AGX Orin 64GB，SM87；已验证 L4T 36.4.3、CUDA 12.6、Torch 2.11.0、Triton 3.5.1。
- philbert440/Qwen3.8-27B-W4A16-AWQ；revision/hash 见 model-reference.json。
  HF 配置实际为 qwen3_5。需要 compressed-tensors pack-quantized、非对称 W4、group=128。
- 单 GPU、TP=1、文本输入，FP16 activation、Q8 KV、FP32 GDN state。
- 每批最多 8192 token，上下文上限 262144；8 并发使用较短输入。
- Prefix cache 使用 mamba-cache-mode=align。MTP 使用单层草稿器，draft 长度独立配置。
- 完整 patch 包含 24 个预编译 SM87 cubin，不需要 TileLang 编译器。

算子流水
Marlin Q4 权重 -> LUT 解包到共享临时 W8 -> per-token A8 -> INT8 GEMM。
权重使用 per-channel scale；GEMM 固定 128x128x128、2 stages、128 threads、M 分组=2。
融合 Gemma norm/residual、SiLU/mul/A8、GDN output norm/gate/A8。
8192-token 单序列使用 head-first exp2 WY；其他形状使用原生 WY。
1024 <= M <= 8192 时启用优化投影；短输入、decode 和不支持的参数使用原生算子。
临时权重与激活 buffer 在同一 CUDA stream 顺序复用，不保留整模型 W8 副本。

源码结构
  runtime/qwen38_orin/config.py       固定模型形状、token 范围、padding
  runtime/qwen38_orin/backend.py      权重加载和 W8A8 投影
  runtime/qwen38_orin/fusions.py      Gemma norm 和 FFN 融合
  runtime/qwen38_orin/gdn_fusion.py    GDN 输出融合
  runtime/qwen38_orin/wy_backend.py    8K WY 调用和原生回退
  runtime/qwen38_orin/cuda_runtime.py CUDA driver 当前 stream 启动器
  runtime/qwen38_orin/kv8_attention.py Q8 KV prefill 的临时 FP16 FlashAttention 2 路径
  runtime/qwen38_orin/mtp.py          MTP 权重完整性检查
  runtime/qwen38_orin/draft.py        MTP 草稿投影与多行 LM head 的接入
  runtime/qwen38_orin/draft_kernels.py 运行时 M 的 Triton W8A8 草稿投影
  patches/vllm020-int8-attention.patch 分段 MTP attention 的原生源码修改
  kernels/                           对应 TileLang 源码和构建脚本

QWEN38_ORIN_PREFILL 与 QWEN38_ORIN_DRAFT 启用开关均由 serve.py 设置。
TP、dtype 和 workspace 形状约束固定；scratch 在同一 CUDA stream 顺序复用。
日志 ORIN_INSTALLED 表示 hooks 安装；ORIN_ATTACH count=256 表示模型投影已接入。
长输入执行 ORIN_PREFILL / ORIN_FUSION，8K 输入执行 ORIN_WY M=8192。
日志每类只打印一次。没有计时包装、direct W4A8、备用解包/GEMM 或布局搜索路径。

测量与精度
  python orin_qwen38_patch/benchmark.py --url http://127.0.0.1:8000
工具先预热，再测三次冷请求；每次前缀不同，避免 prefix cache 命中。
Prefill TPS=未缓存输入 token / 客户端 TTFT，包含首 token 和 HTTP 开销。
Decode TPS 扣除整个首 chunk；E2E 输出 TPS=全部输出 token / 完整请求耗时。
历史同机单并发结果：512/2048/8192 prefill 约 455/836/815 TPS；decode 约 10 TPS。
原生 AWQ 对照和测量条件见 evidence/performance.json。工具使用自己的固定提示，
数字可能不同；功耗模式、GPU 频率、上下文和其他负载也会影响结果。
有限 teacher-forced 语料相对原始 AWQ 的聚合 PPL 约 +0.923%，不是任务准确率损失。
数据含重叠前缀，未做完整 code-agent 或长输出评估，详见 evidence/ACCURACY.txt。

重建源码版本
在已配置的 TileLang 0.1.13 / CUDA SM87 环境中：
  python orin_qwen38_patch/kernels/build.py
生成 24 个 cubin、manifest 和校验文件。纯源码 patch 需要先执行这一步。
编译中间 CUDA 由 TileLang 生成，不随 patch 重复分发。
新的编译器产物需要重新验证数值、CUDA Graph 和整模型吞吐。
Python hooks 依赖 vLLM 0.20 接口，预编译 cubin 不依赖 vLLM/Torch C++ ABI。

卸载与许可证
先停止该服务，再运行：
  git apply -R --check /path/to/vllm-0.20-qwen38-orin-q8-dynamic-mtp-full.patch
  git apply -R /path/to/vllm-0.20-qwen38-orin-q8-dynamic-mtp-full.patch
源码 Apache-2.0，见 LICENSE；TileLang 模板声明见 THIRD_PARTY_NOTICES.txt 和 LICENSES/。
patch 不包含模型权重、镜像或依赖安装包。
