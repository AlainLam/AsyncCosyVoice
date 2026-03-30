# Async CosyVoice

[中文](./README.md) | [English](./README.EN.md)

本项目将原本 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 中的同步 LLM 推理改为基于 `AsyncLLMEngine` 的异步实现，并在推理过程中做了一些针对场景的优化调整。

## 主要改动

- 将 CosyVoice 的同步 LLM 推理链路改为基于 `AsyncLLMEngine` 的异步实现
- 调整了流式首包阶段的 token hop 策略，用于平衡首包延迟与音频质量
- 对 CosyVoice3 的指令输入做了规范化处理
- 避免对同一音频反复调用 `load_wav`
- 为部分关键变量补充了类型注解，方便阅读和维护
- 提供了一个用于验证和接入的 HTTP 服务层，同时补充了 OpenAI 兼容端点
- 自动加载`assets`目录下的`*.wav`以及`*.txt`文件

## 首包策略调整

官方训练中使用 `token_hop_len=25`，且官方推荐对齐 25 token 网格。当前实现支持让首个流式 chunk 使用更小的 hop 长度，用于在首包延迟和音频质量之间做权衡，并在后续 chunk 自动回归到对齐网格。

- 推荐使用 `initial_token_hop_len=15`

## 关于 ONNX fp16

我尝试过使用 yuekai 的 [yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX](https://huggingface.co/yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX)

- 在 4090 环境中没有得到实际提升，在 5090 环境中提升也比较有限
- 目前我直接使用官方的 onnx 在 4090 上没有遇到 NaN 问题
- 为了保持首包性能并尽量减少额外复杂度，这部分暂时没有纳入当前实现
- 如果你对这部分有兴趣，可以参考 `./CosyVoice/runtime/triton_trtllm` 中的相关实现

## 已知限制

- 当前主要针对 CosyVoice 3.0 异步优化，未修复 CosyVoice 3.0 本身已知的其他音色问题
- CosyVoice 2.0 目前无法正确运行，暂未修复

## 如何使用

1. 克隆项目及 submodule

```bash
git clone --recursive https://github.com/AlainLam/AsyncCosyVoice.git
```

2. 确保 submodule 已正确初始化

```bash
cd AsyncCosyvoice
git submodule update --init --recursive
```

3. 使用 conda 创建隔离环境

```bash
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice
```

4. CosyVoice 需要低于 82.0 的 `setuptools`，这里直接使用一组兼容版本

```bash
pip install --force-reinstall numpy==1.26.4 setuptools==59.6.0 wheel==0.37.1 wheel_stub==0.5.0
```

5. 安装依赖：先安装 CosyVoice 的依赖，再安装当前项目的依赖

```bash
pip install --no-build-isolation -r CosyVoice/requirements.txt
pip install --no-build-isolation -r requirements.txt
```

6. 确保你的模型文件已经准备好，如果没有，请使用以下命令下载：

建议使用独立的 `.venv` 环境安装 `huggingface_hub` 或 `modelscope` SDK 来下载模型，以避免与当前项目的依赖冲突。

从 Hugging Face Hub 下载：

```bash
python -m venv .venv
.venv/bin/pip install -U huggingface_hub
.venv/bin/hf download "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" --local_dir ./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
.venv/bin/hf download "FunAudioLLM/CosyVoice-ttsfrd" --local_dir ./CosyVoice/pretrained_models/CosyVoice-ttsfrd
rm -rf .venv
```

从 Modelscope 下载：

```bash
python -m venv .venv
.venv/bin/pip install -U modelscope
.venv/bin/modelscope download "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" --local_dir ./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
.venv/bin/modelscope download "iic/CosyVoice-ttsfrd" --local_dir ./CosyVoice/pretrained_models/CosyVoice-ttsfrd
rm -rf .venv
```

7. 启动服务

如果模型放在默认目录下，可以直接启动：

```bash
python -m app.main
```

如果需要显式指定模型目录，可以使用：

```bash
python -m app.main \
  --model-dir ./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
  --ttsfrd-dir ./CosyVoice/pretrained_models/CosyVoice-ttsfrd \
  --port 8000
```

## 测试结果

单卡4090测试。

说明：

- 下表中只有第一项 `并发=1` 可以视为真正的冷启动
- 后续 `并发=2/3/4` 时服务实际上已经完成首轮初始化，不再属于严格意义上的冷启动
- 根据实际经验，TensorRT 需要经过多次预热后状态才会更稳定，因此这里保留了 `1-4` 并发的预热阶段测试结果
- 实际参考时建议优先关注正式测试部分
- **不同的参考音频可能会对性能和质量产生较大影响**，这里使用官方的测试音频进行测试

预热阶段测试：

| 并发 | Min | Max | Avg | 成功 |
| --- | --- | --- | --- | --- |
| 1 | 1125 ms | 1125 ms | 1125 ms | 1/1 |
| 2 | 227 ms | 345 ms | 286 ms | 2/2 |
| 3 | 229 ms | 359 ms | 276 ms | 3/3 |
| 4 | 269 ms | 417 ms | 315 ms | 4/4 |

正式测试：

| 并发 | Min | Max | Avg | 成功 |
| --- | --- | --- | --- | --- |
| 1 | 197 ms | 197 ms | 197 ms | 1/1 |
| 2 | 193 ms | 231 ms | 212 ms | 2/2 |
| 3 | 246 ms | 293 ms | 277 ms | 3/3 |
| 4 | 221 ms | 368 ms | 328 ms | 4/4 |
| 8 | 374 ms | 661 ms | 514 ms | 8/8 |

日志摘录：

```text
2026-03-29 10:10:51,836 [INFO] app.cosyvoice.cosyvoice: synthesis text 晨光透过百叶窗，在地板上划出平行的金色条纹。
WARNING 03-29 10:10:51 [preprocess.py:60] Using None for EOS token id because tokenizer is not initialized
2026-03-29 10:10:52,021 [INFO] app.cosyvoice.cosyvoice: yield speech len 0.440, rtf 0.419, latency 0.184 sec
2026-03-29 10:10:52,180 [INFO] app.cosyvoice.cosyvoice: yield speech len 1.920, rtf 0.083, latency 0.159 sec
2026-03-29 10:10:52,320 [INFO] app.cosyvoice.cosyvoice: yield speech len 2.000, rtf 0.070, latency 0.139 sec
2026-03-29 10:10:52,447 [INFO] app.cosyvoice.cosyvoice: yield speech len 0.920, rtf 0.137, latency 0.126 sec
```

## 补充说明

- `example/test_client.py` 仅用于快速验证，不作为生产示例
- `app/http.py`
  - 采用“先注册，再推理”的使用方式，需要先上传参考音频生成 `voice_id`
  - 当前 HTTP 接口没有处理 voice 持久化

## 致谢

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
