# Async CosyVoice

[中文](./README.md) | [English](./README.EN.md)

This project modifies the original synchronous LLM inference in CosyVoice to an asynchronous implementation using AsyncLLMEngine, along with some scenario-based optimizations in the inference process.

## Main Changes

- Replaced the original synchronous CosyVoice LLM inference path with an `AsyncLLMEngine`-based asynchronous implementation
- Adjusted the token hop strategy for the first streaming chunk to balance first-packet latency and audio quality
- Normalized instruction handling for CosyVoice3
- Avoids calling `load_wav` repeatedly for the same audio input
- Added type annotations for some key variables to make the code easier to read and maintain
- Added an HTTP service layer for validation and integration, with OpenAI-compatible endpoints
- Automatically loads `*.wav` and `*.txt` files from the `assets` directory

## First-Packet Strategy

Official training uses `token_hop_len=25`, and the official recommendation is to stay aligned to the 25-token grid. This implementation allows the first streaming chunk to use a smaller hop length to trade off first-packet latency against audio quality, and then automatically realigns subsequent chunks back to the original grid.

- Recommended setting: `initial_token_hop_len=15`

## About ONNX fp16

I also tried yuekai's [yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX](https://huggingface.co/yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX).

- It did not show a practical improvement on an RTX 4090, and the gain on an RTX 5090 was also limited
- I have not seen NaN issues on an RTX 4090 when using the official ONNX export
- To keep first-packet performance stable and avoid extra complexity, this optimization is not included for now
- If you want to explore it further, you can refer to the related implementation under `./CosyVoice/runtime/triton_trtllm`

## Known Limitations

- The current work mainly targets asynchronous optimization for CosyVoice 3.0 and does not attempt to fix other known voice issues in CosyVoice 3.0 itself
- CosyVoice 2.0 is currently not working correctly and has not been fixed

## Usage

1. Clone the project with its submodule

```bash
git clone --recursive https://github.com/AlainLam/AsyncCosyVoice.git
```

2. Make sure the submodule is initialized correctly

```bash
cd AsyncCosyvoice
git submodule update --init --recursive
```

3. Create an isolated conda environment

```bash
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice
```

4. CosyVoice requires a `setuptools` version lower than 82.0, so a compatible set of package versions is used here

```bash
pip install --force-reinstall numpy==1.26.4 setuptools==59.6.0 wheel==0.37.1 wheel_stub==0.5.0
```

5. Install dependencies: first install CosyVoice dependencies, then install this project's dependencies

```bash
pip install --no-build-isolation -r CosyVoice/requirements.txt
pip install --no-build-isolation -r requirements.txt
```

6. Make sure your model files are ready. If not, download them with one of the following methods:

It is recommended to use an isolated `.venv` environment to install `huggingface_hub` or `modelscope` so you do not introduce dependency conflicts into the main environment.

Download from Hugging Face Hub:

```bash
python -m venv .venv
.venv/bin/pip install -U huggingface_hub
.venv/bin/hf download "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" --local_dir ./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
.venv/bin/hf download "FunAudioLLM/CosyVoice-ttsfrd" --local_dir ./CosyVoice/pretrained_models/CosyVoice-ttsfrd
rm -rf .venv
```

Download from ModelScope:

```bash
python -m venv .venv
.venv/bin/pip install -U modelscope
.venv/bin/modelscope download "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" --local_dir ./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
.venv/bin/modelscope download "iic/CosyVoice-ttsfrd" --local_dir ./CosyVoice/pretrained_models/CosyVoice-ttsfrd
rm -rf .venv
```

7. Start the service

If your models are placed in the default directories, you can start the service directly:

```bash
python -m app.main
```

If you want to specify the model paths explicitly:

```bash
python -m app.main \
  --model-dir ./CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
  --ttsfrd-dir ./CosyVoice/pretrained_models/CosyVoice-ttsfrd \
  --port 8000
```

## Test Results

Single RTX 4090 test.

Notes:

- Only the first `concurrency=1` entry below should be treated as a true cold start
- The later `concurrency=2/3/4` runs are no longer strict cold-start runs because the service has already completed its first initialization
- In practice, TensorRT usually needs several warm-up rounds before it reaches a more stable state, so the `1-4` warm-up-stage results are kept here
- For practical reference, the formal benchmark section is the more important one
- **Different reference audio can noticeably affect both performance and quality**; the test here uses the official reference audio

Warm-up stage:

| Concurrency | Min | Max | Avg | Success |
| --- | --- | --- | --- | --- |
| 1 | 1125 ms | 1125 ms | 1125 ms | 1/1 |
| 2 | 227 ms | 345 ms | 286 ms | 2/2 |
| 3 | 229 ms | 359 ms | 276 ms | 3/3 |
| 4 | 269 ms | 417 ms | 315 ms | 4/4 |

Formal benchmark:

| Concurrency | Min | Max | Avg | Success |
| --- | --- | --- | --- | --- |
| 1 | 197 ms | 197 ms | 197 ms | 1/1 |
| 2 | 193 ms | 231 ms | 212 ms | 2/2 |
| 3 | 246 ms | 293 ms | 277 ms | 3/3 |
| 4 | 221 ms | 368 ms | 328 ms | 4/4 |
| 8 | 374 ms | 661 ms | 514 ms | 8/8 |

Log excerpt:

```text
2026-03-29 10:10:51,836 [INFO] app.cosyvoice.cosyvoice: synthesis text 晨光透过百叶窗，在地板上划出平行的金色条纹。
WARNING 03-29 10:10:51 [preprocess.py:60] Using None for EOS token id because tokenizer is not initialized
2026-03-29 10:10:52,021 [INFO] app.cosyvoice.cosyvoice: yield speech len 0.440, rtf 0.419, latency 0.184 sec
2026-03-29 10:10:52,180 [INFO] app.cosyvoice.cosyvoice: yield speech len 1.920, rtf 0.083, latency 0.159 sec
2026-03-29 10:10:52,320 [INFO] app.cosyvoice.cosyvoice: yield speech len 2.000, rtf 0.070, latency 0.139 sec
2026-03-29 10:10:52,447 [INFO] app.cosyvoice.cosyvoice: yield speech len 0.920, rtf 0.137, latency 0.126 sec
```

## Notes

- `example/test_client.py` is only for quick verification and is not intended as a production example
- `app/http.py`
  - It follows a "register first, then infer" workflow, so you need to upload reference audio to create a `voice_id` before using the inference endpoint
  - The current HTTP interface does not handle voice persistence

## Acknowledgements

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)
