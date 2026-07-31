# Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load

Prompt-driven image editing on top of [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511), running in four sampling steps with twenty LoRA styles that download the first time you pick them and stay in memory after that.

The pipeline pairs the 2511 base model with the `prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19` transformer and, where the hardware supports it, a Flash Attention 3 processor. Nothing is preloaded except the base model, so startup does not wait on adapter weights you may never use. Runs on NVIDIA (CUDA 13) and on AMD Instinct through ROCm, including MI300X droplets on DigitalOcean.

Live demo: [prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast)

<img width="1714" height="1596" alt="image" src="https://github.com/user-attachments/assets/7c665c83-d5a0-492a-9ede-074382c6c46a" />

## Editing styles

Every entry below is a LoRA in the registry (`ADAPTER_SPECS` in `app.py`). Pick one from the dropdown, and the weights are pulled from the Hub on first use, fused into the transformer, and cached for the rest of the session.

| Style | Adapter repository | Weights file |
| --- | --- | --- |
| Multiple-Angles | `dx8152/Qwen-Edit-2509-Multiple-angles` | `镜头转换.safetensors` |
| Fal-Multiple-Angles | `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` | `qwen-image-edit-2511-multiple-angles-lora.safetensors` |
| Photo-to-Anime | `autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime` | `Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors` |
| Anime-V2 | `prithivMLmods/Qwen-Image-Edit-2511-Anime` | `Qwen-Image-Edit-2511-Anime-2000.safetensors` |
| Anything2Real | `lrzjason/Anything2Real_2601` | `anything2real_2601.safetensors` |
| Hyper-Realistic-Portrait | `prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait` | `HRP_20.safetensors` |
| Ultra-Realistic-Portrait | `prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait` | `URP_20.safetensors` |
| Pixar-Inspired-3D | `prithivMLmods/Qwen-Image-Edit-2511-Pixar-Inspired-3D` | `PI3_20.safetensors` |
| Noir-Comic-Book | `prithivMLmods/Qwen-Image-Edit-2511-Noir-Comic-Book-Panel` | `Noir-Comic-Book-Panel_20.safetensors` |
| Manga-Tone | `nappa114514/Qwen-Image-Edit-2509-Manga-Tone` | `tone001.safetensors` |
| Polaroid-Photo | `prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo` | `Qwen-Image-Edit-2511-Polaroid-Photo.safetensors` |
| Midnight-Noir-Eyes-Spotlight | `prithivMLmods/Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight` | `Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight.safetensors` |
| Style-Transfer | `zooeyy/Style-Transfer` | `Style Transfer-Alpha-V0.1.safetensors` |
| Light-Migration | `dx8152/Qwen-Edit-2509-Light-Migration` | `参考色调.safetensors` |
| Any-light | `lilylilith/QIE-2511-MP-AnyLight` | `QIE-2511-AnyLight_.safetensors` |
| Studio-DeLight | `prithivMLmods/QIE-2511-Studio-DeLight` | `QIE-2511-Studio-DeLight-5000.safetensors` |
| Cinematic-FlatLog | `prithivMLmods/QIE-2511-Cinematic-FlatLog-Control` | `QIE-2511-Cinematic-FlatLog-Control-3200.safetensors` |
| Passport-Photo | `prithivMLmods/QIE-2511-Studio-DeLight` | `QIE-2511-Studio-DeLight-5000.safetensors` |
| Upscaler | `starsfriday/Qwen-Image-Edit-2511-Upscale2K` | `qwen_image_edit_2511_upscale.safetensors` |
| Unblur-Anything | `prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale` | `Qwen-Image-Edit-Unblur-Upscale_15.safetensors` |

Several styles read more than one input. Style-Transfer takes the source in slot 1 and the reference in slot 2. Any-light and Light-Migration copy lighting or color tone from the second image onto the first.

Passport-Photo reuses the Studio-DeLight weights and adds handling of its own: unless your prompt already says "passport", the app appends a suffix asking for a 3.5:4.5 medium shot with headroom, a plain neutral studio background, uniform lighting, and preserved facial identity and hair, then forces the output to 784x1008 instead of matching the input aspect ratio.

## How the lazy loading works

`infer()` looks up the selected style, and if its adapter name is not in `LOADED_ADAPTERS` yet it calls `pipe.load_lora_weights(...)`, records the name, and activates it with `pipe.set_adapters([name], adapter_weights=[1.0])`. Repeat runs on the same style skip straight to activation. A spec can declare `fallback_repo` and `fallback_weights`, which the loader tries if the primary repository fails.

Attention and precision are decided at startup. The app tries to install `QwenDoubleStreamAttnProcessorFA3` and prints a warning while keeping the default processor if that fails, which is what happens on hardware without an FA3 build. Weights load in bfloat16 when the GPU reports support, float16 on other accelerators, and float32 on CPU.

## Interface

The UI is a single Gradio page with custom HTML, CSS, and JavaScript rather than stock components. Images live in a client-side gallery: drop files on the canvas or click to browse, and each thumbnail carries an index badge and its own remove button, with Remove and Clear All in the toolbar. Base64 payloads are passed to the pipeline through a hidden textbox.

Results open in a viewer with four tabs:

- Split Slider, with a draggable divider over the original
- Side-by-Side
- Generated (Zoom)
- Original Input, with a picker when you uploaded more than one image

Scroll to zoom, drag to pan, hold Space to swap the two images for a quick A/B, press Esc to close. The header shows the output's pixel dimensions, and Save writes a PNG named after your first uploaded file.

Seventeen quick prompt chips fill the prompt box in one click, and twenty example cards load a full setup: images, prompt, and matching style. Thumbnails for those cards are generated at startup and inlined as base64.

Advanced settings expose seed (0 to 2147483647), a randomize toggle that is on by default, guidance from 1 to 10, and steps from 1 to 50. Defaults are 4 steps and guidance 1.0, which is what the Rapid-AIO transformer is tuned for.

Input images are resized so the longest side is 1024 and both dimensions land on multiples of 8, which keeps the latent shapes valid. Passport-Photo overrides this with its fixed 784x1008.

## Repository layout

```text
├── examples/                          # 26 sample images used by the example cards
├── qwenimage/
│   ├── __init__.py
│   ├── pipeline_qwenimage_edit_plus.py
│   ├── qwen_fa3_processor.py
│   └── transformer_qwenimage.py
├── app.py                             # pipeline, adapter registry, and the whole UI
├── digitalocean_startup.sh            # cloud-init script for MI300X droplets
├── pre-requirements.txt
├── requirements.txt                   # CUDA 13 stack
├── requirements-rocm.txt              # ROCm 6.2 stack
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

## Requirements

A GPU with enough VRAM for the 2511 transformer, either CUDA or ROCm. The code falls back to CPU float32, which is not practical for real use.

Python 3.14 or newer if you install with `uv`, since that is what `pyproject.toml` pins. The pip route is looser in practice, though the CUDA wheels in `requirements.txt` target `torch==2.11.0` on the cu130 index.

## Install with uv

[`uv`](https://docs.astral.sh/uv/) resolves the environment from `uv.lock`, so you get the same versions the lockfile was built with.

Install uv on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then clone, sync, and run:

```bash
git clone https://github.com/GeekLord/DigitalOcean-AMD_GPU-Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git
cd DigitalOcean-AMD_GPU-Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load
uv sync
uv run app.py
```

## Install with pip (CUDA)

```bash
pip install "pip>=26.1.2"
pip install -r requirements.txt
python app.py
```

`requirements.txt` pulls torch and torchvision from the CUDA 13 index:

```text
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.11.0
torchvision==0.26.0
transformers==5.14.1
accelerate==1.14.0
diffusers==0.39.0
peft==0.19.1
gradio==6.20.0
av==17.1.0
spaces==0.51.1
huggingface-hub==1.24.0
kernels==0.16.0
```

## AMD MI300X and ROCm on DigitalOcean

### Automated droplet setup

When you create an MI300X GPU Droplet, expand Advanced Options, check User Data, and paste the contents of [`digitalocean_startup.sh`](digitalocean_startup.sh).

The script installs build tools, opens ports 22 and 7860 in UFW, creates `/mnt/scratch/hf_cache` on the scratch NVMe, clones this repository to `/opt/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load`, builds a venv with ROCm PyTorch, and registers a `qwen-image-edit` systemd service with `Restart=always`. Progress is logged to `/var/log/qwen-startup.log`. After a few minutes the app answers on `http://<DROPLET_PUBLIC_IP>:7860/`.

### Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements-rocm.txt
```

Open the port if UFW is active:

```bash
sudo ufw allow 7860/tcp
```

Point the Hub cache at the scratch disk so adapter downloads do not fill the root volume, then start the app:

```bash
export HF_HOME=/mnt/scratch/hf_cache
python app.py
```

`requirements-rocm.txt` pins `torch==2.5.1+rocm6.2` and `torchvision==0.20.1+rocm6.2` and leaves out `kernels`. Expect the FA3 warning on this stack; the app keeps running with the default attention processor.

## Serving and network exposure

`app.py` binds `GRADIO_SERVER_NAME` (default `0.0.0.0`) on `GRADIO_SERVER_PORT` (default `7860`), queues up to 50 requests, disables SSR, exposes the app as an MCP server, and calls `launch(share=True)`.

Two things to know before you put this on a public host. The server listens on every interface and has no authentication, so anything that can reach port 7860 can run inference on your GPU. And `share=True` publishes a public `gradio.live` tunnel URL on every start, which stays reachable even if your firewall blocks 7860. If you would rather not have that, set `share=False`, put the app behind a reverse proxy or SSH tunnel, and add `auth=` to `launch()`.

## Using the app

1. Add images by dropping them on the canvas or clicking to browse. Order matters for the multi-image styles: the edit target goes in slot 1, the reference in slot 2.
2. Choose a style from the Editing Style / LoRA dropdown. The first run on a new style downloads its weights.
3. Write a prompt, or click a quick prompt chip. Example cards fill the images, prompt, and style together.
4. Adjust seed, guidance, or steps under Advanced Settings if you want to. The 4-step default is the fast path.
5. Click Edit Image. When the result lands, click it to open the compare viewer, or hit Save for a PNG.

## Links

- GitHub: [GeekLord/DigitalOcean-AMD_GPU-Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load](https://github.com/GeekLord/DigitalOcean-AMD_GPU-Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load)
- Hugging Face Space: [prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast)
- Base model: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- Transformer: [prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19](https://huggingface.co/prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19)
- License: [Apache 2.0](LICENSE)
