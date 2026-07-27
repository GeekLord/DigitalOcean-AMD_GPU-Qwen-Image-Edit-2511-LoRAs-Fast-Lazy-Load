# **[Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast)**

Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load is an experimental, high-performance image editing and style-transfer platform built on top of the `Qwen/Qwen-Image-Edit-2509` base model and an optimized transformer architecture (`prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V4`). The application integrates Flash Attention 3 (`QwenDoubleStreamAttnProcessorFA3`) to achieve low VRAM footprints and accelerated 4-step image manipulation.

Using a **Lazy Loading** design for LoRA adapters, the system dynamically downloads and fuses task-specific adapters on demand—including Photo-to-Anime, Multiple Angles, Light Restoration, Relight, Multi-Angle Lighting, Edit Skin, Next Scene, Flat Log, and Upscaling. The web workspace is wrapped in a custom, dark-mode Gradio interface featuring client-side JavaScript gallery management, drag-and-drop file uploaders, live toast notifications, and fast prompt chips.

<img width="1714" height="1596" alt="image" src="https://github.com/user-attachments/assets/7c665c83-d5a0-492a-9ede-074382c6c46a" />

### **Key Features**

* **Lazy-Loaded Adapter Registry:** On-demand downloading and weight-fusing for specialized LoRA adapters (e.g., *Passport-Photo*, *Photo-to-Anime*, *Multiple-Angles*, *Light-Restoration*, *Relight*, *Multi-Angle-Lighting*, *Edit-Skin*, *Next-Scene*, *Flat-Log*, *Upscale-Image*, *Upscale2K*, and *Dotted-Illustration*).
* **Passport Photo Preset:** Engineered to transform low-quality, blurry, angled, or poorly lit portrait images into standardized, high-quality passport-compatible photos with uniform studio background, front-facing pose, and exact facial identity preservation.
* **Flash Attention 3 (FA3) Acceleration:** Hooks natively into the `QwenDoubleStreamAttnProcessorFA3` processor layer to accelerate cross-attention inference phases while reducing active GPU memory consumption.
* **Text-Guided Image Editing:** Offers camera angle rotations, shadow removal, uniform studio relighting, skin detail refinement, scene propagation, and 4K upscaling.
* **Polished Dark-Mode Interface:** A modern web UI with custom JavaScript event listeners, drag-and-drop file uploaders, live toast notifications, and animated status indicators.
* **Smart Aspect Ratio Snapping:** Automatically resizes uploaded images to stay within 1024px while snapping width and height to multiples of 8 to prevent shape mismatch errors during inference.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 10.jpeg
│   ├── 11.jpg
│   ├── 12.jpg
│   ├── 13.jpg
│   ├── 14.jpg
│   ├── 2.jpeg
│   ├── 4.jpg
│   ├── 5.jpg
│   ├── 6.jpg
│   ├── 7.jpg
│   ├── 8.jpg
│   ├── 9.jpg
│   ├── DI.jpg
│   └── ELS.jpg
├── qwenimage/
│   ├── __init__.py
│   ├── pipeline_qwenimage_edit_plus.py
│   ├── qwen_fa3_processor.py
│   └── transformer_qwenimage.py
├── app.py
├── LICENSE
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

### **Installation and Requirements**

To set up the Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Minimum Python **3.10** is required; Python **3.12** or **3.14** is recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for better compatibility.
* **CUDA Version:** CUDA **13.0** is recommended (`--extra-index-url` [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)), matching the environment used on the live Hugging Face demo.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on the `uv.lock` file.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/GeekLord/DigitalOcean-AMD_GPU-Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git
cd Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load

```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run app.py

```

#### **Standard PIP Installation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install pip>=26.1.2

```

**2. Install Core Dependencies**
Install the primary deep learning stack, transformer libraries, and core computing utilities listed in `requirements.txt`:

```bash
pip install -r requirements.txt

```

#### **AMD MI300X GPU / ROCm Installation (DigitalOcean AMD GPU)**

The project is fully compatible with AMD Instinct GPUs (including **MI300X 192GB VRAM** on DigitalOcean GPU Droplets) running PyTorch with ROCm.

##### **Option A: 1-Click Automated Droplet Setup (DigitalOcean User Data)**

When creating a new **MI300X GPU Droplet** in the DigitalOcean Control Panel:
1. Expand **Advanced Options** at the bottom of the Droplet creation page.
2. Check **User Data**.
3. Paste the contents of [`digitalocean_startup.sh`](file:///h:/AI/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load/digitalocean_startup.sh) into the text box.
4. Click **Create Droplet**.

The Droplet will automatically install ROCm PyTorch, clone the repository, open port `7860`, set up systemd auto-restart, and launch the server. Within ~2-3 minutes, open your browser to:
`http://<YOUR_DIGITALOCEAN_PUBLIC_IP>:7860/`

---

##### **Option B: Manual Setup**

**1. Create Virtual Environment & Upgrade PIP**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**2. Install ROCm PyTorch & Dependencies**
Install PyTorch for ROCm using `requirements-rocm.txt`:
```bash
pip install -r requirements-rocm.txt
```

**3. Allow Port 7860 in Firewall (DigitalOcean)**
If UFW firewall is active on your DigitalOcean Linux instance, open port 7860:
```bash
sudo ufw allow 7860/tcp
```

**4. Launch Application on Public IP**
`app.py` is configured to bind to `0.0.0.0:7860` by default so it is accessible via your droplet's public IP:
```bash
# Optional: Store model weights on DigitalOcean's 5 TB scratch NVMe disk
export HF_HOME=/mnt/scratch/hf_cache

python app.py
```

Once launched, access the web workspace in your browser at:
`http://<YOUR_DIGITALOCEAN_PUBLIC_IP>:7860/`

#### **Core Requirements List (`requirements.txt` - CUDA)**

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

### **Usage**

Once the web deployment initializes, open your browser to the output URL in your terminal (typically `http://127.0.0.1:7860/` for local runs, or `http://<YOUR_DIGITALOCEAN_PUBLIC_IP>:7860/` for cloud instances).

1. **Upload Asset:** Drag and drop an image into the upload drop-zone (or click the preview window to replace the image).
2. **Select Style / LoRA:** Choose your target editing task from the **Editing Style / LoRA** dropdown menu. The adapter weights will download lazily on first use.
3. **Refine Instructions:** Type your instructions inside the prompt field, or click one of the **Quick Prompts** chips to instantly fill it.
4. **Advanced Settings (Optional):** Expand the Advanced Settings panel to toggle seed randomization, scale structural guidance metrics, or adjust sampling steps.
5. **Execute:** Click the **Edit Image** button (with the thunderbolt icon). The interface loader will blur the screen while the pipeline processes, displaying the final image upon completion.


### **Links and Source**

* **GitHub Repository:** [https://github.com/GeekLord/DigitalOcean-AMD_GPU-Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git](https://github.com/GeekLord/DigitalOcean-AMD_GPU-Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast](https://huggingface.co/spaces/prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast)
* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load/blob/main/LICENSE)
