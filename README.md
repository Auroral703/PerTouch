<div align="center">

<p align="center">
    <img src="./docs/static/image/PerTouch_logo.png" alt="PerTouch Logo" style="width: 300px;">
</p>

<h1 align="center">[AAAI 2026] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching</h1>

<p align="center">
   Zewei Chang, Zheng-Peng Duan, Jianxing Zhang, <a href="https://mmcheng.net/clguo/">Chun-Le Guo</a>, Siyu Liu<br>Hyungju Chun, Hyunhee Park, Zikun Liu, <a href="https://li-chongyi.github.io/">Chongyi Li<sup>†</sup></a>
</p>

<p align="center">
  †Corresponding Author
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2511.12998"><img src='https://img.shields.io/badge/Paper-2511.12998-red' alt='Paper PDF'></a>
   <a href='https://huggingface.co/Snowy1123/PerTouch'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
    <a href="https://auroral703.github.io/PerTouch/"><img src='https://img.shields.io/badge/Project-Page-green' alt='Project Page'></a>
</p>

</div>

## 📜 Abstract

Image retouching aims to enhance visual quality while aligning with users' personalized aesthetic preferences. To address the challenge of **balancing controllability and subjectivity**, we propose a unified diffusion-based image retouching framework called **PerTouch**.

Our method supports semantic-level image retouching while maintaining global aesthetics. Using parameter maps containing attribute values in specific semantic regions as input, PerTouch constructs an explicit **parameter-to-image** mapping for fine-grained image retouching. To improve semantic boundary perception, we introduce **semantic replacement** and **parameter perturbation mechanisms** during training.

To connect natural language instructions with visual control, we develop a **VLM-Driven agent** to handle both strong and weak user instructions. Equipped with mechanisms of **feedback-driven rethinking** and **scene-aware memory**, PerTouch better aligns with user intent and captures long-term preferences.

<div align="center">
    <img src="./docs/static/image/Overview of our PerTouch pipeline.png" alt="teaser" width="80%">
</div>

⭐ If PerTouch is helpful to your projects, please help star this repo. Thank you! 👈

---

## 📖 Table of Contents
- [🔥 News](#-news)
- [🛠️ Dependencies and Installation](#-dependencies-and-installation)
- [📊 Dataset Preparation](#-dataset-preparation)
- [🚀 Training](#-training)
- [🎨 Agent Gradio Demo](#-agent-gradio-demo)
- [📝 Citation](#-citation)
- [TODO](#-todo)

---

## 🔥 News
- **[2026-2-4]** ✅ **All TODO items completed!** Released VLM-Driven Agent Code, Gradio Demo, Dataset Preparation Code, and trained weights.
- **[2026-1-23]** 🎉 The training code of **PerTouch** diffusion backbone is released!
- **[2025-11-17]** 🚀 Our paper **PerTouch** has been released at [Arxiv](https://arxiv.org/abs/2511.12998)!
- **[2025-11-08]** 📝 Our paper **PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching** is accepted by **AAAI 2026**!

## 🛠️ Dependencies and Installation
1.  **Clone the repository and navigate to the project directory:**
    ```bash
    git clone https://github.com/Auroral703/PerTouch.git
    cd PerTouch
    ```
2.  **Create and activate the conda environment:**
    ```bash
    conda env create -f environment.yaml
    conda activate pertouch
    ```
3.  **Prepare necessary models:**

    The segmentation method used by our data structure and Agent has been updated to SAM3. Please configure it according to the [SAM3 Official Repository](https://github.com/facebookresearch/sam3) or the following operations:
    ```bash
    git clone https://github.com/facebookresearch/sam3.git
    ```
    ⚠️ Before using SAM 3, please request access to the checkpoints on the SAM 3 Hugging Face [repo](https://huggingface.co/facebook/sam3). Once accepted, You need to download the corresponding weight of SAM3 and place it in the following path.:
    ```bash
    model/sam3/sam3.pt
    ```
    Additionally, you need to download some of our PerTouch Backbone weights, please refer to the PerTouch Hugging Face [repo](https://huggingface.co/Snowy1123/PerTouch) and place the weights in the following path. The remaining weights will be downloaded automatically when used.
    ```bash
    model/ckpt
    ```
    > If you have trouble downloading, try to using the following command:
    ```bash
    export HF_ENDPOINT="https://hf-mirror.com"
    ```

## 📊 Dataset Preparation

Our data construction pipeline supports one-click construction of Parameter Maps required for training. You need to store all input images and corresponding images modified by multiple experts in the following path:

```
PerTouch/
├── data/
│   ├── train/
│   │   ├── Expert/         # Edited results from various experts
│   │   │   ├── Expert A/
│   │   │   ├── Expert B/
│   │   │   └── ...
│   │   ├── Input/          # Corresponding low-quality inputs
│   │   │   ├── Input A/
│   │   │   ├── Input B/
│   │   │   └── ...
│   ├── test/
│   ├── data_preparation.py # Data preparation and normalization
```

Then execute our data processing pipeline. If the path changes, update the configuration information in the file. 

*Note: SAM3 is required to run this pipeline.*

```bash
cd data
python data_preparation.py
```

## 🚀 Training

Edit hyperparameters in `train.sh` if needed, then run:

```bash
./train.sh
```

*Note: The script is compatible with Weights & Biases (wandb) for logging; make sure the environment is properly configured. Our experiments were conducted in FP32, and the correctness of BF16 and FP16 was not verified.*

## 🎨 Agent Gradio Demo

PerTouch provides a VLM-driven interactive image retouching demo built with Gradio, supporting personalized image retouching through natural language instructions.

1. **Ensure dependencies are installed**
   Complete the [Dependencies and Installation](#-dependencies-and-installation) section first, and download all required models.

2. **Launch the demo**
   ```bash
   cd agent
   python main.py
   ```

3. **Access the interface**
   The Gradio interface will be available at `http://127.0.0.1:7860`.


*Note: Main configuration options are located in `agent/config.py`.*

---

## 📝 Citation

If you find our work useful, please consider citing:
```bibtex
@inproceedings{chang2026pertouch,
    title     = {PerTouch: A Unified Diffusion-based Image Retouching Framework with VLM-driven Agent},
    author    = {Chang, Zewei and Duan, Zheng-Peng and Zhang, Jianxing and others},
    year      = 2026,
    booktitle = {The 40th Annual AAAI Conference on Artificial Intelligence},
    address   = {Singapore, Singapore},
}
```

## TODO

- [x] Code of Dataset Preparation.
- [x] Trained weights of our PerTouch.
- [x] VLM-Driven Agent Code and Gradio.