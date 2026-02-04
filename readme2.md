# 📘 PerTouch Documentation

## Project Title  
> **PerTouch**: VLM-Driven Agent for Personalized and Semantic Image Retouching

---

## 🧾 Project Overview  
This project presents a personalized image retouching system that leverages the generative priors of Stable Diffusion. It is designed to perform region-aware edits while maintaining global aesthetic quality. The system integrates user preferences and supports both strong and weak instructions through an interactive agent.

---

## 📁 Project Structure

```
PerTouch/
├── train.py               # Training script
├── train.sh               # Training configuration
├── infer.py               # Inference script
├── infer.sh               # Inference configuration
├── data/                  # Data preparation and preprocessing
├── models/                # Model definitions
├── model/                 # Pretrained or fine-tuned model weights
├── agent/                 # Agent with memory and instruction handling
├── environment.yaml       # Conda environment specification
└── README.md              # Usage documentation
```

---

## ⚙️ Environment Setup

We recommend using Conda to create a clean environment:

```bash
conda env create -f environment.yaml
conda activate PerTouch
```

---

## 🚀 Training & Inference

### 📦 Dataset Preparation

Organize the training and testing data as follows:

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
│   ├── main_sam.py         # Data generation script
│   └── main_sam_norm_q.py  # Normalization script
```

To generate processed datasets:

```bash
cd data
python main_sam.py
python main_sam_norm_q.py
cd ..
```

---

### 🏋️ Model Training

Edit hyperparameters in `train.sh` if needed, then run:

```bash
./train.sh
```

*Note: The script is compatible with Weights & Biases (wandb) for logging; make sure the environment is properly configured.*

---

### 🧪 Model Inference

Inference supports multiple evaluation modes as specified in `infer.sh`:

```bash
./infer.sh
```

---

### 🧠 Agent Demo (Interactive Retouching)

An agent interface is included to support:

- Scene-aware memory
- Strong vs. weak instruction parsing
- Iterative feedback refinement

To launch the demo:

```bash
cd agent
python main.py
```

Sample prompts:

- **Weak instruction**: “Optimize the overall look.”
- **Strong instruction**: “Significantly increase the brightness of the eagle.”
- **End signal**: “Looks good now.” (Triggers memory update and image saving)

*⚠️ Note: API keys or external services have been omitted. *
