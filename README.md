# 🐍 Python Tutor for Beginners (LoRA Fine-Tuned)

An AI-powered Python programming tutor built with a **LoRA fine-tuned TinyLlama model**, zero-shot topic classification, and a beginner-friendly Streamlit UI. Ask any Python question and get clear, concise answers with code examples.

---

## 🚀 Live Demo

> 👉 **[Try the live app on Hugging Face Spaces](https://huggingface.co/spaces/krisha06/Python_tutor)**

<img width="1322" height="618" alt="image" src="https://github.com/user-attachments/assets/d1624998-66e9-4355-94d1-b48acd1aa25e" />

---

## 🧠 What Makes This Different

Most chatbot projects just call an API. This project **fine-tunes an actual LLM** using **LoRA (Low-Rank Adaptation)** — a parameter-efficient technique used in production ML systems — and deploys it as an interactive tutor.

```
User Question
    ↓
[ Zero-Shot Classifier ] (facebook/bart-large-mnli)
    ↓
Python-related? ──NO──→ "Sorry, I'm a Python tutor only."
    ↓ YES
[ LoRA Fine-Tuned TinyLlama ]
  Base: TinyLlama-1.1B-Chat-v1.0
  Adapter: Custom LoRA weights (PEFT)
    ↓
[ Code Block Formatter ]
    ↓
Beginner-friendly answer with syntax-highlighted code
```

---

## ✨ Features

- 🎯 **LoRA Fine-Tuned Model** — custom adapter trained on Python Q&A data for domain-specific accuracy
- 🏷️ **Topic Guard** — zero-shot classifier blocks non-Python questions automatically
- 💻 **Code Formatting** — auto-detects and wraps code blocks with syntax highlighting
- ⚡ **Optimized Inference** — `torch.inference_mode()` + `repetition_penalty` for clean outputs
- 🧹 **Hallucination Filtering** — strips repeated questions from generated output
- 📱 **Clean UI** — collapsible answer panel, spinner feedback, beginner-friendly design

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `TinyLlama-1.1B-Chat-v1.0` | Base causal language model |
| `PEFT (LoRA)` | Parameter-efficient fine-tuning |
| `facebook/bart-large-mnli` | Zero-shot topic classification |
| `HuggingFace Transformers` | Model loading & inference |
| `Streamlit` | Web UI |
| `PyTorch` | Inference backend |
| `accelerate` | Optimized model loading |

---

## 🔬 About the LoRA Fine-Tuning

[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) is a **parameter-efficient fine-tuning** technique that:
- Freezes the base model weights
- Adds small trainable rank-decomposition matrices
- Achieves task-specific behavior with **<1% of full fine-tuning cost**

The adapter files hosted on Hugging Face:
```
adapter_config.json        # LoRA configuration
adapter_model.safetensors  # Trained adapter weights
tokenizer.json             # Tokenizer
tokenizer.model            # Tokenizer model
tokenizer_config.json      # Tokenizer config
special_tokens_map.json    # Special tokens
```

---

## 📁 Project Structure

```
python-tutor-lora/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── lora_adapter/           # LoRA adapter files (or loaded from HuggingFace)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer.model
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/python-tutor-lora.git
cd python-tutor-lora
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

> ⚠️ First run downloads TinyLlama (~2.2GB) and BART-MNLI (~1.6GB) from HuggingFace. Ensure a stable internet connection.

---

## 📦 requirements.txt

```
transformers
peft
accelerate
torch
streamlit
```

---

## 📌 Key Concepts Demonstrated

- **LoRA / PEFT Fine-Tuning** — efficient LLM adaptation for domain-specific tasks
- **Custom LLM Deployment** — hosting adapter weights on HuggingFace Model Hub
- **Zero-Shot Classification** — intent/topic detection without labeled training data
- **LLM Inference Optimization** — `inference_mode`, `repetition_penalty`, output cleaning
- **End-to-End ML App** — from fine-tuning to deployed interactive application

---
