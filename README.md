# 🧠 LLM Evaluation Lab

A **practical & educational repository** for evaluating **Large Language Models (LLMs)** — from log-likelihood to question answering metrics.
This project explains *how evaluation actually works* under the hood for models like **Gemma**, **LLaMA**, **GPT**, and **Mistral**.

---

## 🚀 What’s Inside

- 🔢 **Log-Likelihood computation**
- ⚙️ Integration with `lm-evaluation-harness`

---

## 📁 Repository Structure

```
llm-eval-lab/
│
├── README.md
├── docs/
│   ├── log_likelihood_explained.md
│   └── evaluation_pipeline.md
│
├── src/
│   ├── eval_utils.py
│   ├── dataset_loader.py
│   └── plot_utils.py
│
├── notebooks/
│
├── examples/
│
├── requirements.txt
└── LICENSE
```

---

## 📚 Documentation Overview

| Topic                                                        | Description                                    |
| ------------------------------------------------------------ | ---------------------------------------------- |
| [log_likelihood_explained.md](docs/log_likelihood_explained.md) | Formula, step-by-step explanation, and example |
|                                                              |                                                |

---

## ⚡ Quick Start

### 1️⃣ Setup environment

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🧠 Why This Project?

> “Most people *use* evaluation metrics — few understand how they are computed.”

This repository aims to:

- Support **Persian-language** tasks and datasets evaluation

---

## 🔬 Coming Soon

- ✅ Calibration metrics (ECE, MCE)
- 📊 Streamlit dashboard for metric visualization
- 🇮🇷 Persian BoolQ / Persian SQuAD integration
- 🧩 Custom task setup for `lm-evaluation-harness`

---

## 🤝 Contributing

Contributions are welcome!
If you’d like to add new tasks, notebooks, or docs, please open a Pull Request.

---

## 📜 License

Released under the [MIT License](LICENSE).

---

## ✨ Author

OrinenebulaEngineer
