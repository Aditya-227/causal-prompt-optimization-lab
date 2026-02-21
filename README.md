# 🧪 Causal Prompt Optimization Lab

> A research-grade experimental framework for measuring the **causal impact of prompt components** on Large Language Model (LLM) performance.

---

## 🚀 Live Dashboard

🔗 **Streamlit App:**  
👉 [Open Interactive Dashboard]([YOUR_STREAMLIT_LINK_HERE](https://causal-prompt-optimization-lab-enwzey49fapzuulrvchcl3.streamlit.app/))

_(Replace with your deployed Streamlit URL later.)_

---

## 📌 Project Overview

This project implements a **factorial experimental design** to estimate the causal effects of prompt engineering components.

### 🔬 Independent Variables

| Factor | Levels |
|--------|--------|
| 🎭 Role Instruction | 0 / 1 |
| 🧠 Chain-of-Thought (CoT) | 0 / 1 |
| 📚 Few-shot Examples | 0 / 1 |
| 📏 Output Constraint | 0 / 1 |

### 🧩 Experimental Setup

- 📊 Dataset: GSM8K (12-question subset)
- 🤖 Model: Mistral-7B-Instruct (4-bit quantized)
- 🧪 Design: Within-subject fractional factorial
- 🎯 Goal: Identify causal main effects and interactions

---

## 📈 Metrics Collected

- ✅ Binary Accuracy
- 🔢 Total Tokens Used
- ⏱ Latency (seconds)
- 📊 Accuracy per Token
- ⚡ Accuracy per Second

---

## 🧠 Statistical Analysis

This project goes beyond heuristic prompt tuning and applies formal causal inference methods:

- 📌 Fixed-effects OLS regression
- 📌 Cluster-robust standard errors
- 📌 Bootstrap confidence intervals (1000 samples)
- 📌 Interaction testing
- 📌 Average Treatment Effects (ATE)

---

## 🏆 Key Findings

- 🧠 Chain-of-Thought significantly improves accuracy.
- 🎭 Role instruction reduces performance.
- 📏 Output constraint reduces token and latency cost.
- 🔍 No strong interaction effects detected.

### ✅ Optimal Prompt Strategy

> **Chain-of-Thought + Output Constraint**

This configuration maximizes performance while controlling computational cost.

---

## 📊 Dashboard Features

The interactive Streamlit dashboard includes:

- 📊 Main effects visualization (bar + line charts)
- 🔥 Interaction heatmap (CoT × Few-shot)
- 📈 Token–Accuracy frontier
- ⏱ Latency–Accuracy frontier
- 🏅 Efficiency ranking table
- 📑 Regression summary viewer
- 📌 Live ATE calculator
- 📥 CSV download functionality

---

## 🗂 Project Structure
causal-prompt-optimization-lab/
│
├── experiment.ipynb
├── causal_prompt_results.csv
├── app.py
├── requirements.txt
└── README.md
