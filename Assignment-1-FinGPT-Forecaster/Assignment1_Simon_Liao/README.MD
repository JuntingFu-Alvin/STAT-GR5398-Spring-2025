# FinGPT Forecaster - Assignment 1

## 📌 Overview
This project explores **FinGPT-Forecaster**, a state-of-the-art **financial Large Language Model (LLM)**, to assess its effectiveness in **market insights synthesis** and **stock price forecasting**.

We fine-tuned and compared two models:
- **LLaMA-3.1 8B**
- **DeepSeek-R1-Distill-Llama-8B**

using **Low-Rank Adaptation (LoRA)** to enhance their performance for financial applications.

## 🚀 Key Findings
1. **LLaMA-3.1 8B** learns **faster** and achieves **lower evaluation loss**, making it better for **risk management** and long-term financial forecasting.
2. **DeepSeek-R1 8B** has **faster evaluation speed** and **higher binary accuracy**, making it ideal for **real-time trading and fraud detection**.
3. **ROUGE Score Comparison**:
   - **DeepSeek-R1 8B** performs better in **text summarization**.
4. **Performance Metrics Compared**:
   - Training & Evaluation Loss
   - Evaluation Speed (Samples per Second)
   - ROUGE Scores (1,2,L)
   - Mean Squared Error (MSE)
   - Binary Accuracy
   - Training Runtime & Efficiency

## ⚙️ Setup & Execution
### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Configure GPU (Colab Example)
```sh
!nvidia-smi
```
Ensure your `config.json` is set correctly for **ZeroOneAdam** with `zero_optimization: stage 0`.

### 3️⃣ Prepare the Dataset
Use the provided **chat template transformation script**:
```sh
python llama3_dsr18b_datasetpreparation_dow30.py
```

### 4️⃣ Train the Models
Run:
```sh
bash train.sh
```

## 🔍 Comparative Analysis
| **Metric**          | **DeepSeek-R1 8B** | **LLaMA-3.1 8B** | **Best Choice** |
|--------------------|------------------|------------------|----------------|
| **Train Loss**     | Slower convergence | Faster convergence | **LLaMA-3.1 8B** |
| **Eval Loss**      | Higher loss       | Lower loss       | **LLaMA-3.1 8B** |
| **Evaluation Speed** | Faster | Slower | **DeepSeek-R1 8B** |
| **ROUGE Scores**   | Higher | Lower | **DeepSeek-R1 8B** |
| **MSE**            | Higher | Lower | **LLaMA-3.1 8B** |
| **Binary Accuracy** | Higher | Lower | **DeepSeek-R1 8B** |
| **Training Runtime** | Faster | Slower | **DeepSeek-R1 8B** |

## 📌 Financial Applications
- **LLaMA-3.1 8B** is better for **risk management and financial forecasting**.
- **DeepSeek-R1 8B** is preferred for **high-frequency trading and fraud detection**.

## 💪 Future Work
- **LoRA Parameter Optimization** for better efficiency.
- **Data Expansion** with additional stock indicators.
- **Feature Engineering** for improved predictive accuracy.

## 🤝 Acknowledgments
This project was conducted as part of the **Columbia Mentor Program**, focusing on FinGPT-based financial forecasting.

---

