#  FinGPT Forecaster - Assignment 1

## **Overview**
This project explores **FinGPT-Forecaster**,  to get deep into**market insights synthesis** and **stock price forecasting**.

I used and compared **two leading models** using **Low-Rank Adaptation (LoRA)** for **financial applications**:

- **LLaMA-3.1 8B**  
- **DeepSeek-R1-Distill-Llama-8B**  

---

## Findings
|  **Metric** | **Best Choice** |
|--------------|----------------|
| **ROUGE Score (Summarization)** |  **DeepSeek-R1 8B** |
| **Mean Squared Error (MSE)** |  **LLaMA-3.1 8B** |
| **Binary Accuracy** |  **DeepSeek-R1 8B** |
| **Training Runtime** |  **DeepSeek-R1 8B** |

### **Insights**
- **DeepSeek-R1 8B** has **faster evaluation speed** and **higher binary accuracy**, making it ideal for **real-time trading & fraud detection**.
- **LLaMA-3.1 8B** did learn **faster**, make **lower evaluation loss**, and it is **better for risk management & long-term financial forecasting**.
- **DeepSeek-R1 8B** did quicker and higher accuracy than LLaMA, and it spent less time than it. I prefer **DeepSeek-R1**.

##  Setup & Execution
### **Install Dependencies**
Ensure all required libraries are installed:
```bash
pip install -r requirements.txt
```
Fine-tune both models using:
```bash
bash train.sh
```
Most important thing, have to make sure GPU availabilty, in case you can actually run the model, instead of waiting too much time
```bash
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
```
