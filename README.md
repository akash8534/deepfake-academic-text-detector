# Deepfake Academic Text Detection 🕵️‍♂️📑

> **Automated the identification of LLM-generated academic fraud by developing a hybrid CNN-RNN model trained on human/AI text pairs from ChatGPT-4, Claude, and Gemini outputs.**

This project leverages deep learning to distinguish between human-written academic texts and AI-generated content. By combining the spatial pattern recognition of Convolutional Neural Networks (CNNs) with the sequential tracking of Long Short-Term Memory (LSTM) networks, it captures both forensic stylometric patterns and temporal drift.

## 📊 Model Performance
Trained on the highly diverse **DAIGT V4 Dataset** containing over 73,000 records.

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | `95.07%` |
| **Validation AUC** | `0.9873` |
| **Training Epochs** | Early Stopping triggered at Epoch 5 |

## 🛠️ Technology Stack
| Tool / Library | Usage in Project |
| :--- | :--- |
| **Python** | Core programming language |
| **TensorFlow / Keras** | Hybrid Neural Network (Conv1D, LSTM, Embeddings) |
| **Scikit-learn** | Data partitioning (Train/Test Split) |
| **Pandas & NumPy** | Data ingestion, cleaning, and matrix manipulation |
| **Matplotlib** | Visualizing training loss and accuracy dynamics |

## 🧠 Architecture Highlights
- **Latent Space Embeddings:** Transforms raw tokenized text into dense 128-dimensional vectors.
- **Multi-Scale CNNs:** Utilizes parallel 1D Convolutions (kernel sizes 3 and 5) to extract varying n-gram stylometric features.
- **LSTM Networks:** Analyzes temporal sequence drift, capturing the deterministic predictability of LLM generation versus human variability.
- **Latent Perturbation (Dropout):** Aggressive dropout layers act as latent space perturbations to enhance model robustness and prevent overfitting.

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/akash8534/deepfake-academic-text-detector.git](https://github.com/akash8534/deepfake-academic-text-detector.git)
cd deepfake-academic-text-detector
