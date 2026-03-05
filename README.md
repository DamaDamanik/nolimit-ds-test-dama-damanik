# Technical Test Data Scientist - Sentiment Analysis
**Deep Learning Powered Sentiment Analysis for Gojek App Reviews**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [EDA (Exploratory Data Analysis)](#-eda)
- [Text Preprocessing](#-text-preprocessing)
- [Modeling](#-modeling)
- [Evaluation](#-evaluation)
- [Live Demo](#-live-demo)
- [Fitur Aplikasi](#-fitur-aplikasi)
- [Requirements](#-requirements)

---

## 🎯 Overview

Project ini merupakan sistem **Analisis Sentimen** untuk review aplikasi **Gojek** menggunakan **Deep Learning** dengan arsitektur Neural Network dan **SentenceTransformer** embeddings. Sistem mampu mengklasifikasikan sentimen review menjadi tiga kategori: **Positif**, **Netral**, dan **Negatif** dengan akurasi ~75%.

---

## 📊 Dataset

Dataset terdiri dari **225,002 reviews** aplikasi Gojek yang diambil dari Kaggle.

| Column | Description |
|--------|-------------|
| review | Teks ulasan customer |
| rating | Rating 1-5 |
| sentiment | Label hasil mapping rating → sentiment |

### Sentiment Mapping

- 4-5 → **positif**
- 3 → **netral**
- 1-2 → **negatif**

### Distribusi Data

| Sentimen | Jumlah | Persentase | Rating |
|----------|--------|------------|--------|
| 😊 **Positif** | 161,369 | 71.7% | 4-5 |
| 😠 **Negatif** | 54,171 | 24.1% | 1-2 |
| 😐 **Netral** | 9,460 | 4.2% | 3 |

### Preprocessing Pipeline

1. **Text Cleaning** - Remove URLs, special characters, numbers
2. **Normalization** - Lowercase, whitespace handling
3. **Filtering** - Minimum 2 words per review
4. **Embedding** - SentenceTransformer multilingual MiniLM

---

## 🔍 EDA

EDA dilakukan untuk:
- Mengecek distribusi rating & sentiment
- Menganalisis panjang review (karakter & kata)
- Memvalidasi kualitas teks
- Mendeteksi imbalance dataset

---

## 🧹 Text Preprocessing

Preprocessing dilakukan untuk membersihkan data teks:
- Remove URLs
- Remove punctuation (kecuali emoji)
- Remove numbers
- Lowercase
- Remove extra whitespace
- Filter review dengan minimal 2 kata

---

## 🧠 Modeling

Model yang digunakan:
- **Embedding**: SentenceTransformer (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Classifier**: Neural Network (Dense layers dengan BatchNorm & Dropout)

### Arsitektur Model

**Input**
- Input (384 dimensions)
- SentenceTransformer Embeddings
  
**Layer 1**
- Dense(256) + ReLU + BatchNorm
- Dropout(0.4)
  
**Layer 2**
- Dense(128) + ReLU + BatchNorm
- Dropout(0.3)
  
**Layer 3**
- Dense(64) + ReLU + BatchNorm
- Dropout(0.2)
  
**Output**
- Dense(3) + Softmax
- Output: [Negatif, Netral, Positif]


### Konfigurasi Training

- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch Size:** 512
- **Epochs:** 50 (with early stopping)
- **Class Weights:** Balanced untuk handle imbalanced data

---

## 📈 Evaluation

Model dievaluasi menggunakan dataset test dengan hasil sebagai berikut:

| Metric | Value |
|--------|-------|
| Accuracy | 75.48% |
| F1-Score (Macro) | 0.6043 |
| Precision (Macro) | 0.6261 |
| Recall (Macro) | 0.6552 |

Per-class performance:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| negatif | 0.7904 | 0.6375 | 0.7058 |
| netral | 0.1391 | 0.4887 | 0.2165 |
| positif | 0.9488 | 0.8393 | 0.8907 |

---

## 🚀 Live Demo

<!-- 🎉 **Try it now!** 🎉 -->

### 🌐 [Click here to try the live demo!]
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]
(https://nolimit-ds-test-dama-damanik-7ze9xgkeh2puicj7qogh6g.streamlit.app/)

### 📸 Screenshots

*Home*
<img width="3309" height="2339" alt="4235307f-1" src="https://github.com/user-attachments/assets/b779cd6a-eac3-4181-98ac-a90382a8345c" />

*Prediksi*

<img width="3309" height="2339" alt="dbca7627-2" src="https://github.com/user-attachments/assets/9b8c7a97-deef-478e-af70-e692a6771a6e" />
<img width="3309" height="2339" alt="dbca7627-1" src="https://github.com/user-attachments/assets/0ee75ef9-832f-4ce1-9c81-6679917e9063" />

*Statistik*

<img width="2339" height="1653" alt="57b47c74-1" src="https://github.com/user-attachments/assets/277d2327-60bd-4b62-bba2-ef28f7ad19ed" />
<img width="2339" height="1653" alt="57b47c74-2" src="https://github.com/user-attachments/assets/b820f692-4122-4241-9e45-f231f28b4917" />

---

## ✨ Fitur Aplikasi

| Fitur | Deskripsi |
|---------|-------------|
| 🔮 **Real-time Prediction** | Input teks review dan dapatkan prediksi instan |
| 📊 **Confidence Score** | Lihat probabilitas untuk setiap kelas sentimen |
| 📈 **Visual Analytics** | Grafik distribusi dan metrik model |
| 🎨 **Modern UI** | Interface yang clean dan responsif dengan Streamlit |
| ⚡ **Fast Inference** | Optimized dengan caching dan batch processing |

---

## 🛠️ Requirements

Daftar dependency tersedia di: requirements.txt

---

## 📝 How to Use

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download pre-trained model (atau train sendiri):
- gojek_sentiment_best_model.keras
- label_encoder.pkl

3. Jalankan aplikasi:

```bash
streamlit run src/app.py
```

---

## ✉️ Contact
Untuk pertanyaan atau saran, silahkan buat issue di repository.
