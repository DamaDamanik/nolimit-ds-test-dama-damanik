# Technical Test Data Scientist - Sentiment Analysis
Analisis Sentimen Review Aplikasi Gojek

## 🎯 Overview

Project ini merupakan sistem **Analisis Sentimen** untuk review aplikasi **Gojek** menggunakan **Deep Learning** dengan arsitektur Neural Network dan **SentenceTransformer** embeddings. Sistem mampu mengklasifikasikan sentimen review menjadi tiga kategori: **Positif**, **Netral**, dan **Negatif** dengan akurasi ~85%.

---

## 📊 Dataset

Dataset terdiri dari **225,002 reviews** aplikasi Gojek yang di-scrape dari Google Play Store.

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

## 🚀 Live Demo

<!-- 🎉 **Try it now!** 🎉 -->

### 🌐 [Click here to try the live demo!]
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]
(https://nolimit-ds-test-dama-damanik-7ze9xgkeh2puicj7qogh6g.streamlit.app/)

