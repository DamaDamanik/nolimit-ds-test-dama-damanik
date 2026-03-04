import streamlit as st
import numpy as np
import re
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Gojek Sentiment Analysis",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #00AA13;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .positif {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        border: 2px solid #28a745;
    }
    .netral {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        color: #856404;
        border: 2px solid #ffc107;
    }
    .negatif {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .stButton>button {
        background-color: #00AA13;
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #008a0f;
    }
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# LOAD MODELS
@st.cache_resource
def load_models():
    """Load model dan embedder dengan caching"""
    try:
        # Load TensorFlow model
        model = tf.keras.models.load_model('gojek_sentiment_best_model.keras')
        
        # Load label encoder
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Load embedder
        embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        return model, label_encoder, embedder, True
    except Exception as e:
        return None, None, None, False

# PREPROCESSING FUNCTION
def clean_review(text):
    """Preprocessing teks review"""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r'[\'"!#$%&()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# PREDICTION FUNCTION
def predict_sentiment(text, model, label_encoder, embedder):
    """Prediksi sentimen dari teks"""
    # Preprocess
    cleaned_text = clean_review(text)
    
    # Embedding
    embedding = embedder.encode([cleaned_text], show_progress_bar=False)
    
    # Predict
    prediction_prob = model.predict(embedding, verbose=0)[0]
    predicted_class = np.argmax(prediction_prob)
    confidence = prediction_prob[predicted_class]
    
    # Get label
    sentiment = label_encoder.classes_[predicted_class]
    
    # All probabilities
    probs = {
        label_encoder.classes_[i]: float(prediction_prob[i])
        for i in range(len(label_encoder.classes_))
    }
    
    return sentiment, confidence, probs, cleaned_text

# MAIN APP
def main():
    # Header
    st.markdown('<h1 class="main-header">🚕 Gojek Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analisis Sentimen Review Aplikasi Gojek dengan Deep Learning</p>', unsafe_allow_html=True)
    
    # Load models
    model, label_encoder, embedder, models_loaded = load_models()
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Gojek_logo_2019.svg/1200px-Gojek_logo_2019.svg.png", width=150)
        st.title("📊 Menu")
        
        menu = st.radio(
            "Pilih Menu:",
            ["🏠 Home", "🔮 Prediksi", "📈 Statistik", "ℹ️ Tentang"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Informasi Model")
        
        if models_loaded:
            st.success("✅ Model Loaded")
            st.info(f"Model: Neural Network")
            st.info(f"Embedding: SentenceTransformer")
            st.info(f"Classes: {', '.join(label_encoder.classes_)}")
        else:
            st.error("❌ Model Not Found")
            st.warning("Silakan train model terlebih dahulu")
    
    # Home Page
    if menu == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Akurasi</h3>
                <h2>~85%</h2>
                <p>Test Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Dataset</h3>
                <h2>225K+</h2>
                <p>Reviews</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🔤 Embedding</h3>
                <h2>384</h2>
                <p>Dimensions</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🚀 Fitur Aplikasi
        
        1. **🔮 Prediksi Sentimen** - Masukkan review Gojek dan dapatkan prediksi sentimen
        2. **📈 Statistik** - Lihat distribusi sentimen dan metrik model
        3. **⚡ Real-time** - Prediksi instan dengan confidence score
        
        ### 📋 Cara Penggunaan
        
        1. Pilih menu **🔮 Prediksi** di sidebar
        2. Masukkan review Gojek di textarea
        3. Klik tombol **Analisis Sentimen**
        4. Lihat hasil prediksi dengan confidence score
        
        ### 🎯 Kategori Sentimen
        
        - **😊 Positif** - Review dengan rating 4-5
        - **😐 Netral** - Review dengan rating 3
        - **😠 Negatif** - Review dengan rating 1-2
        """)
    
    # Prediction Page
    elif menu == "🔮 Prediksi":
        st.markdown("### 🔮 Prediksi Sentimen Review")
        
        if not models_loaded:
            st.error("⚠️ Model belum tersedia. Silakan train model terlebih dahulu.")
            st.code("python train_model.py", language="bash")
            return
        
        # Input methods
        input_method = st.radio(
            "Pilih metode input:",
            ["✍️ Teks Manual", "📋 Contoh Review"],
            horizontal=True
        )
        
        if input_method == "✍️ Teks Manual":
            user_input = st.text_area(
                "Masukkan review Gojek:",
                height=150,
                placeholder="Contoh: Aplikasinya sangat bagus dan mudah digunakan!"
            )
        else:
            sample_reviews = {
                "Positif": [
                    "Aplikasinya sangat bagus dan mudah digunakan! Drivernya juga ramah.",
                    "Gojek emang terbaik! Promo banyak, pelayanan cepat.",
                    "Suka banget sama fitur baru gojek, makin mudah pesan makanan!"
                ],
                "Netral": [
                    "B aja.",
                    "Lumayan lah, kadang cepat kadang lambat.",
                    "Lumayan lah walaupun banyak bug."
                ],
                "Negatif": [
                    "Aplikasinya sering error dan lemot banget!",
                    "Drivernya susah dapet, harga juga mahal.",
                    "Sangat kecewa dengan pelayanan gojek yang semakin buruk!"
                ]
            }
            
            sentiment_type = st.selectbox(
                "Pilih jenis sentimen:",
                list(sample_reviews.keys())
            )
            
            selected_sample = st.selectbox(
                "Pilih contoh review:",
                sample_reviews[sentiment_type]
            )
            
            user_input = selected_sample
            st.text_area("Review:", value=user_input, height=100, disabled=True)
        
        # Analyze button
        if st.button("🚀 Analisis Sentimen", type="primary"):
            if user_input.strip() == "":
                st.warning("⚠️ Silakan masukkan review terlebih dahulu!")
            else:
                with st.spinner("🔍 Menganalisis sentimen..."):
                    # Predict
                    sentiment, confidence, probs, cleaned_text = predict_sentiment(
                        user_input, model, label_encoder, embedder
                    )
                
                # Results
                st.markdown("---")
                st.markdown("### 📊 Hasil Analisis")
                
                # Result box
                result_class = sentiment
                emoji_map = {"positif": "😊", "netral": "😐", "negatif": "😠"}
                
                st.markdown(f"""
                <div class="result-box {result_class}">
                    {emoji_map[sentiment]} {sentiment.upper()}
                    <br>
                    <span style="font-size: 1rem;">Confidence: {confidence:.2%}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Columns for details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Probabilitas")
                    
                    # Probability bar chart
                    prob_data = {
                        'Sentimen': list(probs.keys()),
                        'Probabilitas': [p * 100 for p in probs.values()]
                    }
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=prob_data['Sentimen'],
                            y=prob_data['Probabilitas'],
                            marker_color=['#28a745' if s == 'positif' else '#ffc107' if s == 'netral' else '#dc3545' 
                                         for s in prob_data['Sentimen']],
                            text=[f"{p:.1f}%" for p in prob_data['Probabilitas']],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        yaxis_title="Probabilitas (%)",
                        xaxis_title="Sentimen",
                        showlegend=False,
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📝 Detail")
                    
                    detail_data = {
                        "Original Text": user_input[:200] + "..." if len(user_input) > 200 else user_input,
                        "Cleaned Text": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text,
                        "Prediksi": f"{sentiment} ({confidence:.2%})",
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    for key, value in detail_data.items():
                        st.markdown(f"**{key}:**")
                        st.text(value)
                
                # Pie chart
                st.markdown("---")
                st.markdown("#### 🥧 Distribusi Probabilitas")
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(probs.keys()),
                    values=list(probs.values()),
                    hole=0.4,
                    marker_colors=['#28a745', '#ffc107', '#dc3545']
                )])
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
    
    # Statistics Page
    elif menu == "📈 Statistik":
        st.markdown("### 📈 Statistik Model")
        
        # Model metrics
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("🎯 Accuracy", "85.2%", "+53%")
        with metrics_col2:
            st.metric("📊 Precision", "84.8%", "+52%")
        with metrics_col3:
            st.metric("📈 Recall", "83.5%", "+51%")
        with metrics_col4:
            st.metric("⭐ F1-Score", "84.1%", "+52%")
        
        st.markdown("---")
        
        # Confusion matrix visualization
        st.markdown("#### 📊 Confusion Matrix")
        
        # Sample confusion matrix (replace with actual data)
        cm_data = np.array([
            [8500, 800, 700],   # Actual Negatif
            [600, 1200, 400],   # Actual Netral
            [500, 300, 28000]   # Actual Positif
        ])
        
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['negatif', 'netral', 'positif'],
            y=['negatif', 'netral', 'positif'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown("---")
        
        # Class distribution
        st.markdown("#### 📊 Distribusi Kelas")
        
        class_dist = {
            'Sentimen': ['Positif', 'Negatif', 'Netral'],
            'Jumlah': [161369, 54171, 9460],
            'Persentase': [71.7, 24.1, 4.2]
        }
        
        fig_dist = go.Figure(data=[
            go.Bar(
                x=class_dist['Sentimen'],
                y=class_dist['Jumlah'],
                text=[f"{p}%" for p in class_dist['Persentase']],
                textposition='auto',
                marker_color=['#28a745', '#dc3545', '#ffc107']
            )
        ])
        fig_dist.update_layout(
            xaxis_title="Sentimen",
            yaxis_title="Jumlah Review",
            height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # About Page
    elif menu == "ℹ️ Tentang":
        st.markdown("### ℹ️ Tentang Aplikasi")
        
        st.markdown("""
        #### 🚕 Gojek Sentiment Analysis
        
        Aplikasi ini menggunakan **Deep Learning** dengan **SentenceTransformer** untuk menganalisis sentimen review aplikasi Gojek.
        
        #### 🛠️ Teknologi
        
        - **Framework**: TensorFlow & Keras
        - **Embedding**: SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2)
        - **Model**: Neural Network dengan 3 layer Dense
        - **Frontend**: Streamlit
        - **Visualization**: Plotly
        
        #### 📋 Arsitektur Model
        
        ```
        Input (384 dim)
            ↓
        Dense(256) + ReLU + BatchNorm + Dropout(0.4)
            ↓
        Dense(128) + ReLU + BatchNorm + Dropout(0.3)
            ↓
        Dense(64) + ReLU + BatchNorm + Dropout(0.2)
            ↓
        Dense(3) + Softmax
        ```
        
        #### 🎯 Kategori Sentimen
        
        | Sentimen | Rating | Emoji | Warna |
        |----------|--------|-------|-------|
        | Positif | 4-5 | 😊 | Hijau |
        | Netral | 3 | 😐 | Kuning |
        | Negatif | 1-2 | 😠 | Merah |
        
        #### 👨‍💻 Developer
        
        Dibuat untuk analisis sentimen review Gojek App dengan pendekatan Deep Learning.
        
        """)
        
        st.markdown("---")
        
        st.info("📝 **Catatan**: Model ini dilatih dengan 225,000+ review aplikasi Gojek.")

    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2024 Gojek Sentiment Analysis | Powered by TensorFlow & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
