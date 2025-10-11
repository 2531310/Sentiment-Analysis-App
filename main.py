import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import plotly.express as px
import pandas as pd

# ==============================
# ⚙️ Load model & tokenizer
# ==============================
@st.cache_resource
def load_model_and_tokenizer(model_name="tabularisai/multilingual-sentiment-analysis"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# ==============================
# 🗺️ Sentiment labels
# ==============================
sentiment_map = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

# ==============================
# 🎨 Giao diện Streamlit
# ==============================
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #2E86C1;
        font-size: 2.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 25px;
    }
    .result {
        background-color: #F0F3F4;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.1em;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">💬 Sentiment Analysis App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Dựa trên mô hình <b>tabularisai/multilingual-sentiment-analysis</b></div>', unsafe_allow_html=True)

# ==============================
# ✏️ Nhập văn bản
# ==============================
user_input = st.text_area("Nhập câu (đa ngôn ngữ):", "", height=120, placeholder="Ví dụ: Sản phẩm này thật tuyệt vời!")

if st.button("🔍 Phân tích cảm xúc", use_container_width=True):
    text = user_input.strip()
    if not text:
        st.warning("⚠️ Vui lòng nhập văn bản trước khi phân tích.")
    else:
        # Tokenize & inference
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]

        # Dự đoán
        pred_id = torch.argmax(probs).item()
        pred_label = sentiment_map[pred_id]
        pred_score = probs[pred_id].item()

        # ==============================
        # 💡 Hiển thị kết quả chính
        # ==============================
        st.markdown(f'<div class="result"><b>Kết quả dự đoán:</b> {pred_label}<br>'
                    f'<b>Độ tin cậy:</b> {pred_score:.2%}</div>', unsafe_allow_html=True)

        # Thanh tiến trình biểu diễn confidence
        st.progress(float(pred_score))

        # ==============================
        # 📊 Biểu đồ xác suất
        # ==============================
        df = pd.DataFrame({
            "Sentiment": [sentiment_map[i] for i in range(len(probs))],
            "Confidence": [float(p) for p in probs]
        })
        fig = px.bar(
            df,
            x="Sentiment",
            y="Confidence",
            text=[f"{p:.2%}" for p in probs],
            color="Sentiment",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            yaxis_range=[0, 1],
            title="Mức độ tin cậy của từng nhãn",
            title_x=0.5,
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
