import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

@st.cache_resource
def load_model_and_tokenizer(model_name="tabularisai/multilingual-sentiment-analysis"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Bản đồ nhãn
sentiment_map = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

st.title("Sentiment Analysis App")
st.write("Sử dụng model **tabularisai/multilingual-sentiment-analysis**")

user_input = st.text_area("Nhập câu (đa ngôn ngữ):", "")

if st.button("Phân tích"):
    text = user_input.strip()
    if not text:
        st.warning("Vui lòng nhập văn bản")
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
        logits = outputs.logits  # shape (1, 5)
        probs = F.softmax(logits, dim=-1)[0]  # vector độ tin cậy cho 5 lớp
        
        # Lấy nhãn dự đoán và score
        pred_id = torch.argmax(probs).item()
        pred_label = sentiment_map[pred_id]
        pred_score = probs[pred_id].item()
        
        # Hiển thị kết quả
        st.write(f"**Label dự đoán:** {pred_label}")
        st.write(f"**Độ tin cậy:** {pred_score:.4f}")

        # Nếu muốn: hiển thị xác suất cho tất cả nhãn
        st.write("**Xác suất từng lớp:**")
        for idx, prob in enumerate(probs):
            st.write(f"- {sentiment_map[idx]}: {prob:.4f}")
