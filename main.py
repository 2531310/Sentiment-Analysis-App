import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import langid
import plotly.express as px

# ==============================
# ⚙️ Tải model & tokenizer
# ==============================
@st.cache_resource
def load_model_and_tokenizer(model_name="tabularisai/multilingual-sentiment-analysis"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# ==============================
# 🏷️ Bản đồ nhãn cảm xúc (Tiếng Việt)
# ==============================
sentiment_map = {
    0: "Rất tiêu cực",
    1: "Tiêu cực",
    2: "Trung lập",
    3: "Tích cực",
    4: "Rất tích cực"
}

# ==============================
# 🎨 Cấu hình giao diện Streamlit
# ==============================
st.set_page_config(page_title="Phân tích cảm xúc đa ngôn ngữ", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
    .title { text-align:center; color:#2E86C1; font-size:2.2em; font-weight:bold; margin-bottom:10px; }
    .sub { text-align:center; color:#666; font-size:1.1em; margin-bottom:25px; }
    .result { background-color:#F0F3F4; padding:15px; border-radius:10px; font-size:1.1em; margin-top:10px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">💬 Ứng dụng phân tích cảm xúc đa ngôn ngữ</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Sử dụng mô hình <b>tabularisai/multilingual-sentiment-analysis</b></div>', unsafe_allow_html=True)

# ==============================
# ✏️ Ô nhập văn bản
# ==============================
user_input = st.text_area(
    "Nhập câu cần phân tích (hỗ trợ đa ngôn ngữ):",
    "",
    height=120,
    placeholder="Ví dụ: Tôi rất thích sản phẩm này hoặc I love this product!"
)

# ==============================
# 🔍 Phân tích khi người dùng bấm nút
# ==============================
if st.button("🔍 Phân tích cảm xúc", use_container_width=True):
    text = user_input.strip()
    if not text:
        st.warning("⚠️ Vui lòng nhập văn bản trước khi phân tích.")
    else:
        # ==============================
        # 🌍 Phát hiện ngôn ngữ với langid
        # ==============================
        lang_code, confidence = langid.classify(text)

        lang_map = {
            "vi": "Tiếng Việt",
            "en": "Tiếng Anh",
            "fr": "Tiếng Pháp",
            "de": "Tiếng Đức",
            "es": "Tiếng Tây Ban Nha",
            "zh": "Tiếng Trung",
            "ja": "Tiếng Nhật",
            "ko": "Tiếng Hàn",
            "id": "Tiếng Indonesia",
            "th": "Tiếng Thái"
        }
        lang_name = lang_map.get(lang_code, f"Mã ngôn ngữ: {lang_code}")
        st.info(f"🌐 **Ngôn ngữ phát hiện:** {lang_name} (Độ tin cậy: {confidence:.2f})")

        # ==============================
        # 🔮 Phân tích cảm xúc
        # ==============================
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

        # Lấy nhãn dự đoán
        pred_id = torch.argmax(probs).item()
        pred_label = sentiment_map[pred_id]
        pred_score = probs[pred_id].item()

        # ==============================
        # 💡 Hiển thị kết quả dự đoán
        # ==============================
        st.markdown(f'<div class="result"><b>Kết quả dự đoán:</b> {pred_label}<br>'
                    f'<b>Mức độ tin cậy:</b> {pred_score:.2%}</div>', unsafe_allow_html=True)

        st.progress(float(pred_score))

        # ==============================
        # 📊 Biểu đồ tròn thể hiện xác suất
        # ==============================
        df = pd.DataFrame({
            "Cảm xúc": [sentiment_map[i] for i in range(len(probs))],
            "Xác suất": [float(p) for p in probs]
        })

        fig = px.pie(
            df,
            values="Xác suất",
            names="Cảm xúc",
            color="Cảm xúc",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            title="Biểu đồ thể hiện mức độ tin cậy của từng nhãn"
        )
        fig.update_traces(
            textinfo="label+percent",
            pull=[0.08 if i == pred_id else 0 for i in range(len(df))],
            textfont_size=14
        )
        fig.update_layout(title_x=0.5, height=450)
        st.plotly_chart(fig, use_container_width=True)
