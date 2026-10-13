import streamlit as st
import helper
import pickle

st.set_page_config(
    page_title="Quora Duplicate Detector",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .hero {
        text-align: center;
        padding: 35px 20px;
        background: linear-gradient(135deg, #1e1b4b, #312e81, #164e63);
        border-radius: 20px;
        border: 1px solid #3730a3;
        margin-bottom: 30px;
    }

    .hero h1 {
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero p {
        color: #c7d2fe;
        font-size: 17px;
        margin-bottom: 0;
    }

    .section-title {
        color: #e2e8f0;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .question-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 20px;
    }

    .result-box {
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin-top: 25px;
    }

    .duplicate {
        background-color: #3f1d27;
        border: 1px solid #f87171;
    }

    .not-duplicate {
        background-color: #14352b;
        border: 1px solid #4ade80;
    }

    .result-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .result-description {
        font-size: 15px;
        color: #e2e8f0;
    }

    div.stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 12px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        font-size: 17px;
        font-weight: 700;
    }

    div.stButton > button:hover {
        border: 1px solid #a5b4fc;
        color: white;
        background: #4f46e5;
    }

    div[data-baseweb="textarea"] textarea {
        background-color: #0d1117;
        color: white;
        border-radius: 10px;
        border: 1px solid #30363d;
    }

    .footer {
        text-align: center;
        color: #8b949e;
        font-size: 13px;
        margin-top: 45px;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>🔍 Quora Duplicate Detector</h1>
    <p>
        Intelligent question matching powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="section-title">🧠 Compare Two Questions</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter two questions below to check whether they express the same meaning."
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 💬 Question 1")
    q1 = st.text_area(
        "Enter the first question",
        placeholder="e.g. How can I learn machine learning?",
        height=160,
        key="q1"
    )

with col2:
    st.markdown("### 💬 Question 2")
    q2 = st.text_area(
        "Enter the second question",
        placeholder="e.g. What is the best way to start learning ML?",
        height=160,
        key="q2"
    )


st.write("")

_, button_col, _ = st.columns([1, 1.5, 1])

with button_col:
    find_clicked = st.button(
        "✨  Detect Duplicate Questions",
        use_container_width=True
    )


if find_clicked:

    if not q1.strip() or not q2.strip():
        st.warning("Please enter both questions before running the prediction.")

    else:
        try:
            with st.spinner("Analyzing both questions..."):
                query = helper.query_point_creator(q1, q2)
                result = model.predict(query)[0]

            st.divider()

            st.markdown(
                '<div class="section-title" style="text-align:center;">'
                '🎯 Prediction Result</div>',
                unsafe_allow_html=True
            )

            if result == 1:
                st.markdown("""
                <div class="result-box duplicate">
                    <div class="result-title">🔴 Duplicate Questions</div>
                    <div class="result-description">
                        The model predicts that both questions express
                        the same intent.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="result-box not-duplicate">
                    <div class="result-title">🟢 Not Duplicate</div>
                    <div class="result-description">
                        The model predicts that the questions express
                        different intents.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(
                "This prediction is generated by a machine learning model "
                "and may not always be correct."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


st.markdown("""
<div class="footer">
    Built with Streamlit · Python · TF-IDF · Machine Learning
    <br><br>
    Quora Question Pairs Classification Project
</div>
""", unsafe_allow_html=True)
