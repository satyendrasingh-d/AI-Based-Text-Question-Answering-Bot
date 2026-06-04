import streamlit as st
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.big-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #2563eb;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}

.answer-box {
    background-color: #ecfeff;
    padding: 20px;
    border-radius: 15px;
    border-left: 8px solid #06b6d4;
    margin-top: 10px;
}

.metric-card {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD FILE ----------------
with open("Personal_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---------------- FUNCTIONS ----------------
def clean_text(text):
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, '')
    return text

def extract_keywords(text):
    vectorizer = TfidfVectorizer(max_features=10)
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

def summarize(text):
    sentences = nltk.sent_tokenize(text)
    return " ".join(sentences[:3])

def answer_question(text, question):
    sentences = nltk.sent_tokenize(text)
    sentences.append(question)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    similarity = cosine_similarity(X[-1], X[:-1])
    index = similarity.argmax()

    return sentences[index]

def difficulty(text):
    words = text.split()
    avg = sum(len(w) for w in words) / len(words)

    if avg < 4:
        return "🟢 Easy"
    elif avg < 6:
        return "🟡 Medium"
    else:
        return "🔴 Hard"

# ---------------- HEADER ----------------
st.markdown(
    "<div class='big-title'>🤖 Smart AI Research Assistant</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Ask questions, generate summaries, extract keywords and analyze documents instantly.</div>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📄 Document Insights")

    st.metric("Total Words", len(text.split()))
    st.metric("Characters", len(text))

    st.write("### Difficulty Level")
    st.success(difficulty(text))

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Sentences", len(nltk.sent_tokenize(text)))

with col2:
    st.metric("Words", len(text.split()))

with col3:
    st.metric("Keywords", len(extract_keywords(text)))

st.divider()

# ---------------- QUESTION ANSWERING ----------------
st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your question here...",
    placeholder="Example: What is Artificial Intelligence?"
)

if st.button("🔍 Get Answer", use_container_width=True):

    if question:

        with st.spinner("Searching..."):

            answer = answer_question(text, question)

            st.markdown(
                f"""
                <div class='answer-box'>
                <h4>📌 Answer</h4>
                {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------- SUMMARY ----------------
st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("📝 Document Summary")

    if st.button("Generate Summary"):
        st.info(summarize(text))

with col2:

    st.subheader("🔑 Top Keywords")

    if st.button("Extract Keywords"):
        keywords = extract_keywords(text)

        for word in keywords:
            st.badge(word)

# ---------------- DOCUMENT PREVIEW ----------------
st.divider()

with st.expander("📖 View Complete Document"):
    st.write(text)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed using Streamlit | NLP | TF-IDF | Cosine Similarity")
