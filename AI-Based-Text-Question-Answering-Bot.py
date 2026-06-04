import streamlit as st
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- NLTK ----------------
nltk.download('punkt', quiet=True)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
    background-color:#0f172a;
}

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:white;
    margin-top:20px;
}

.sub-title{
    text-align:center;
    color:#94a3b8;
    font-size:18px;
    margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    "<div class='main-title'>🤖 AI Research Assistant</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Ask anything from your document knowledge base</div>",
    unsafe_allow_html=True
)

# ---------------- LOAD DOCUMENT ----------------
with open("Personal_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---------------- QA FUNCTION ----------------
def answer_question(text, question):

    greetings = {
        "hi": "Hello! 👋 How can I help you today?",
        "hello": "Hi there! 👋 How may I assist you?",
        "hey": "Hello! Ask me anything from the document.",
        "good morning": "Good Morning! ☀️",
        "good afternoon": "Good Afternoon! 😊",
        "good evening": "Good Evening! 🌙"
    }

    q = question.lower().strip()

    if q in greetings:
        return greetings[q]

    sentences = nltk.sent_tokenize(text)

    sentences.append(question)

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(sentences)

    similarities = cosine_similarity(
        X[-1],
        X[:-1]
    )

    index = similarities.argmax()

    return sentences[index]

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello 👋 I am your AI Research Assistant. Ask me anything about the document."
        }
    ]

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input(
    "Ask your question here..."
)

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    answer = answer_question(
        text,
        user_input
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()
