import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
    background:#0B1120;
}

.block-container{
    max-width:950px;
    padding-top:2rem;
}

.title{
    text-align:center;
    color:white;
    font-size:48px;
    font-weight:700;
}

.subtitle{
    text-align:center;
    color:#94A3B8;
    margin-bottom:30px;
}

[data-testid="stChatMessage"]{
    border-radius:15px;
    padding:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div class="title">
AI Research Assistant
</div>

<div class="subtitle">
Document Question Answering System
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DOCUMENT ----------------

with open("Personal_data.txt","r",encoding="utf-8") as f:
    text = f.read()

# ---------------- SENTENCE SPLITTER ----------------

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

# ---------------- QA FUNCTION ----------------

def answer_question(document, question):

    greetings = {
        "hi":"Hello. How can I help you today?",
        "hello":"Hello. What would you like to know?",
        "hey":"Hello. Ask me anything related to the document.",
        "good morning":"Good morning. How may I assist you?",
        "good evening":"Good evening. How may I assist you?"
    }

    q = question.lower().strip()

    if q in greetings:
        return greetings[q]

    sentences = split_sentences(document)

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(
        sentences + [question]
    )

    similarity = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    )

    best_score = similarity.max()

    best_index = similarity.argmax()

    # Irrelevant Question Detection

    if best_score < 0.15:
        return (
            "I couldn't find any relevant information "
            "about this question in my knowledge base."
        )

    return sentences[best_index]

# ---------------- CHAT HISTORY ----------------

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role":"assistant",
            "content":
            "Welcome. Ask any question related to the loaded document."
        }

    ]

# ---------------- DISPLAY CHAT ----------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- INPUT ----------------

user_input = st.chat_input(
    "Ask a question..."
)

if user_input:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":user_input
        }
    )

    response = answer_question(
        text,
        user_input
    )

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":response
        }
    )

    st.rerun()
