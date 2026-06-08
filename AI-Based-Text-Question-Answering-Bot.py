import streamlit as st
import re
import string
import nltk
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
    margin-bottom:25px;
}

[data-testid="stChatMessage"]{
    border-radius:15px;
    padding:12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div class="title">
AI Research Assistant
</div>

<div class="subtitle">
Intelligent Document Question Answering System
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DOCUMENT ----------------

with open("Personal_data.txt","r",encoding="utf-8") as f:
    document = f.read()

# ---------------- HELPERS ----------------

def clean_text(text):

    text = text.lower()

    for p in string.punctuation:
        text = text.replace(p," ")

    text = re.sub(r"\s+"," ",text)

    return text.strip()

def split_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    sentences = [s.strip() for s in sentences if s.strip()]

    return list(dict.fromkeys(sentences))

# ---------------- QA ENGINE ----------------

def answer_question(document, question):

    greetings = {
        "hi":"Hello. How can I help you today?",
        "hello":"Hello. What would you like to know?",
        "hey":"Hello. Ask me anything related to the document.",
        "good morning":"Good morning. How may I assist you?",
        "good afternoon":"Good afternoon. How may I assist you?",
        "good evening":"Good evening. How may I assist you?"
    }

    q = question.lower().strip()

    # Greeting Responses

    if q in greetings:
        return greetings[q]

    sentences = split_sentences(document)

    # ---------------- Exact Match Rules ----------------

    keywords = [
        "name",
        "age",
        "email",
        "phone",
        "mobile",
        "city",
        "address",
        "education",
        "college",
        "university",
        "skill",
        "skills",
        "experience"
    ]

    for key in keywords:

        if key in q:

            matches = [
                s for s in sentences
                if key in s.lower()
            ]

            if matches:
                return matches[0]

    # ---------------- TF-IDF ----------------

    cleaned_sentences = [
        clean_text(s)
        for s in sentences
    ]

    cleaned_query = clean_text(question)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1,2),
        max_features=5000
    )

    doc_vectors = vectorizer.fit_transform(
        cleaned_sentences
    )

    query_vector = vectorizer.transform(
        [cleaned_query]
    )

    similarities = cosine_similarity(
        query_vector,
        doc_vectors
    )

    best_score = similarities.max()

    # ---------------- Irrelevant Question ----------------

    if best_score < 0.15:

        return (
            "I do not have information related to this question "
            "in my knowledge base."
        )

    # ---------------- Top 3 Answers ----------------

    top_indices = similarities.argsort()[0][-3:]

    answers = []

    for idx in reversed(top_indices):

        if similarities[0][idx] > 0:

            answers.append(
                sentences[idx]
            )

    return " ".join(answers)

# ---------------- CHAT SESSION ----------------

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role":"assistant",
            "content":
            "Welcome. Ask any question related to the loaded document."
        }

    ]

# ---------------- SHOW CHAT ----------------

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

    answer = answer_question(
        document,
        user_input
    )

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )

    st.rerun()
