# %%
import nltk
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('punkt_tab')

# %%
#from google.colab import drive
#drive.mount('/content/drive')

# %%
with open("Personal_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# %%
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# How to use it:
my_text = read_text_file("Personal_data.txt")

# %%
from os import sendfile
# Text Cleaning
def clean_text(text):
  text = text .lower()
  for p in string.punctuation:
    text = text.replace(p, '')
  return text

# Keyword Extraction
def extract_keywords(text):
  vectorizer = TfidfVectorizer(max_features=10)
  X = vectorizer.fit_transform([text])
  keywords = vectorizer.get_feature_names_out()
  return keywords

# Summary Function
def summarize(text):
  sentences = nltk.sent_tokenize(text)
  if len(sentences) == 0:
    return text
  return " ".join(sentences[:3])

# Question Answering
def answer_question(text, question):
  sentences = nltk.sent_tokenize(text)
  sentences.append(question)
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(sentences)
  similarities = cosine_similarity(X[-1], X[:-1])
  index = similarities.argmax() # Corrected typo: changed 'similarity' to 'similarities'
  return sentences[index]

# Difficulty Detection
def difficulty(text):
  words = text.split()
  avg = sum(len(w) for w in words) / len(words)
  if avg < 4:
    return "Easy"
  elif avg < 6:
    return "Medium"
  else:
    return "Hard"

# %%
# Streamlit Interface
st.title("Smart AI Research Assistant")
file = st.file_uploader("Upload PDF File")
if file is not None:
    text = read_pdf(file)
    text = clean_text(text)
    st.subheader("Summary")
    st.write(summarize(text))
    st.subheader("Keywords")
    st.write(extract_keywords(text))
    st.subheader("Difficulty Level")
    st.write(difficulty(text))
    st.subheader("Ask Question")
    question = st.text_input("Enter Question")
    if question:
        answer = answer_question(text,question)
        st.write("Answer:")
        st.write(answer)
# %%







