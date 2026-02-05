# %%
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Downloads tokenizer models used to split text into words/sentences
nltk.download('punkt')
# Downloads WordNet database, required for lemmatization
nltk.download('wordnet')
# Downloads common stopwords like "is", "the", "and" etc.
nltk.download('stopwords')
# Downloads additional tokenizer resources (sometimes needed to avoid errors)
nltk.download('punkt_tab')

# %%
# Reads text file content from Google Drive into memory
with open("QA TXT DATA (1).txt", "r", encoding="utf-8") as f:
    text_data = f.read()

# Cleans text using tokenization, stopword removal, and lemmatization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(tokens)

# Splits text into sentences and preprocesses each one
sentences = nltk.sent_tokenize(text_data)
processed_sentences = [preprocess(sentence) for sentence in sentences]

# Converts processed sentences into TF-IDF numerical vectors
vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(processed_sentences)

# Saves TF-IDF vectorizer and sentence vectors to disk
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(sentence_vectors, open("vectors.pkl", "wb"))

# Finds most similar sentence to user query
def answer_question(query):
    processed_query = preprocess(query)
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, sentence_vectors)
    most_similar_index = similarities.argmax()
    return sentences[most_similar_index]

if __name__ == "__main__":
    import streamlit as st

st.title("Satyendra's & Shriyansh Chatbot")

user_query = st.text_input("Ask a question:", placeholder="Write your Question Here...")

if user_query:
    answer = answer_question(user_query)
    st.subheader("Answer:")
    st.success(answer)

# %%








