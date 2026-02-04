# %% [markdown]
# 
# # **AI-Based Text Question Answering Bot**

# %% [markdown]
# # Project Building
# 
# This project is built by loading a text document, preprocessing it using NLP techniques like tokenization, stopword removal, and lemmatization. The cleaned sentences are converted into TF-IDF vectors, and cosine similarity is used to find the most relevant answer to a user’s question.

# %% [markdown]
# # Tech Stack
# 
# *  Python
# 
# *  NLTK (text preprocessing)
# 
# *  Scikit-learn (TF-IDF, cosine similarity)
# 
# *  Pickle (model saving)

# %% [markdown]
# # Importing necessary Library




# %%
import streamlit as st

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
with open("C:\\Users\\asus\\Desktop\\QA txt data.txt", "r", encoding="utf-8") as f:
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
    print("QA Bot is running... (type 'exit' to quit)")

    # Continuously takes user input and returns matching answers
    while True:
        q = input("Ask a question: ")
        if q.lower() == "exit":
            print("Goodbye!")
            break
        print("Answer:", answer_question(q))

# %% [markdown]
# # Closes the opened file to free system resources

# %%
f.close()

# %% [markdown]
# # Final Outcome
# 
# ## The system allows users to ask questions in natural language and returns the most relevant answer from the given text document accurately and efficiently.

# %%




