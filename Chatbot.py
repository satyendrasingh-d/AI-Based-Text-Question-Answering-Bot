# %%
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('punkt_tab')

# %%
#from google.colab import drive
#drive.mount('/content/drive')

# %%
with open("/content/Personal_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# %%
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# How to use it:
my_text = read_text_file("/content/Personal_data.txt")

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
def ask_from_document(document_text):
  while True:
    user_question = input("Please ask a question about the document (type 'quit' to exit): ")
    if user_question.lower() == 'quit':
      break
    if user_question:
      answer = answer_question(document_text, user_question)
      print(f"Answer: {answer}")
    else:
      print("Please enter a question.")

# Example of how to use it with your 'text' variable:
ask_from_document(text)

# %%



