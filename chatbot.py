import streamlit as st
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Téléchargement des ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')

# ----------- Prétraitement du texte -----------

@st.cache_data
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

@st.cache_data
def preprocess(text):
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)

    stop_words = set(stopwords.words('french'))
    processed_sentences = []

    for sentence in sentences:
        sentence = sentence.lower()
        words = re.findall(r'\b\w+\b', sentence)
        words = [w for w in words if w not in stop_words]
        processed_sentences.append(' '.join(words))

    return sentences, processed_sentences

def get_most_relevant_sentence(query, original_sentences, processed_sentences):
    stop_words = set(stopwords.words('french'))
    query = query.lower()
    words = re.findall(r'\b\w+\b', query)
    query_processed = ' '.join([w for w in words if w not in stop_words])

    corpus = processed_sentences + [query_processed]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)

    similarity_scores = cosine_similarity(tfidf[-1], tfidf[:-1])
    index = similarity_scores.argmax()
    return original_sentences[index]

# ----------- Interface Streamlit -----------

def main():
    st.set_page_config(page_title="Chatbot Littéraire", page_icon="🤖", layout="centered")

    st.title("🤖 Chatbot Littéraire")
    st.markdown("Posez une question sur le texte, et le chatbot vous répondra avec la phrase la plus pertinente.")

    text = load_text("pg6593.txt")
    original_sentences, processed_sentences = preprocess(text)

    user_input = st.text_input("💬 Votre question :", "")
    if st.button("Envoyer"):
        if user_input.strip():
            response = get_most_relevant_sentence(user_input, original_sentences, processed_sentences)
            st.success("📘 Réponse :")
            st.write(response)
        else:
            st.warning("Veuillez entrer une question.")

if __name__ == "__main__":
    main()
