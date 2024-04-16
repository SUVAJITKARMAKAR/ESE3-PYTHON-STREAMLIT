import streamlit as stream
import pandas as panda
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import io

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# SETTING THE PAGE CONFIGURATION
stream.set_page_config(
    page_title="TEXTANALYSIS",
    page_icon=":rocket",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to preprocess text
def preprocess_text(text):
    # TOKENIZATION
    tokens = word_tokenize(text)
    stream.sidebar.subheader("TOKENS :")
    stream.sidebar.write(tokens)
    # REMOVING THE STOPWORDS AND PUNCTUATION
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    stream.sidebar.subheader("CLEANED WORDS : ")
    stream.sidebar.write(filtered_tokens)
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    stream.sidebar.subheader("LEMMATIZED WORDS : ")
    stream.sidebar.write(lemmatized_tokens)
    return ' '.join(lemmatized_tokens)

# CALCULATION OF THE WORD COUNT
def word_count(text):
    return len(text.split())

# FUNCTION TO GENERATE THE WORD CLOUD
def generate_wordcloud(text):
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        stream.pyplot(plt.gcf())
    else:
        stream.sidebar.write("NO WORDS TO DISPLAY")

# JACCARD SIMILARITY CALCULATION
def jaccard_similarity(text1, text2):
    tokens1 = set(word_tokenize(text1.lower()))
    tokens2 = set(word_tokenize(text2.lower()))
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union != 0 else 0

# COSINE SIMILARITY CALCULATION
def cosine_similarity(text1, text2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    return sklearn_cosine_similarity(tfidf_matrix)[0,1]

# SIDBAR
stream.sidebar.title('TEXT ANALYSIS OPTIONS')

stream.header("TEXT ANALYSIS")
uploaded_file = stream.sidebar.file_uploader("UPLOAD A CSV FILE TO CONTINUE", type=['csv'])
if uploaded_file is not None:
    df = panda.read_csv(io.StringIO(uploaded_file.read().decode('utf-8')))
    row_selection = stream.sidebar.multiselect('SELECT ANY TWO ROWS:', df['Review Text'])

    if len(row_selection) == 2:
        text1 = df[df['Review Text'] == row_selection[0]]['Review Text'].iloc[0]
        text2 = df[df['Review Text'] == row_selection[1]]['Review Text'].iloc[0]

        stream.header('FIRST SENTENCE TEXT ANALYSIS')
        stream.subheader("WORD COUNT : ")
        stream.write(word_count(text1))
        cleaned_text1 = preprocess_text(text1)
        generate_wordcloud(cleaned_text1)

        stream.header('SECOND SENTENCE TEXT ANALYSIS')
        stream.subheader("WORD COUNT : ")
        stream.write(word_count(text2))
        cleaned_text2 = preprocess_text(text2)
        generate_wordcloud(cleaned_text2)

        stream.header('SIMILARITY ANALYSIS OF THE TWO SELECTED SENTENCES')
        jaccard_sim = jaccard_similarity(cleaned_text1, cleaned_text2)
        cosine_sim = cosine_similarity(cleaned_text1, cleaned_text2)
        stream.write('JACCARD SIMILARITY :', jaccard_sim)
        stream.write('COSINE SIMILARITY :', cosine_sim)

        # PLOTTING THE SIMILARITIES
        fig, ax = plt.subplots()
        similarity_labels = ['Jaccard Similarity', 'Cosine Similarity']
        similarity_scores = [jaccard_sim, cosine_sim]
        ax.bar(similarity_labels, similarity_scores, color=['blue', 'green'])
        ax.set_xlabel('SIMILARITY MEASURES')
        ax.set_ylabel('SIMILARITY SCORES')
        ax.set_title('SIMILARTITY ANALYSIS')
        stream.pyplot(fig)
