import streamlit as stream
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.metrics import jaccard_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time

# DOWNLOAD THE NLTK PACKAGES
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# SETTING THE PAGE CONFIGURATION
stream.set_page_config(
    page_title="VISUALIZATION",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FUNCTION TO PERFORM TEXT ANALYSIS
def analyze_text(text):
    # TOKENIZATION
    tokens = word_tokenize(text)
    
    # REMOVE PUNCTUATION
    tokens = [word for word in tokens if word.isalnum()]
    
    # REMOVING THE STEP WORDS
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # LEMMTIZATION
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    
    return tokens

# FUNCTION TO CALCULATE WORD COUNT
def word_count(text):
    tokens = analyze_text(text)
    unique_words = set(tokens)
    return len(unique_words)

# FUNCTION TO PLOT THE WORD CLOUD
def plot_word_cloud(text):
    tokens = analyze_text(text)
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(tokens))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off') 
    stream.pyplot(fig)

# FUNCTION TO REMOVE STOP WORDS
def remove_stop_words(text):
    tokens = analyze_text(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# FUNCTION TO PERFORM TOKENIZATION
def perform_tokenization(text):
    tokens = analyze_text(text)
    return tokens

# FUNCTION TO PERFORM STEMMING
def perform_stemming(text):
    stemmer = PorterStemmer()
    tokens = analyze_text(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens

# FUNCTION TO PERFROM LEMMATIZATION
def perform_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    tokens = analyze_text(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized_tokens

# FUNCTION TO CLEAN TEXT
def clean_text(text):
    tokens = word_tokenize(text)
    clean_tokens = [word.lower() for word in tokens if word.isalnum()]
    return ' '.join(clean_tokens)

# JACCARD SIMILARITY
def calculate_jaccard_similarity(text1, text2):
    tokens1 = set(analyze_text(text1))
    tokens2 = set(analyze_text(text2))
    return 1 - jaccard_distance(tokens1, tokens2)

# COSINE SIMILARITY
def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix)[0][1]

# MAIN FUNCTION
def main():
    stream.title("TEXT ANALYSIS")

    #SETTING UP THE SIDEBAR 
    stream.sidebar.header("TEXT ANALYSIS OPTIONS")
    
    # Text input
    text_option = stream.radio("SLELCT TEXT INPUT OPTION :", ("UPLOAD TEXT FILE", "WRITE A TEXT"))
    text = None 
    
    if text_option == "UPLOAD TEXT FILE":
        uploaded_file = stream.file_uploader("UPLOAD A TEXT FILE", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.read()
            stream.write("TEXT UPLOADED SUCCESSFULLY !")
    else:
            text = stream.text_area("WRITE YOUR TEXT HERE : ", key='text_area_1')
    
    if text is not None: 
        text_analysis_success = stream.success("PERFORMING TEXT ANALYSIS ...")
        time.sleep(2)
        text_analysis_success.empty()

        # WORD COUNT
        if stream.sidebar.checkbox("WORD COUNT"):
            stream.header("WORD COUNT : ")
            stream.write(word_count(text))
        
        # PLOT WORD CLOUD
        if stream.sidebar.checkbox("PLOT WORD CLOUD"):
            plot_word_cloud(text)
        
        # Remove Stop Words
        if stream.sidebar.checkbox("REMOVE STOP WORDS"):
            text_without_stopwords = remove_stop_words(text)
            stream.header("TEXT WITHOUT STOP WORDS :")
            stream.write(text_without_stopwords)
        
        # TOKENIZATION
        if stream.sidebar.checkbox("TOKENIZATION"):
            tokens = perform_tokenization(text)
            stream.header("TOKENS :")
            stream.write(tokens)
        
        # STEMMING
        if stream.sidebar.checkbox("STEMMING"):
            stemmed_tokens = perform_stemming(text)
            stream.header("STEMMED TOKENS :")
            stream.write(stemmed_tokens)
        
        # LEMMATIZATION
        if stream.sidebar.checkbox("LEMMATIZATION"):
            lemmatized_tokens = perform_lemmatization(text)
            stream.header("LEMMATIZED TOKENS :")
            stream.write(lemmatized_tokens)
        
        # CLEAN TEXT
        if stream.sidebar.checkbox("CLEAN TEXT"):
            cleaned_text = clean_text(text)
            stream.header("CLEANED TEXT :")
            stream.write(cleaned_text)
        
        # JACCARD SIMILARITY
        if stream.sidebar.checkbox("JACCARD SIMILARITY"):
            text2 = stream.text_area("ENTER SECOND TEXT FOR COMPARISON :", key='text_area_2')
            similarity = calculate_jaccard_similarity(text, text2)
            stream.header("JACCARD SIMILARITY :")
            stream.write(similarity)
            
            # PLOT JACCARD SIMILARITY
            fig, ax = plt.subplots()
            ax.bar(["Text 1", "Text 2"], [1, similarity])
            ax.set_ylabel("SIMILARITY")
            ax.set_title("JACCARD SIMILARITY COMPARISION")
            stream.pyplot(fig)
        
        # Cosine Similarity
        if stream.sidebar.checkbox("COSINE SIMILARITY"):
            text2 = stream.text_area("ENTER THE SECOND TEXT FOR COMPARISON :", key='text_area_3')
            similarity = calculate_cosine_similarity(text, text2)
            stream.header("COSINE SIMILARITY :")
            stream.write(similarity)
            
            # PLOT COSINE SIMILARITY
            fig, ax = plt.subplots()
            ax.bar(["Text 1", "Text 2"], [1, similarity])
            ax.set_ylabel("SIMILARITY")
            ax.set_title("COSINE SIMILARITY COMPARISON")
            stream.pyplot(fig)


if __name__ == '__main__':
    main()
