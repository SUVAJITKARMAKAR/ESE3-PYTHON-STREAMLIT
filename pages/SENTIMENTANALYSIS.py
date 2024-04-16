import streamlit as st
from textblob import TextBlob
import nltk


nltk.download('punkt')

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Main function
def main():
    st.title("Sentiment Analysis")
    
    # Text input
    text_option = st.radio("Select text input option:", ("Upload Text File", "Write Text"))
    text = None  # Initialize text variable
    
    if text_option == "Upload Text File":
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.read()
            st.write("Text Uploaded Successfully!")
    else:
        text = st.text_area("Write your text here:")
    
    if text is not None:  # Check if text is assigned a value
        if st.button("Perform Sentiment Analysis"):
            st.write("Performing Sentiment Analysis...")
            
            # Perform sentiment analysis
            sentiment = perform_sentiment_analysis(text)
            
            # Display sentiment
            st.write("Sentiment:", sentiment)

if __name__ == '__main__':
    main()
