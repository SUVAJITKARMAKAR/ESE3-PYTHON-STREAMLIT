import streamlit as stream
from textblob import TextBlob
import nltk
import time


nltk.download('punkt')

# SETTING THE PAGE CONFIGURATION
stream.set_page_config(
    page_title="SENTIMENTANALYSIS",
    page_icon=":heart",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    stream.title("SENTIMENT ANALYSIS")
    
    # Text input
    text_option = stream.radio("SELECT A TEXT INPUT TO CONTINUE", ("Upload Text File", "Write Text"))
    text = None  
    
    if text_option == "Upload Text File":
        uploaded_file = stream.file_uploader("UPLAOD A TEXT FILE", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.read()
            stream.successs("TEXT UPLOADED SUCCESSFULLY")
    else:
        text = stream.text_area("WRITE YOUR TEXT HERE : :")
    
    if text is not None:  
        if stream.button("SENSE THE TONE"):
            tone_sensation_info = stream.info("PERFORMING SENTIMENT ANALYSIS")
            time.sleep(2)
            tone_sensation_info.empty()
            
            # Perform sentiment analysis
            sentiment = perform_sentiment_analysis(text)
            
            # Display sentiment
            stream.subheader("SENTIMENT")
            stream.success(sentiment)

if __name__ == '__main__':
    main()
