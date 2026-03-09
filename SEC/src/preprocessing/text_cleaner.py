import os
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup NLTK
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"NLTK Download error: {e}")

DATA_DIR = "SEC/data"
INPUT_FILE = f"{DATA_DIR}/section_text.parquet"
OUTPUT_FILE = f"{DATA_DIR}/cleaned_section_text.parquet"

# Financial modal words to keep
FINANCIAL_MODALS = {"may", "might", "expect", "believe", "could", "should", "would", "will", "can"}

def clean_text(text):
    """
    Cleaner pipeline: HTML removal, whitespace cleaning, 
    boilerplate removal, and basic normalization.
    """
    if pd.isna(text) or not text:
        return ""
        
    # Remove HTML tags
    try:
        text = BeautifulSoup(str(text), "lxml").get_text()
    except:
        text = re.sub('<[^<]+?>', '', str(text))
    
    # Remove speaker boilerplate
    text = re.sub(r'^[A-Z][a-z]+ [A-Z][a-z]+: ', '', text, flags=re.MULTILINE)
    
    # Remove non-alphanumeric (keep some punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    """
    NLP step: Stopword removal (keeping modals) and NLTK lemmatization.
    """
    if pd.isna(text) or not text:
        return ""
        
    stop_words = set(stopwords.words('english')) - FINANCIAL_MODALS
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize and filter
    words = text.lower().split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return " ".join(cleaned_words)

def process_sections():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        # Fallback to json if parquet not generated yet
        fallback = f"{DATA_DIR}/section_text.json"
        if os.path.exists(fallback):
            df = pd.read_json(fallback)
        else:
            return
    else:
        df = pd.read_parquet(INPUT_FILE)
        
    print(f"Cleaning {len(df)} sections...")
    # Apply cleaning
    df["cleaned_text_raw"] = df["section_text"].apply(clean_text)
    df["processed_text"] = df["cleaned_text_raw"].apply(preprocess_text)
        
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Preprocessing complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_sections()
