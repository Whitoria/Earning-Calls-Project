import os
import pandas as pd
import re
from textstat import flesch_reading_ease, gunning_fog
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "SEC/data"
INPUT_FILE = f"{DATA_DIR}/cleaned_section_text.parquet"
OUTPUT_FILE = f"{DATA_DIR}/nlp_metrics.parquet"
OUTPUT_KEYWORDS = f"{DATA_DIR}/keyword_year_table.parquet"

# NLP Signals (Expanded for better representation)
POSITIVE_CUES = ["expect", "confident", "strong", "growth", "opportunity", "momentum", "favorable", "benefit", "success", "improve"]
NEGATIVE_CUES = ["risk", "uncertainty", "may", "could", "volatility", "challenge", "decline", "adversely", "loss", "difficult"]
FORWARD_LOOKING_PATTERNS = [r"will\s+\w+", r"expect\s+to", r"projected\s+to", r"believe\s+that"]

def compute_metrics(text):
    if pd.isna(text) or not text:
        return {
            "word_count": 0, "sentence_count": 0, "avg_sentence_length": 0,
            "vocab_size": 0, "type_token_ratio": 0, "readability_flesch": 0,
            "readability_fog": 0, "positive_score": 0, "negative_score": 0,
            "polarity_score": 0, "forward_looking_score": 0, "risk_density": 0
        }
    
    words = text.split()
    sentences = re.split(r'[\.\!\?]\s', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    
    word_count = len(words)
    # Filter noisy tiny sections (e.g. empty MD&A "Not applicable")
    if word_count < 50:
         return {
            "word_count": word_count, "sentence_count": 0, "avg_sentence_length": 0,
            "vocab_size": 0, "type_token_ratio": 0, "readability_flesch": 0,
            "readability_fog": 0, "positive_score": 0, "negative_score": 0,
            "polarity_score": 0, "forward_looking_score": 0, "risk_density": 0
        }
        
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    vocab_size = len(set(words))
    ttr = vocab_size / word_count if word_count > 0 else 0
    
    # Readability (using textstat)
    try:
        flesch = flesch_reading_ease(text)
        fog = gunning_fog(text)
    except:
        flesch, fog = 0, 0
    
    # Signal counts
    text_lower = text.lower()
    pos_count = sum(text_lower.count(word) for word in POSITIVE_CUES)
    neg_count = sum(text_lower.count(word) for word in NEGATIVE_CUES)
    
    fl_count = 0
    for pattern in FORWARD_LOOKING_PATTERNS:
        fl_count += len(re.findall(pattern, text, re.I))
        
    # Bounded Polarity Score: (Pos - Neg) / (Pos + Neg) -> [-1, 1]
    polarity = (pos_count - neg_count) / (pos_count + neg_count + 1e-6)
        
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "vocab_size": vocab_size,
        "type_token_ratio": ttr,
        "readability_flesch": flesch,
        "readability_fog": fog,
        "positive_score": pos_count / word_count if word_count > 0 else 0,
        "negative_score": neg_count / word_count if word_count > 0 else 0,
        "polarity_score": polarity,
        "forward_looking_score": fl_count / word_count if word_count > 0 else 0,
        "risk_density": (text_lower.count("risk") / word_count) * 1000 if word_count > 0 else 0
    }

def extract_topics(df):
    """ Extract top TF-IDF keywords per year, picking from each ticker to ensure diversity. """
    df['year'] = pd.to_datetime(df['date']).dt.year
    
    custom_stops = list(TfidfVectorizer(stop_words='english').get_stop_words()) + [
        'apple', 'amazon', 'alphabet', 'google', 'meta', 'microsoft', 'tesla', 
        'berkshire', 'hathaway', 'nvidia', 'broadcom', 'lilly', 'eli', 'vmware', 
        'facebook', 'inc', 'company', 'corp', 'corporation', 'we', 'our', 'us', 'its', 'year', 'ended',
        'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', '2023', '2024', '2025', '2026', 'q1', 'q2', 'q3', 'q4',
        'income', 'net', 'shares', 'stock', 'cash', 'operating', 'statements', 'financial', 'quarter', 'common', 'dividend', 'interest', 'tax', 
        'results', 'consolidated', 'december', 'september', 'october', 'march', 'june', 'share', 'million', 'billion', 'approximately'
    ]
    
    # We'll extract the most prominent unique words for EACH TICKER per year, then take the union
    ticker_year_topics = []
    
    tickers = df['ticker'].unique()
    years = sorted(df['year'].unique())
    
    for year in years:
        year_df = df[df['year'] == year]
        if year_df.empty: continue
        
        # Aggregate text across all tickers for THIS year (as a reference)
        # Vectorize individual tickers to find what's unique about them
        ticker_texts = year_df.groupby('ticker')['processed_text'].apply(lambda x: ' '.join(x.dropna())).tolist()
        ticker_names = year_df.groupby('ticker').groups.keys()
        
        if len(ticker_texts) < 2: 
           # If only one ticker, use raw frequency (TTR)
           vectorizer = TfidfVectorizer(max_df=1.0, stop_words=custom_stops, max_features=20)
        else:
           # TF-IDF across tickers for THIS year to find the most unique "news"
           vectorizer = TfidfVectorizer(max_df=0.6, min_df=0.1, stop_words=custom_stops, max_features=1000)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(ticker_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            year_keys = set()
            for i in range(len(ticker_texts)):
                scores = tfidf_matrix[i].toarray()[0]
                top_idx = scores.argsort()[-4:][::-1] # Top 4 unique words for each company
                year_keys.update([feature_names[idx] for idx in top_idx if not feature_names[idx].isdigit()])
            
            ticker_year_topics.append({"year": year, "top_keywords": ", ".join(list(year_keys)[:12])})
        except:
            continue
            
    return pd.DataFrame(ticker_year_topics)

def process_nlp():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        # fallback
        fallback = f"{DATA_DIR}/cleaned_section_text.json"
        if os.path.exists(fallback):
             df = pd.read_json(fallback)
        else:
            return
    else:
        df = pd.read_parquet(INPUT_FILE)
        
    print(f"Computing NLP metrics for {len(df)} sections...")
    
    # Compute metrics
    metrics_df = df["cleaned_text_raw"].apply(compute_metrics).apply(pd.Series)
    
    # Combine metadata with metrics
    meta_cols = ["ticker", "date", "filing_type", "section_name"]
    final_df = pd.concat([df[meta_cols], metrics_df], axis=1)
    
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"NLP Metrics saved to {OUTPUT_FILE}")
    
    # Topic Extraction
    try:
        keywords_df = extract_topics(df)
        if not keywords_df.empty:
            keywords_df.to_parquet(OUTPUT_KEYWORDS, index=False)
            print(f"Keywords by year saved to {OUTPUT_KEYWORDS}")
    except Exception as e:
        print(f"TF-IDF Extraction failed: {e}")

if __name__ == "__main__":
    process_nlp()
