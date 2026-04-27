import pandas as pd
import re

INPUT_FILE = "data/finbert_scores.csv"
OUTPUT_FILE = "data/call_features.csv"

FORWARD_LOOKING_PATTERNS = [
    # Future tense
    r"will\s+\w+",
    r"(shall|would)\s+\w+",

    # Core forward-looking verbs (handles all inflections in one pattern)
    r"(expect|anticipate|believe|estimate|forecast|project|predict|intend|plan|aim|seek|target|assume|endeavor)\w*\s+(to|that|a|an|the|\w+)",

    # We + verb constructions
    r"we\s+(expect|believe|anticipate|plan|intend|aim|target|remain|feel|are)\w*",

    # Modals scoped to outcome verbs
    r"(may|might|could|should)\s+(result|lead|cause|impact|contribute|enable|create|generate|increase|decrease|improve|expand|grow|deliver|drive)\w*",

    # Planning / progress
    r"(plan|intend|aim|seek|look|work|target)\w*\s+to\w*",
    r"on\s+track\s+to",
    r"(well[- ]positioned|poised)\s+to",

    # Time horizon references
    r"(going|looking)\s+forward",
    r"(next|upcoming|coming|future)\s+(quarter|year|fiscal|month|period|cycle|years?)",
    r"(long|near|medium)[- ]term\s+(growth|outlook|target|goal|strategy|objective)",
    r"over\s+the\s+(next|coming)\s+\w+",
    r"by\s+(fiscal|year[- ]end|\d{4}|the\s+end)",

    # Confidence / momentum
    r"(remain|are|feel|stay)\s+(confident|optimistic|committed|focused|on\s+track)",
    r"(strong|solid|robust|healthy)\s+(pipeline|demand|momentum|outlook|backlog)",
    r"(continue|continuing|continued)\s+to\s+\w+",

    # Conditional
    r"(assuming|subject\s+to|contingent\s+(on|upon)|based\s+on\s+current|provided\s+that)\s+\w+",

    # Loughran-McDonald growth/execution verbs
    r"(drive|enable|scale|ramp|launch|deploy|expand|grow|deliver|monetize|transition)\w*\s+(growth|revenue|value|margin|adoption|to|into|\w+)",
    ]

SPEAKER_PATTERN = r"([A-Z][a-z]+ [A-Z][a-z]+):"

def is_forward_looking(text: str) -> bool:
    return any(re.search(p, str(text), re.IGNORECASE) for p in FORWARD_LOOKING_PATTERNS)

def build_call_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['transcript_id', 'sentence_idx'])

    # Vectorized Q&A split — avoids heavy groupby apply on 13M rows
    df['is_forward_looking'] = df['text'].apply(is_forward_looking)
    
    counts = df.groupby('transcript_id')['sentence_idx'].transform('count')
    rank   = df.groupby('transcript_id')['sentence_idx'].rank(method='first')
    df['is_qa'] = rank / counts > 0.6

    prepared = df[~df['is_qa']]
    qa        = df[df['is_qa']]

    # Core aggregates
    base = df.groupby('transcript_id').agg(
        ticker=('ticker', 'first'),
        fiscal_year=('fiscal_year', 'first'),
        fiscal_quarter=('fiscal_quarter', 'first'),
        report_date=('report_date', 'first'),
        company_id=('company_id', 'first'),
        mean_pos=('finbert_conf_pos', 'mean'),
        mean_neg=('finbert_conf_neg', 'mean'),
        mean_neu=('finbert_conf_neu', 'mean'),
        std_pos=('finbert_conf_pos', 'std'),
        std_neg=('finbert_conf_neg', 'std'),
        neg_spike_count=('finbert_conf_neg', lambda x: (x > 0.6).sum()),
        sentence_count=('sentence_idx', 'count'),
        forward_looking_count=('is_forward_looking', 'sum'),
    ).reset_index()

    base['forward_looking_ratio'] = base['forward_looking_count'] / base['sentence_count']

    prep_pos = prepared.groupby('transcript_id')['finbert_conf_pos'].mean().rename('prep_mean_pos')
    prep_neg = prepared.groupby('transcript_id')['finbert_conf_neg'].mean().rename('prep_mean_neg')
    qa_pos   = qa.groupby('transcript_id')['finbert_conf_pos'].mean().rename('qa_mean_pos')
    qa_neg   = qa.groupby('transcript_id')['finbert_conf_neg'].mean().rename('qa_mean_neg')

    base = base.join(prep_pos, on='transcript_id')
    base = base.join(prep_neg, on='transcript_id')
    base = base.join(qa_pos,   on='transcript_id')
    base = base.join(qa_neg,   on='transcript_id')

    base['qa_sentiment_drop'] = base['prep_mean_pos'] - base['qa_mean_pos']
    base['qa_neg_increase']   = base['qa_mean_neg']   - base['prep_mean_neg']

    base = base.sort_values(['ticker', 'fiscal_year', 'fiscal_quarter'])
    base['sentiment_vs_prior'] = (
        base.groupby('ticker')['mean_pos']
            .transform(lambda x: (x - x.shift(1).expanding().mean()) / (x.shift(1).expanding().std() + 1e-6))
    )

    return base

if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    features = build_call_features(df)
    features.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(features)} call-level rows to {OUTPUT_FILE}")
    print(features.head())