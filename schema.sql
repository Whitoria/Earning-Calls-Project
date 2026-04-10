-- ===========================================
-- Database Schema for Research Pipeline
-- ===========================================

-- ------------------------------
-- Table: companies
-- ------------------------------
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE NOT NULL,
    cik TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL
);

-- ------------------------------
-- Table: filings
-- ------------------------------
CREATE TABLE IF NOT EXISTS filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    filing_type TEXT NOT NULL,
    filing_date DATE NOT NULL,
    FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE
);

-- ------------------------------
-- Table: filing_sections
-- ------------------------------
CREATE TABLE IF NOT EXISTS filing_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id INTEGER NOT NULL,
    section_name TEXT NOT NULL,
    raw_text TEXT,
    cleaned_text TEXT,
    FOREIGN KEY (filing_id) REFERENCES filings(id) ON DELETE CASCADE
);

-- ------------------------------
-- Table: earnings_calls
-- ------------------------------
CREATE TABLE IF NOT EXISTS earnings_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    call_date DATE NOT NULL,
    FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE
);

-- ------------------------------
-- Table: call_sentences
-- ------------------------------
CREATE TABLE IF NOT EXISTS call_sentences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id INTEGER NOT NULL,
    sentence_text TEXT NOT NULL,
    cleaned_text TEXT,
    FOREIGN KEY (call_id) REFERENCES earnings_calls(id) ON DELETE CASCADE
);

-- ------------------------------
-- Table: nlp_metrics
-- ------------------------------
CREATE TABLE IF NOT EXISTS nlp_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    source_type TEXT NOT NULL,  -- 'filing' or 'call'
    source_id INTEGER NOT NULL,  -- filing_sections.id OR call_sentences.id
    polarity REAL,
    risk_density REAL,
    forward_looking_score REAL,
    readability REAL,
    word_count INTEGER,
    created_at DATE DEFAULT CURRENT_DATE,
    FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE
);

-- ------------------------------
-- Table: sentiment_scores
-- ------------------------------
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    source_type TEXT NOT NULL,  -- 'filing' or 'call'
    source_id INTEGER NOT NULL,
    date DATE NOT NULL,
    label TEXT,
    confidence REAL,
    return_n REAL,
    FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE
);

-- ------------------------------
-- Table: price_history
-- ------------------------------
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL,
    date DATE NOT NULL,
    close_price REAL NOT NULL,
    FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE,
    UNIQUE (company_id, date)
);

-- ------------------------------
-- Indexes for faster joins
-- ------------------------------
CREATE INDEX IF NOT EXISTS idx_filings_company ON filings(company_id);
CREATE INDEX IF NOT EXISTS idx_filing_sections_filing ON filing_sections(filing_id);
CREATE INDEX IF NOT EXISTS idx_calls_company ON earnings_calls(company_id);
CREATE INDEX IF NOT EXISTS idx_sentences_call ON call_sentences(call_id);
CREATE INDEX IF NOT EXISTS idx_nlp_metrics_source ON nlp_metrics(source_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_scores_source ON sentiment_scores(source_id);
CREATE INDEX IF NOT EXISTS idx_price_history_company_date ON price_history(company_id, date);