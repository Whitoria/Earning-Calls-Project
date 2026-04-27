import pandas as pd
import torch
import scipy.special
import uuid
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sqlite3
import os
import re
from huggingface_hub import login
from dotenv import load_dotenv

api_key = os.getenv("HF_KEY")
login(token=api_key)
DB_PATH = "data/earnings_calls_v2.db"
OUTPUT_FILE = "data/finbert_scores.csv"
BATCH_SIZE = 32

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()
model.to(device)

def split_sentences(text: str, min_chars: int = 30) -> list[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) >= min_chars]

def load_transcripts(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT 
            id,
            company_id,
            ticker,
            fiscal_year,
            fiscal_quarter,
            report_date,
            transcript_text
        FROM transcripts
        WHERE transcript_text IS NOT NULL
    """, conn)
    conn.close()

    rows = []
    for _, record in df.iterrows():
        sentences = split_sentences(record["transcript_text"])
        for i, sentence in enumerate(sentences):
            rows.append({
                "transcript_id":  record["id"],
                "company_id":     record["company_id"],
                "ticker":         record["ticker"],
                "fiscal_year":    record["fiscal_year"],
                "fiscal_quarter": record["fiscal_quarter"],
                "report_date":    record["report_date"],
                "sentence_idx":   i,
                "text":           sentence
            })

    print(f"Loaded {len(rows)} sentences from {len(df)} transcripts")
    return rows

def score_batch(sentences: list[dict]) -> list[dict]:
    texts = [s["text"] for s in sentences]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # move input tensors to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    # move back to CPU for numpy/scipy
    probs = scipy.special.softmax(logits.cpu().numpy(), axis=1)

    results = []
    for i, s in enumerate(sentences):
        results.append({
            "id":                 str(uuid.uuid4()),
            "transcript_id":      s["transcript_id"],
            "company_id":         s["company_id"],
            "ticker":             s["ticker"],
            "fiscal_year":        s["fiscal_year"],
            "fiscal_quarter":     s["fiscal_quarter"],
            "report_date":        s["report_date"],
            "sentence_idx":       s["sentence_idx"],
            "text":               s["text"],
            "finbert_label":      model.config.id2label[probs[i].argmax()],
            "finbert_conf_pos":   float(probs[i][2]),
            "finbert_conf_neu":   float(probs[i][1]),
            "finbert_conf_neg":   float(probs[i][0]),
            "finbert_label_conf": float(probs[i].max()),
        })
    return results

def run():
    rows = load_transcripts(DB_PATH)

    already_done = set()
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        already_done = set(existing["transcript_id"].unique())
        print(f"Resuming — {len(already_done)} transcripts already scored")

    rows = [r for r in rows if r["transcript_id"] not in already_done]
    print(f"{len(rows)} sentences remaining to score")

    if not rows:
        print("Nothing to do.")
        return

    # increase batch size if on GPU since we have more memory to work with
    batch_size = 64 if device.type == "cuda" else 32
    print(f"Batch size: {batch_size}")

    write_header = not os.path.exists(OUTPUT_FILE)
    buffer = []

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        results = score_batch(batch)
        buffer.extend(results)

        if len(buffer) >= 1000:
            pd.DataFrame(buffer).to_csv(
                OUTPUT_FILE,
                mode="a",
                header=write_header,
                index=False
            )
            write_header = False
            buffer = []
            print(f"  scored {min(i + batch_size, len(rows))} / {len(rows)}")

    if buffer:
        pd.DataFrame(buffer).to_csv(
            OUTPUT_FILE,
            mode="a",
            header=write_header,
            index=False
        )

    print(f"Done. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run()