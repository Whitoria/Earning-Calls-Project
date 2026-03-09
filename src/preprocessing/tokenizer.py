import json
from tqdm import tqdm
from transformers import AutoTokenizer

INPUT_FILE = "data/processed_transcripts.json"
OUTPUT_FILE = "data/tokenized_transcripts.json"

# Load FinBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

MAX_TOKENS = 450


def chunk_text(text, max_tokens=MAX_TOKENS):
    """
    Break long text into chunks of tokens.
    """
    tokens = tokenizer.encode(text)

    for i in range(0, len(tokens), max_tokens):
        yield tokens[i:i + max_tokens]


def tokenize_dataset():

    print("Loading processed transcripts...")

    with open(INPUT_FILE) as f:
        transcripts = json.load(f)

    tokenized_results = []

    for transcript in tqdm(transcripts, desc="Tokenizing transcripts"):

        ticker = transcript["ticker"]
        date = transcript["event_date"]

        text = transcript["prepared_remarks"] + " " + transcript["qa_section"]

        chunks = list(chunk_text(text))

        for chunk in chunks:

            tokenized_results.append({
                "ticker": ticker,
                "date": date,
                "input_ids": chunk
            })

    print("Saving tokenized dataset...")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(tokenized_results, f)

    print(f"Saved {len(tokenized_results)} tokenized chunks.")


if __name__ == "__main__":

    tokenize_dataset()