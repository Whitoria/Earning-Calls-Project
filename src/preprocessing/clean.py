import re
import sqlite3
import json
from typing import Dict, List


class TranscriptProcessor:
    """Process earnings call transcripts"""

    def __init__(self):
        self.speaker_pattern = r"([A-Z][a-z]+ [A-Z][a-z]+):"

    def process(self, raw_transcript: str, ticker: str, event_date: str) -> Dict:
        parsed = self.parse_transcript(raw_transcript)

        return {
            "ticker": ticker,
            "event_date": event_date,
            "prepared_remarks": parsed["prepared_remarks"],
            "qa_section": parsed["qa_section"],
            "speakers": parsed["speakers"],
            "management_tone": parsed["management_tone"],
            "forward_looking_statements": parsed["forward_looking_statements"]
        }

    def parse_transcript(self, transcript_text: str) -> Dict:
        sections = self._split_into_sections(transcript_text)
        speakers = self._extract_speakers(transcript_text)

        return {
            "prepared_remarks": sections.get("prepared_remarks", ""),
            "qa_section": sections.get("qa", ""),
            "speakers": speakers,
            "management_tone": self._analyze_management_tone(sections),
            "forward_looking_statements": self._extract_forward_looking(transcript_text)
        }

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        qa_start = text.lower().find("question-and-answer")

        if qa_start == -1:
            qa_start = text.lower().find("q&a")

        if qa_start != -1:
            return {
                "prepared_remarks": text[:qa_start],
                "qa": text[qa_start:]
            }

        return {"prepared_remarks": text, "qa": ""}

    def _extract_speakers(self, text: str) -> List[str]:
        speakers = re.findall(self.speaker_pattern, text)
        return list(set(speakers))

    def _analyze_management_tone(self, sections: Dict) -> float:
        return 0.5  # placeholder

    def _extract_forward_looking(self, text: str) -> List[str]:
        forward_keywords = [
            "expect", "anticipate", "believe", "forecast",
            "guidance", "outlook", "plan", "intend"
        ]

        sentences = text.split(".")
        forward_looking = []

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in forward_keywords):
                forward_looking.append(sentence.strip())

        return forward_looking[:10]


def process_database(db_path: str, output_file: str):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    processor = TranscriptProcessor()

    results = []

    cursor.execute("""
        SELECT ticker, report_date, transcript_text
        FROM transcripts
    """)

    rows = cursor.fetchall()

    for ticker, date, text in rows:

        result = processor.process(
            raw_transcript=text,
            ticker=ticker,
            event_date=date
        )

        results.append(result)

    conn.close()

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} transcripts")


if __name__ == "__main__":

    DATABASE_PATH = "data/earnings_calls.db"
    OUTPUT_FILE = "data/processed_transcripts.json"

    process_database(DATABASE_PATH, OUTPUT_FILE)