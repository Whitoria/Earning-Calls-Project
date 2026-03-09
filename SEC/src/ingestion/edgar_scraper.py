import requests
import json
import os
import time
import pandas as pd
from bs4 import BeautifulSoup
import re

# SEC EDGAR API Configuration
# USER: Update these with your real contact info if needed
USER_AGENT = "Sample Research Application (contact@example.com)"
HEADERS = {"User-Agent": USER_AGENT}
BASE_URL_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
BASE_URL_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_num}/{doc_name}"

# Top 10 S&P 500 Tickers & CIKs
TICKER_CIK_MAP = {
    "NVDA": "0001045810",
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "AVGO": "0001730168",
    "META": "0001326801",
    "TSLA": "0001318605",
    "BRK-B": "0001067983",
    "LLY": "0000059478"
}

DATA_DIR = "SEC/data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_submissions(cik):
    url = BASE_URL_SUBMISSIONS.format(cik=cik)
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    return None

def extract_sections(html_content, form_type="10-K"):
    sections = {}
    if not html_content: return sections
    
    # Silence BeautifulSoup warnings
    import warnings
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    
    try:
        soup = BeautifulSoup(html_content, "lxml")
    except Exception:
        return sections
    text = soup.get_text()
    
    if form_type == "10-K":
        patterns = {
            "Item 1: Business": r"(?i)(ITEM\s+1\.\s+BUSINESS.*?)(?=ITEM\s+1A\.?\s+RISK|ITEM\s+1B\.?\s+UNRESOLVED|ITEM\s+2\.?\s+PROPERTIES)",
            "Item 1A: Risk Factors": r"(?i)(ITEM\s+1A\.?\s+RISK\s+FACTORS.*?)(?=ITEM\s+1B\.?\s+UNRESOLVED|ITEM\s+2\.?\s+PROPERTIES)",
            "Item 1B: Unresolved Staff Comments": r"(?i)(ITEM\s+1B\.?\s+UNRESOLVED\s+STAFF.*?)(?=ITEM\s+2\.?\s+PROPERTIES)",
            "Item 2: Properties": r"(?i)(ITEM\s+2\.\s+PROPERTIES.*?)(?=ITEM\s+3\.?\s+LEGAL)",
            "Item 3: Legal Proceedings": r"(?i)(ITEM\s+3\.\s+LEGAL\s+PROCEEDINGS.*?)(?=ITEM\s+4\.?\s+MINE)",
            "Item 4: Mine Safety Disclosures": r"(?i)(ITEM\s+4\.\s+MINE\s+SAFETY.*?)(?=PART\s+II|ITEM\s+5\.?)",
            "Item 5: Market for Common Equity": r"(?i)(ITEM\s+5\.\s+MARKET\s+FOR\s+REGISTRANT.*?)(?=ITEM\s+6\.?|ITEM\s+7\.?)",
            "Item 6: Selected Financial Data": r"(?i)(ITEM\s+6\.\s+SELECTED\s+FINANCIAL\s+DATA.*?)(?=ITEM\s+7\.?|ITEM\s+7A\.?)",
            "Item 7: MD&A": r"(?i)(ITEM\s+7\.?\s+MANAGEMENT[’\']S\s+DISCUSSION.*?)(?=ITEM\s+7A\.?\s+QUANTITATIVE|ITEM\s+8\.?\s+FINANCIAL)",
            "Item 7A: Quantitative Disclosures": r"(?i)(ITEM\s+7A\.?\s+QUANTITATIVE\s+AND\s+QUALITATIVE.*?)(?=ITEM\s+8\.?\s+FINANCIAL)",
            "Item 8: Financial Statements": r"(?i)(ITEM\s+8\.\s+FINANCIAL\s+STATEMENTS.*?)(?=ITEM\s+9\.?\s+CHANGES|ITEM\s+9A\.?\s+CONTROLS)",
            "Item 9: Changes in Accountants": r"(?i)(ITEM\s+9\.\s+CHANGES\s+IN\s+AND\s+DISAGREEMENTS.*?)(?=ITEM\s+9A\.?\s+CONTROLS)",
            "Item 9A: Controls and Procedures": r"(?i)(ITEM\s+9A\.?\s+CONTROLS\s+AND\s+PROCEDURES.*?)(?=ITEM\s+9B\.?\s+OTHER|PART\s+III|ITEM\s+10\.?)",
            "Item 9B: Other Information": r"(?i)(ITEM\s+9B\.?\s+OTHER\s+INFORMATION.*?)(?=PART\s+III|ITEM\s+10\.?)",
            "Item 10: Directors and Officers": r"(?i)(ITEM\s+10\.\s+DIRECTORS.*?)(?=ITEM\s+11\.?)",
            "Item 11: Executive Compensation": r"(?i)(ITEM\s+11\.\s+EXECUTIVE\s+COMPENSATION.*?)(?=ITEM\s+12\.?)",
            "Item 12: Security Ownership": r"(?i)(ITEM\s+12\.\s+SECURITY\s+OWNERSHIP.*?)(?=ITEM\s+13\.?)",
            "Item 13: Certain Relationships": r"(?i)(ITEM\s+13\.\s+CERTAIN\s+RELATIONSHIPS.*?)(?=ITEM\s+14\.?)",
            "Item 14: Principal Accountant Fees": r"(?i)(ITEM\s+14\.\s+PRINCIPAL\s+ACCOUNTANT.*?)(?=PART\s+IV|ITEM\s+15\.?)",
            "Item 15: Exhibits and Schedules": r"(?i)(ITEM\s+15\.\s+EXHIBITS.*?)(?=ITEM\s+16\.?|SIGNATURES)"
        }
    else: # 10-Q roughly
        patterns = {
            "Item 1 (Q): Financial Statements": r"(?i)(ITEM\s+1\.\s+FINANCIAL\s+STATEMENTS.*?)(?=ITEM\s+2\.?\s+MANAGEMENT)",
            "Item 2 (Q): MD&A": r"(?i)(ITEM\s+2\.?\s+MANAGEMENT[’\']S\s+DISCUSSION.*?)(?=ITEM\s+3\.?\s+QUANTITATIVE|PART\s+II)",
            "Item 3 (Q): Quantitative Disclosures": r"(?i)(ITEM\s+3\.\s+QUANTITATIVE\s+AND\s+QUALITATIVE.*?)(?=ITEM\s+4\.?\s+CONTROLS)",
            "Item 4 (Q): Controls and Procedures": r"(?i)(ITEM\s+4\.\s+CONTROLS\s+AND\s+PROCEDURES.*?)(?=PART\s+II|ITEM\s+1\.?\s+LEGAL)",
            "Item 1 (Q-II): Legal Proceedings": r"(?i)(PART\s+II.*?ITEM\s+1\.\s+LEGAL\s+PROCEEDINGS.*?)(?=ITEM\s+1A\.?\s+RISK)",
            "Item 1A (Q-II): Risk Factors": r"(?i)(ITEM\s+1A\.?\s+RISK\s+FACTORS.*?)(?=ITEM\s+2\.?\s+UNREGISTERED|ITEM\s+3\.?|PART\s+III)",
            "Item 2 (Q-II): Unregistered Sales": r"(?i)(ITEM\s+2\.\s+UNREGISTERED\s+SALES.*?)(?=ITEM\s+3\.?\s+DEFAULTS|ITEM\s+4\.?)",
            "Item 3 (Q-II): Defaults Upon Senior Securities": r"(?i)(ITEM\s+3\.\s+DEFAULTS\s+UPON.*?)(?=ITEM\s+4\.?\s+MINE|ITEM\s+5\.?)",
            "Item 4 (Q-II): Mine Safety Disclosures": r"(?i)(ITEM\s+4\.\s+MINE\s+SAFETY.*?)(?=ITEM\s+5\.?\s+OTHER)",
            "Item 5 (Q-II): Other Information": r"(?i)(ITEM\s+5\.\s+OTHER\s+INFORMATION.*?)(?=ITEM\s+6\.?\s+EXHIBITS)",
            "Item 6 (Q): Exhibits": r"(?i)(ITEM\s+6\.\s+EXHIBITS.*?)(?=SIGNATURES)"
        }
        
    for section_name, pattern in patterns.items():
        try:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                longest_match = max(matches, key=len).strip()
                sections[section_name] = longest_match
            else:
                sections[section_name] = ""
        except Exception:
            sections[section_name] = ""
            
    return sections

def download_filings(tickers, filing_types=["10-K", "10-Q"], limit_per_company=10):
    all_filings = []
    all_sections = []
    for ticker, cik in tickers.items():
        print(f"Processing {ticker}...")
        data = fetch_submissions(cik)
        if not data: continue
        submissions = data.get("filings", {}).get("recent", {})
        if not submissions: continue
        count = 0
        for i in range(len(submissions.get("accessionNumber", []))):
            ftype = submissions["form"][i]
            if ftype in filing_types:
                acc_num = submissions["accessionNumber"][i].replace("-", "")
                doc_name = submissions["primaryDocument"][i]
                date = submissions["filingDate"][i]
                doc_url = BASE_URL_ARCHIVE.format(cik=cik.lstrip('0'), acc_num=acc_num, doc_name=doc_name)
                try:
                    res = requests.get(doc_url, headers=HEADERS)
                    if res.status_code == 200:
                        raw_text = res.text
                        sections = extract_sections(raw_text, form_type=ftype)
                        filing_metadata = {"ticker": ticker, "cik": cik, "filing_type": ftype, "accession_number": acc_num, "date": date, "url": doc_url}
                        all_filings.append(filing_metadata) # Omit raw_text field to save space
                        for s_name, s_text in sections.items():
                            if s_text: all_sections.append({**filing_metadata, "section_name": s_name, "section_text": s_text})
                        count += 1
                        time.sleep(0.12)
                except Exception as e: print(f"Error {ticker} {ftype}: {e}")
                if count >= limit_per_company: break
    pd.DataFrame(all_filings).to_parquet(f"{DATA_DIR}/raw_filings_metadata.parquet", index=False)
    pd.DataFrame(all_sections).to_parquet(f"{DATA_DIR}/section_text.parquet", index=False)
    print("Ingestion complete.")

if __name__ == "__main__":
    download_filings(TICKER_CIK_MAP, limit_per_company=10)
