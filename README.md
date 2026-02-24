## Project Overview
This project analyzes earnings call transcripts to quantify changes in management confidence and risk communication over time using NLP-based sentiment analysis.


## Background
Companies’ earning calls have a lot of important and qualitative information on their current performance, future plans/outlook, and confidence in management. While financial analysts check the numbers and statistics in order to make informed decisions and are already put out there, the language used by executives is an underutilized resource that can give anyone interested in the market an extra edge.
This project will help companies and investors, or even students interested in learning more about the stock market within their busy schedule, to deeper understand market reactions beyond just the numbers. We will be looking into how NLP can bridge a correlation between meaningful sentiment, confidence,  and uncertainty cues with short-term market volatility or price movement. 
Since analysts already did the hard work of crunching the numbers, we can try and find an edge through other means such as analyzing the speaker's words to understand how confident they are with their plan, giving us another metric to look at to consider how well they are doing. Our goal is not to replace financial analysis, but to supplement it with extra data based metrics for faster and more informed decision making in investments.


## Structure
- src/: NLP pipeline
- data/: transcripts and outputs
- dashboard/: visualization interface


## Sentiment/Language Data
- Transcripts from Company Earning Calls from:
- AlphaVantage's API (https://www.alphavantage.co/documentation/)
- Github repo (https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction)
### Management analysis from:
- EDGAR SEC 10K/10Q/8K (https://www.sec.gov/search-filings)
### Cultural context:
- News score from AlphaVantage's API

## Financial Data:
- Tiingo
- S&P 500 Stock Prices 2000 - 2026 (https://www.kaggle.com/datasets/jacksaleeby/s-and-p500-historical-data)
- Kaggle Stock prices (https://www.kaggle.com/datasets/tsaustin/us-historical-stock-prices-with-earnings-data)
- yearly/quarterly financial data from EDGAR SEC 10K/10Q/8K (https://www.sec.gov/search-filings)
- Live stock prices from Alpha Vantage API (https://www.alphavantage.co/documentation/)


## Model:
Train on a finetuned FinBERT model, NLP model for financial language.


## Status
- Transcript ingestion: in progress
- Sentence segmentation: in progress
- Sentiment baseline: in progress


## Dataset Attribution

This project uses the **MAEC: A Multimodal Aligned Earnings Conference Call Dataset for Financial Risk Prediction**.

© Original authors.  
Licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

Source: https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction  
License: https://creativecommons.org/licenses/by-sa/4.0/

Changes made:  
- Extracted and processed transcript text  
- Applied NLP-based sentiment and risk analysis  
- Generated derived sentiment metrics for visualization
