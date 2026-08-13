# FLAIR

AI-powered, country-aware financial literacy research platform.

## Overview

FLAIR is a research prototype investigating how large language models can
reduce behavioral barriers to financial market participation across
different regulatory jurisdictions. Built as part of ongoing work in LLM
personalization, RAG architectures, and behavioral AI for financial
education.

Users select their country on entry; the platform then adapts its examples,
currency, exchange references, and regulator mentions accordingly — so a
user in India sees SEBI/NSE/₹ examples, a user in the US sees SEC/NYSE/$
examples, and so on.

## Supported Countries (v1)

| Country | Regulator | Exchange(s) | Currency |
|---|---|---|---|
| 🇮🇳 India | SEBI | NSE, BSE | ₹ (INR) |
| 🇺🇸 United States | SEC | NYSE, NASDAQ | $ (USD) |
| 🇬🇧 United Kingdom | FCA | LSE | £ (GBP) |
| 🇦🇪 United Arab Emirates | SCA | DFM, ADX | AED |

Adding a new country is a single new entry in the `COUNTRIES` config dict
at the top of `flair_app.py` — no other code changes required.

## Tech Stack

- Python, Streamlit
- HuggingFace Inference API (Qwen 2.5 7B Instruct)
- yfinance, Plotly
- Country-aware prompt templating (regulator / currency / exchange / ticker)

## Setup

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
HF_API_KEY = "your_token_here"
```

(or copy `.env.example` to `.env` and fill it in — both work)

Run:

```bash
streamlit run flair_app.py
```

## Status

Active development. Research ongoing.

## Contact

steaphen.ai@gmail.com
