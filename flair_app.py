"""
FLAIR — Financial Literacy AI Research
Built by Steaphen | Country-aware financial literacy education platform
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="FLAIR", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

# ── API ───────────────────────────────────────────────────────────────────────
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

AI_AVAILABLE = bool(GROQ_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
GROQ_MODEL = "llama-3.3-70b-versatile"  # fast + free-tier friendly on Groq


def ask_flair_ai(question: str, cfg: dict, country: str) -> str:
    """Single source of truth for calling the LLM — used by both the Markets
    panel quick-chat and the full AI Teacher tab, so behavior never drifts
    between the two."""
    if not groq_client:
        return "Add GROQ_API_KEY to .streamlit/secrets.toml to enable AI."
    try:
        _example_co = cfg['example_tickers'][0].split('.')[0]
        msg = (
            f"You are FLAIR, a friendly financial teacher for users in "
            f"{country}. Use simple words, examples in {cfg['currency_code']} "
            f"({cfg['currency_symbol']}), reference {_example_co} or other "
            f"{cfg['exchanges'][0]}-listed companies, and mention "
            f"{cfg['regulator']} ({cfg['regulator_full']}) as the relevant "
            f"regulator if rules/protection come up. "
            f"Under 150 words, encouraging. No direct buy/sell advice.\n\n"
            f"User: {question}"
        )
        res = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": msg}],
            max_tokens=300, temperature=0.7,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        return ("⏳ Rate limited — try again in a moment!"
                if "429" in err else f"Error: {err[:80]}")


def generate_followups(question: str, answer: str, cfg: dict, country: str) -> list:
    """Ask the model for 3 short natural follow-up questions based on what
    was just discussed, so the user has somewhere to go next without typing."""
    if not groq_client:
        return []
    try:
        msg = (
            f"A user in {country} just asked a financial-literacy AI: \"{question}\"\n"
            f"The AI answered: \"{answer[:300]}\"\n\n"
            f"Suggest exactly 3 short, natural follow-up questions (under 8 words each) "
            f"this user might ask next. Return ONLY the 3 questions, one per line, "
            f"no numbering, no extra text."
        )
        res = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": msg}],
            max_tokens=100, temperature=0.8,
        )
        lines = [l.strip("-•* ").strip() for l in res.choices[0].message.content.strip().split("\n")]
        return [l for l in lines if l][:3]
    except Exception:
        return []  # fail silently — follow-ups are a nice-to-have, not critical

# ── COUNTRY / MARKET CONFIG ───────────────────────────────────────────────────
# Single source of truth for jurisdiction-specific facts (regulator, exchange,
# currency, ticker format). Adding a new country later = one new dict entry.
COUNTRIES = {
    "India": {
        "code": "IN",
        "flag": "🇮🇳",
        "regulator": "SEBI",
        "regulator_full": "Securities and Exchange Board of India",
        "exchanges": ["NSE", "BSE"],
        "currency_symbol": "₹",
        "currency_code": "INR",
        "example_tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS", "ITC.NS"],
        "account_note": "You'll need a Demat + Trading account with a SEBI-registered broker, plus KYC.",
        # (ticker, company name) — searchable by either. Not exhaustive, just
        # enough well-known names for the autocomplete to feel useful.
        "search_index": [
            ("RELIANCE.NS", "Reliance Industries"), ("TCS.NS", "Tata Consultancy Services"),
            ("INFY.NS", "Infosys"), ("HDFCBANK.NS", "HDFC Bank"), ("WIPRO.NS", "Wipro"),
            ("ITC.NS", "ITC Limited"), ("ICICIBANK.NS", "ICICI Bank"), ("SBIN.NS", "State Bank of India"),
            ("BHARTIARTL.NS", "Bharti Airtel"), ("HINDUNILVR.NS", "Hindustan Unilever"),
            ("KOTAKBANK.NS", "Kotak Mahindra Bank"), ("LT.NS", "Larsen & Toubro"),
            ("AXISBANK.NS", "Axis Bank"), ("MARUTI.NS", "Maruti Suzuki"), ("TATAMOTORS.NS", "Tata Motors"),
            ("SUNPHARMA.NS", "Sun Pharmaceutical"), ("ASIANPAINT.NS", "Asian Paints"),
            ("TITAN.NS", "Titan Company"), ("ADANIENT.NS", "Adani Enterprises"), ("ZOMATO.NS", "Zomato"),
        ],
    },
    "United States": {
        "code": "US",
        "flag": "🇺🇸",
        "regulator": "SEC",
        "regulator_full": "Securities and Exchange Commission",
        "exchanges": ["NYSE", "NASDAQ"],
        "currency_symbol": "$",
        "currency_code": "USD",
        "example_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"],
        "account_note": "You'll need a brokerage account (e.g. Fidelity, Schwab) — most have no minimum deposit.",
        "search_index": [
            ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("GOOGL", "Alphabet (Google)"),
            ("AMZN", "Amazon"), ("NVDA", "NVIDIA"), ("TSLA", "Tesla"), ("META", "Meta Platforms"),
            ("NFLX", "Netflix"), ("JPM", "JPMorgan Chase"), ("V", "Visa"), ("WMT", "Walmart"),
            ("DIS", "Disney"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"), ("INTC", "Intel"),
            ("AMD", "AMD"), ("BA", "Boeing"), ("NKE", "Nike"), ("MCD", "McDonald's"), ("SBUX", "Starbucks"),
        ],
    },
    "United Kingdom": {
        "code": "UK",
        "flag": "🇬🇧",
        "regulator": "FCA",
        "regulator_full": "Financial Conduct Authority",
        "exchanges": ["LSE"],
        "currency_symbol": "£",
        "currency_code": "GBP",
        "example_tickers": ["HSBA.L", "BP.L", "AZN.L", "ULVR.L", "GSK.L", "BARC.L"],
        "account_note": "You'll need an FCA-regulated brokerage account, or a Stocks & Shares ISA provider.",
        "search_index": [
            ("HSBA.L", "HSBC Holdings"), ("BP.L", "BP"), ("AZN.L", "AstraZeneca"),
            ("ULVR.L", "Unilever"), ("GSK.L", "GSK"), ("BARC.L", "Barclays"),
            ("VOD.L", "Vodafone"), ("RIO.L", "Rio Tinto"), ("SHEL.L", "Shell"),
            ("LLOY.L", "Lloyds Banking Group"), ("DGE.L", "Diageo"), ("TSCO.L", "Tesco"),
            ("BATS.L", "British American Tobacco"), ("NG.L", "National Grid"), ("RR.L", "Rolls-Royce"),
        ],
    },
    "United Arab Emirates": {
        "code": "AE",
        "flag": "🇦🇪",
        "regulator": "SCA",
        "regulator_full": "Securities and Commodities Authority",
        "exchanges": ["DFM", "ADX"],
        "currency_symbol": "AED",
        "currency_code": "AED",
        "example_tickers": ["EMAAR.AE", "DIB.AE", "FAB.AE", "ADNOCDIST.AE"],
        "account_note": "You'll need an SCA-regulated brokerage account with a DFM/ADX-linked bank.",
        "search_index": [
            ("EMAAR.AE", "Emaar Properties"), ("DIB.AE", "Dubai Islamic Bank"),
            ("FAB.AE", "First Abu Dhabi Bank"), ("ADNOCDIST.AE", "ADNOC Distribution"),
            ("ETISALAT.AE", "e& (Etisalat)"), ("DEWA.AE", "DEWA"),
        ],
    },
}

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        'interactions': [],
        'conversations': [],
        'q_input': "",
        'country': None,          # None until user selects → forces the gate
        'ticker_input': None,      # set once country is chosen
        'active_tab': 'markets',
        'stock_result': None,     # persists stock data across reruns
        'ai_result': None,        # persists AI answer across reruns
        'followups': [],          # follow-up question suggestions after an answer
        'pending_question': None, # set by a suggestion click; auto-asked on next run
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,600;0,9..40,700;0,9..40,800;0,9..40,900&display=swap');

/* Tab-switcher buttons live inside st.container(key="tabnav_btns").
   Only THOSE buttons get squashed/hidden — everything else stays clickable. */
.st-key-tabnav_btns div[data-testid="column"] div.stButton > button {
    opacity: 0 !important;
    height: 1px !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    overflow: hidden !important;
    pointer-events: all !important;
    position: absolute !important;
}

/* strip streamlit chrome */
#root>div:first-child{background:#000!important}
.main .block-container{padding:0!important;max-width:100%!important;background:#000!important}
.stApp{background:#000!important}
header[data-testid="stHeader"]{display:none!important}
footer{display:none!important}
#MainMenu{display:none!important}
[data-testid="stSidebar"]{display:none!important}
.stDeployButton{display:none!important}
section[data-testid="stSidebar"]{display:none!important}
div[data-testid="collapsedControl"]{display:none!important}

*{font-family:'DM Sans',-apple-system,BlinkMacSystemFont,sans-serif!important;box-sizing:border-box}
body{background:#000!important;color:#e7e9ea!important}
p,span,label,div{color:#e7e9ea}
h1,h2,h3{color:#fff!important;font-weight:800!important}

::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:#000}
::-webkit-scrollbar-thumb{background:#2f3336;border-radius:2px}

/* inputs */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea{
    background:#000!important;color:#e7e9ea!important;
    border:1px solid #2f3336!important;border-radius:8px!important;
    font-size:15px!important;padding:12px 16px!important;
    transition:border-color .2s!important}
.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus{
    border-color:#1d9bf0!important;
    box-shadow:0 0 0 2px rgba(29,155,240,.1)!important;outline:none!important}
.stTextInput>label,.stTextArea>label{color:#71767b!important;font-size:13px!important}

/* selectbox */
.stSelectbox>div>div{background:#000!important;border:1px solid #2f3336!important;border-radius:8px!important;color:#e7e9ea!important}
.stSelectbox>label{color:#71767b!important;font-size:13px!important}

/* buttons — default style (ghost / outline) */
.stButton>button{
    background:#000!important;color:#e7e9ea!important;
    border:1px solid #2f3336!important;border-radius:9999px!important;
    font-weight:600!important;font-size:13px!important;
    padding:6px 16px!important;transition:background .15s,border-color .15s,transform .1s!important}
.stButton>button:hover{background:#16181c!important;border-color:#536471!important;transform:scale(1.01)!important}

/* primary buttons — blue filled */
.stButton>button[kind="primary"]{
    background:#1d9bf0!important;color:#fff!important;
    border:none!important;font-weight:700!important;font-size:15px!important;padding:10px 24px!important}
.stButton>button[kind="primary"]:hover{background:#1a8cd8!important;transform:scale(1.01)!important}

/* alerts */
.stSuccess{background:rgba(0,186,124,.08)!important;border:1px solid rgba(0,186,124,.25)!important;border-radius:12px!important;color:#00ba7c!important}
.stWarning{background:rgba(255,212,0,.06)!important;border:1px solid rgba(255,212,0,.18)!important;border-radius:12px!important}
.stError{background:rgba(244,33,46,.07)!important;border:1px solid rgba(244,33,46,.18)!important;border-radius:12px!important}

/* metrics */
[data-testid="stMetricValue"]{color:#1d9bf0!important;font-size:20px!important;font-weight:800!important}
[data-testid="stMetricLabel"]{color:#71767b!important;font-size:11px!important}

/* plotly */
.js-plotly-plot{border-radius:12px!important;overflow:hidden!important}

/* expander */
.streamlit-expanderHeader{background:#16181c!important;border:1px solid #2f3336!important;border-radius:8px!important;color:#e7e9ea!important}
.streamlit-expanderContent{background:#16181c!important;border:1px solid #2f3336!important;border-top:none!important;border-radius:0 0 8px 8px!important}

.stSpinner>div{border-top-color:#1d9bf0!important}
[data-testid="column"]{padding:0 8px!important}
hr{border-color:#2f3336!important;margin:16px 0!important}
</style>
""", unsafe_allow_html=True)

# ── COUNTRY SELECTION GATE ─────────────────────────────────────────────────────
# Financial rules (regulator, exchange, currency, ticker format) differ by
# country. We require this choice before showing any market/AI content so
# every downstream answer is jurisdiction-correct.
if st.session_state['country'] is None:
    st.markdown("""
    <style>
    .gatewrap{max-width:640px;margin:60px auto 20px;padding:0 20px;text-align:center}
    .gatelogo{font-size:44px;font-weight:900;color:#fff;letter-spacing:-2px;margin-bottom:6px}
    .gatelogo span{color:#1d9bf0}
    .gatesub{color:#71767b;font-size:15px;margin-bottom:12px}
    </style>
    <div class="gatewrap">
      <p class="gatelogo">FL<span>AIR</span></p>
      <p class="gatesub">Financial rules differ by country — pick yours so every answer is accurate for your market.</p>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        cols = st.columns(2)
        country_keys = list(COUNTRIES.keys())
        for i, cname in enumerate(country_keys):
            cinfo = COUNTRIES[cname]
            with cols[i % 2]:
                label = f"{cinfo['flag']}  {cname}  ·  {cinfo['regulator']} / {cinfo['exchanges'][0]}"
                if st.button(label, key=f"country_{cinfo['code']}", use_container_width=True):
                    st.session_state['country'] = cname
                    st.session_state['ticker_field'] = cinfo['example_tickers'][0]
                    st.rerun()

        st.markdown("""
        <p style="color:#555;font-size:12px;text-align:center;margin-top:20px">
          You can change this later from the top of the page.
        </p>
        """, unsafe_allow_html=True)

    st.stop()  # nothing below renders until a country is chosen

country = st.session_state['country']
cfg = COUNTRIES[country]

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hw{position:relative;width:100%;min-height:230px;background:#000;overflow:hidden;
    display:flex;align-items:center;justify-content:center;border-bottom:1px solid #2f3336}
.bl{position:absolute;border-radius:50%;filter:blur(90px);pointer-events:none}
.b1{width:580px;height:580px;background:radial-gradient(circle,#1d9bf0 0%,transparent 70%);
    top:-260px;left:-70px;opacity:.13;animation:d1 18s ease-in-out infinite}
.b2{width:460px;height:460px;background:radial-gradient(circle,#fff 0%,transparent 70%);
    top:-120px;right:-60px;opacity:.04;animation:d2 22s ease-in-out infinite}
.b3{width:360px;height:360px;background:radial-gradient(circle,#1d9bf0 0%,transparent 70%);
    bottom:-150px;left:45%;opacity:.08;animation:d3 15s ease-in-out infinite}
.b4{width:240px;height:240px;background:radial-gradient(circle,#555 0%,transparent 70%);
    top:50px;left:25%;opacity:.05;animation:d4 20s ease-in-out infinite}
@keyframes d1{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(100px,-50px) scale(1.2)}66%{transform:translate(-60px,80px) scale(.9)}}
@keyframes d2{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(-80px,40px) scale(1.1)}66%{transform:translate(60px,-70px) scale(1.3)}}
@keyframes d3{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(-90px,-40px) scale(1.4)}}
@keyframes d4{0%,100%{transform:translate(0,0) scale(1)}40%{transform:translate(70px,50px) scale(1.2)}80%{transform:translate(-40px,-30px) scale(.8)}}
.hw::after{content:'';position:absolute;inset:0;opacity:.2;pointer-events:none;
    background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.03'/%3E%3C/svg%3E")}
.hi{position:relative;z-index:10;text-align:center;padding:40px 24px}
.hl{font-size:52px;font-weight:900;color:#fff;letter-spacing:-2px;margin:0 0 6px;line-height:1}
.hl span{color:#1d9bf0}
.hd{font-size:14px;color:#555;margin:0 0 16px;font-weight:400}
.hbs{display:flex;gap:8px;justify-content:center;flex-wrap:wrap}
.hb{display:inline-flex;align-items:center;gap:5px;font-size:12px;border-radius:9999px;padding:4px 14px}
.hbg{color:#00ba7c;background:rgba(0,186,124,.08);border:1px solid rgba(0,186,124,.2)}
.hbb{color:#1d9bf0;background:rgba(29,155,240,.08);border:1px solid rgba(29,155,240,.2)}
.hbd{color:#71767b;background:rgba(113,118,123,.08);border:1px solid rgba(113,118,123,.2)}
.dot{width:6px;height:6px;border-radius:50%;background:#00ba7c;display:inline-block;animation:pd 2s infinite}
@keyframes pd{0%,100%{opacity:1}50%{opacity:.3}}
</style>
""" + f"""
<div class="hw">
  <div class="bl b1"></div><div class="bl b2"></div>
  <div class="bl b3"></div><div class="bl b4"></div>
  <div class="hi">
    <p class="hl">FL<span>AIR</span></p>
    <p class="hd">Studying How Personalized AI Explanations Affect Financial Comprehension</p>
    <div class="hbs">
      <span class="hb hbg"><span class="dot"></span> AI Online</span>
      <span class="hb hbb">🔬 Research Platform</span>
      <span class="hb hbd">{cfg['flag']} {country} · {cfg['regulator']} · {'/'.join(cfg['exchanges'])}</span>
      <span class="hb hbd">RAG · LLMs · Behavioral AI</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# small "change country" control, right under the hero
_cc1, _cc2, _cc3 = st.columns([3, 1, 3])
with _cc2:
    if st.button("🌍 Change country", key="change_country_btn", use_container_width=True):
        st.session_state['country'] = None
        st.session_state['stock_result'] = None
        if 'ticker_field' in st.session_state:
            del st.session_state['ticker_field']
        st.rerun()

# ── NAV TABS — single real nav, no fake duplicate ─────────────────────────────
q_count = len(st.session_state['interactions'])
tab_map = {
    'markets':  '📊 Markets',
    'ai':       '🤖 AI Teacher',
    'research': '🔬 Research',
    'global':   '🌍 Global',
}

st.markdown(f"""
<style>
/* Status strip above the real nav buttons */
.navstrip{{display:flex;justify-content:flex-end;gap:8px;padding:6px 24px 0;
          background:#000}}
.xbadge{{font-size:12px;color:#71767b;background:#16181c;border:1px solid #2f3336;
         border-radius:9999px;padding:4px 12px}}

/* Style the REAL Streamlit tab buttons to look like a nav bar */
.st-key-tabnav_btns{{border-bottom:1px solid #2f3336;background:rgba(0,0,0,.94);
                     backdrop-filter:blur(12px);position:sticky;top:0;z-index:100;
                     padding:0 16px}}
.st-key-tabnav_btns div[data-testid="column"] div.stButton > button{{
    background:transparent!important;border:none!important;border-radius:0!important;
    border-bottom:2px solid transparent!important;color:#71767b!important;
    font-size:15px!important;font-weight:500!important;padding:14px 8px!important;
    width:100%!important}}
.st-key-tabnav_btns div[data-testid="column"] div.stButton > button:hover{{
    color:#e7e9ea!important;background:rgba(255,255,255,.03)!important}}
.st-key-tabnav_btns div[data-testid="column"] div.stButton > button[kind="primary"]{{
    color:#e7e9ea!important;font-weight:700!important;
    border-bottom:2px solid #1d9bf0!important}}
</style>
<div class="navstrip">
  <span class="xbadge">💬 {q_count} questions</span>
  <span class="xbadge">{'🟢 AI Online' if AI_AVAILABLE else '🔴 AI Offline'}</span>
</div>
""", unsafe_allow_html=True)

with st.container(key="tabnav_btns"):
    tab_cols = st.columns(len(tab_map))
    for i, (key, label) in enumerate(tab_map.items()):
        with tab_cols[i]:
            if st.button(label, key=f"tab_btn_{key}",
                         type="primary" if st.session_state['active_tab'] == key else "secondary",
                         use_container_width=True):
                st.session_state['active_tab'] = key
                st.rerun()

active = st.session_state['active_tab']
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── SHARED CARD CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
.card{background:#000;border:1px solid #2f3336;border-radius:16px;padding:20px;margin-bottom:12px}
.card-title{font-size:18px;font-weight:800;color:#fff;margin:0 0 14px;letter-spacing:-.4px}
.post{border:1px solid #2f3336;border-radius:16px;padding:18px;margin-bottom:10px;
      background:#000;transition:background .12s}
.post:hover{background:#070707}
.ptag{display:inline-block;background:#16181c;border:1px solid #2f3336;border-radius:9999px;
      padding:3px 10px;font-size:11px;color:#1d9bf0;margin:2px 2px 0 0}
.chip{display:inline-block;background:#16181c;border:1px solid #2f3336;border-radius:9999px;
      padding:3px 11px;font-size:12px;color:#1d9bf0;margin:2px 2px 2px 0}
.ai-bubble{background:#16181c;border:1px solid #2f3336;border-radius:12px;padding:14px;margin-top:10px}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB: MARKETS
# ════════════════════════════════════════════════════════════════════════════
if active == 'markets':
    left, center, right = st.columns([1, 1.5, 1.1], gap="large")

    # ── LEFT: Stock Explorer ──────────────────────────────────────────────
    with left:
        st.markdown('<div class="card"><p class="card-title">🔍 Stock Explorer</p>'
                     '<p style="color:#71767b;font-size:12px;margin:-8px 0 0">'
                     'Search by ticker symbol (e.g. TSLA, not "Tesla")</p></div>',
                    unsafe_allow_html=True)

        # A widget's session_state key can only be written BEFORE that widget is
        # instantiated in the script (Streamlit raises StreamlitAPIException
        # otherwise). So buttons below (which render after text_input) cannot
        # write to "ticker_field" directly. Instead they write to a separate
        # "ticker_field_pending" flag; we apply it here, right before the
        # text_input is created, then clear it.
        if 'ticker_field' not in st.session_state:
            st.session_state['ticker_field'] = cfg['example_tickers'][0]
        if st.session_state.get('ticker_field_pending'):
            st.session_state['ticker_field'] = st.session_state.pop('ticker_field_pending')

        st.text_input("Ticker", label_visibility="collapsed",
                      placeholder=f"Search here — e.g. {cfg['example_tickers'][0]} or company name",
                      key="ticker_field")

        # ── Live search suggestions ──────────────────────────────────────
        # Streamlit reruns on Enter/blur, not every keystroke (no JS in this
        # environment), so "live" here means "updates as soon as you pause
        # typing or hit Enter" — the closest native equivalent to a YouTube-
        # style dropdown without a fragile custom JS component. Matches by
        # ticker OR company name, case-insensitive, substring match.
        _typed = st.session_state.get('ticker_field', '').strip().lower()
        _search_idx = cfg.get('search_index', [])
        if _typed and len(_typed) >= 1:
            _matches = [
                (tkr, name) for tkr, name in _search_idx
                if _typed in tkr.lower() or _typed in name.lower()
            ][:5]
            # Don't show a single suggestion that's already an exact match —
            # nothing useful to click at that point.
            if _matches and not (len(_matches) == 1 and _matches[0][0].lower() == _typed):
                st.markdown("<p style='color:#71767b;font-size:11px;margin:2px 0 2px'>Suggestions:</p>",
                            unsafe_allow_html=True)
                for _tkr, _name in _matches:
                    if st.button(f"{_tkr} — {_name}", key=f"searchsug_{_tkr}", use_container_width=True):
                        st.session_state['ticker_field_pending'] = _tkr
                        st.rerun()
            elif not _matches:
                st.caption(f"No match for \"{st.session_state['ticker_field']}\" — try a ticker or company name below.")

        # Ticker chips — country-specific, clicking queues a pending value
        st.markdown(f"<p style='color:#71767b;font-size:12px;margin:8px 0 4px'>Popular in {cfg['exchanges'][0]}:</p>",
                    unsafe_allow_html=True)
        chips = cfg['example_tickers']
        chip_cols = st.columns(3)
        for i, chip in enumerate(chips):
            with chip_cols[i % 3]:
                if st.button(chip, key=f"chip_{chip}", use_container_width=True):
                    st.session_state['ticker_field_pending'] = chip
                    st.rerun()

        period = st.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y"],
                              index=2, label_visibility="collapsed")
        analyze = st.button("Analyze →", type="primary", use_container_width=True)

        if analyze:
            _tkr = st.session_state['ticker_field'].strip().upper()
            with st.spinner("Fetching..."):
                try:
                    stock = yf.Ticker(_tkr)
                    hist  = stock.history(period=period)
                    info  = stock.info
                    if hist is not None and not hist.empty:
                        st.session_state['stock_result'] = {
                            'hist': hist, 'info': info, 'ticker': _tkr
                        }
                    else:
                        st.session_state['stock_result'] = {'error': True, 'tried': _tkr}
                except Exception as e:
                    st.session_state['stock_result'] = {'error': True, 'msg': str(e), 'tried': _tkr}

        # Recommended tickers — shown regardless of search state, gives users
        # something to explore even if their search failed or they haven't searched yet
        st.markdown(f"<p style='color:#71767b;font-size:12px;margin:16px 0 4px'>You might also like:</p>",
                    unsafe_allow_html=True)
        _reco = cfg['example_tickers'][3:6] if len(cfg['example_tickers']) >= 6 else cfg['example_tickers']
        for _r in _reco:
            if st.button(f"→ {_r}", key=f"reco_{_r}", use_container_width=True):
                st.session_state['ticker_field_pending'] = _r
                st.rerun()

    # ── CENTER: Stock Output Feed ─────────────────────────────────────────
    with center:
        sr = st.session_state.get('stock_result')

        if sr:
            if sr.get('error'):
                _tried = sr.get('tried', '')
                _examples = ", ".join(cfg['example_tickers'][:3])
                st.error(
                    f"❌ Couldn't find **\"{_tried}\"**. Search needs the ticker "
                    f"*symbol*, not the company name (e.g. use `{_examples.split(',')[0].strip()}` "
                    f"instead of the full company name). "
                    f"Try one of: {_examples} — or click a chip on the left."
                )
            else:
                hist   = sr['hist']
                info   = sr['info']
                tkr    = sr['ticker']
                price  = hist['Close'].iloc[-1]
                prev   = hist['Close'].iloc[-2] if len(hist) > 1 else price
                change = price - prev
                pct    = (change / prev) * 100
                arrow  = "▲" if change >= 0 else "▼"
                pcls   = "#00ba7c" if change >= 0 else "#f4212e"
                cname  = info.get('longName', tkr) if info else tkr
                sector = info.get('sector', 'N/A') if info else 'N/A'
                mktcap = info.get('marketCap', 0) if info else 0
                pe     = info.get('trailingPE', None) if info else None
                cur    = cfg['currency_symbol']

                st.markdown(f"""
                <div class="post">
                  <div style="font-weight:800;color:#fff;font-size:17px;margin-bottom:4px">{cname}</div>
                  <div style="color:#71767b;font-size:13px;margin-bottom:12px">{tkr} · {sector}</div>
                  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:14px">
                    <span style="font-size:34px;font-weight:900;color:#fff">{cur}{price:,.2f}</span>
                    <span style="font-size:16px;font-weight:700;color:{pcls}">{arrow} {abs(pct):.2f}%</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Open",  f"{cur}{hist['Open'].iloc[-1]:,.0f}")
                c2.metric("High",  f"{cur}{hist['High'].iloc[-1]:,.0f}")
                c3.metric("Low",   f"{cur}{hist['Low'].iloc[-1]:,.0f}")
                c4.metric("P/E" if pe else "MCap",
                          f"{pe:.1f}" if pe else f"{cur}{mktcap/1e9:.0f}B")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index, y=hist['Close'], mode='lines',
                    line=dict(color='#1d9bf0', width=2),
                    fill='tozeroy', fillcolor='rgba(29,155,240,.05)'
                ))
                fig.update_layout(
                    template='plotly_dark', height=260,
                    margin=dict(l=0,r=0,t=4,b=0),
                    plot_bgcolor='#000', paper_bgcolor='#000',
                    font=dict(color='#71767b', size=11),
                    xaxis=dict(showgrid=False, zeroline=False, color='#71767b'),
                    yaxis=dict(showgrid=True, gridcolor='#16181c', zeroline=False, color='#71767b'),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={'displayModeBar': False})

                if mktcap:
                    # India traditionally reports market cap in Crore (1 Cr = 10,000,000);
                    # everywhere else uses Billion (1 B = 1,000,000,000).
                    if cfg['code'] == "IN":
                        mcap_str = f"{cur}{mktcap/10000000:,.0f} Cr"
                    else:
                        mcap_str = f"{cur}{mktcap/1e9:,.1f}B"
                    st.markdown(f"""
                    <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:13px;
                                padding-top:8px;border-top:1px solid #2f3336">
                      <span style="color:#71767b">🏢 <b style="color:#e7e9ea">{cname}</b></span>
                      <span style="color:#71767b">🏷️ <b style="color:#e7e9ea">{sector}</b></span>
                      <span style="color:#71767b">📊 <b style="color:#e7e9ea">{mcap_str}</b></span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="post">
              <p style="color:#e7e9ea;font-size:15px;line-height:1.65;margin:0 0 10px">
                <strong style="color:#fff">LLM-Driven Financial Literacy Research</strong><br>
                This platform investigates how large language models can reduce behavioral barriers
                to investment participation — targeting markets where a large share
                of the population remains excluded due to knowledge gaps and cognitive biases.
                Currently viewing: {cfg['flag']} {country} ({cfg['regulator']}).
              </p>
              <div><span class="ptag">#RAG</span><span class="ptag">#LLMs</span>
                   <span class="ptag">#BehavioralFinance</span><span class="ptag">#MultiJurisdiction</span></div>
            </div>
            <div class="post">
              <p style="color:#e7e9ea;font-size:15px;line-height:1.65;margin:0 0 10px">
                <strong style="color:#fff">Research Question</strong><br>
                Can personalized LLM explanations — calibrated by user cognitive load, cultural
                context, and local regulatory framework — measurably improve financial decision
                confidence in retail investors? Enter a ticker on the left to explore live data.
              </p>
              <div><span class="ptag">#NLP</span><span class="ptag">#FinancialInclusion</span>
                   <span class="ptag">#AIResearch</span></div>
            </div>
            <div class="post">
              <p style="color:#e7e9ea;font-size:15px;line-height:1.65;margin:0">
                <strong style="color:#fff">Methodology: RAG + Behavioral Analysis</strong><br>
                User interactions are logged to study which explanation strategies — analogical,
                statistical, or narrative — drive the highest comprehension and confidence scores
                across different demographic segments and regulatory environments.
              </p>
            </div>
            """, unsafe_allow_html=True)

        # recent Q&A
        if st.session_state.get('conversations'):
            for conv in reversed(st.session_state['conversations'][-3:]):
                with st.expander(f"💬  {conv['question'][:55]}..."):
                    st.markdown(f"**A:** {conv['answer']}")
                    st.caption(conv['timestamp'].strftime('%H:%M'))

    # ── RIGHT: AI Chat teaser — clicking anything here jumps to the full
    #    AI Teacher tab and auto-asks, per the requested UX ────────────────
    with right:
        st.markdown(f"""
        <div style="background:#000;border:1px solid #2f3336;border-radius:16px;overflow:hidden">
          <div style="background:#000;border-bottom:1px solid #2f3336;padding:14px 18px;
                      display:flex;align-items:center;gap:10px">
            <div style="width:34px;height:34px;background:linear-gradient(135deg,#1d9bf0,#0a5f8f);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:16px">🤖</div>
            <div>
              <div style="font-weight:800;color:#fff;font-size:15px">FLAIR AI</div>
              <div style="color:#71767b;font-size:12px">Powered by Llama 3.3 (Groq) · {cfg['flag']} {country}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        _ex_tkr = cfg['example_tickers'][0].split('.')[0]
        suggestions = [
            "I'm scared to invest. Help?",
            "What is P/E ratio?",
            f"Can I start with {cfg['currency_symbol']}100?",
            "SIP vs lump sum?" if cfg['code'] == "IN" else "Investing regularly vs all at once?",
            "Stocks vs Mutual Funds?" if cfg['code'] == "IN" else "Stocks vs ETFs/Funds?",
            "How to read a balance sheet?",
        ]
        for s in suggestions:
            if st.button(s, key=f"sug_{s}", use_container_width=True):
                # Jump straight to the AI Teacher tab and ask it there —
                # this is the single place answers + follow-ups render.
                st.session_state['pending_question'] = s
                st.session_state['active_tab'] = 'ai'
                st.rerun()

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        quick_q = st.text_area(
            "Ask", placeholder="Ask anything about investing...",
            height=88, label_visibility="collapsed", key="qa_textarea_markets"
        )

        if st.button("Ask AI →", type="primary", use_container_width=True, key="ask_btn_markets"):
            q = st.session_state.get("qa_textarea_markets", "").strip()
            if q:
                st.session_state['pending_question'] = q
                st.session_state['active_tab'] = 'ai'
                st.rerun()
            else:
                st.warning("Type a question first!")

        st.caption("💡 Tip: answers open in the AI Teacher tab, with follow-up questions.")

# ════════════════════════════════════════════════════════════════════════════
# TAB: AI TEACHER (full page)
# ════════════════════════════════════════════════════════════════════════════
elif active == 'ai':
    st.markdown("<div style='max-width:700px;margin:0 auto;padding:0 16px'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
      <p class="card-title">🤖 AI Financial Teacher</p>
      <p style="color:#71767b;font-size:14px;margin:0">
        Powered by Llama 3.3 (Groq) · RAG-enhanced · Calibrated for {cfg['flag']} {country} ({cfg['regulator']})
      </p>
    </div>
    """, unsafe_allow_html=True)

    # If a suggestion pill (from Markets tab, or a follow-up below) queued a
    # question, answer it immediately AND show it in the textarea — this is
    # what makes clicking a suggestion "just work" with visible confirmation
    # of what was asked, instead of a silent redirect.
    _pending = st.session_state.get('pending_question')
    if _pending:
        st.session_state['pending_question'] = None
        st.session_state['ai_tab_textarea'] = _pending  # safe: textarea below not yet instantiated this run
        with st.spinner("Thinking..."):
            answer = ask_flair_ai(_pending, cfg, country)
        st.session_state['ai_result'] = {'question': _pending, 'answer': answer}
        st.session_state['interactions'].append({
            'timestamp': datetime.now().isoformat(), 'question': _pending,
            'topic': 'AI', 'country': country
        })
        st.session_state['conversations'].append({
            'question': _pending, 'answer': answer, 'topic': 'AI',
            'timestamp': datetime.now(), 'country': country
        })
        st.session_state['followups'] = generate_followups(_pending, answer, cfg, country)

    full_q = st.text_area("Your question", placeholder="Ask anything — concepts, companies, strategies...",
                          height=120, label_visibility="collapsed", key="ai_tab_textarea")

    if st.button("Ask AI →", type="primary", use_container_width=True, key="ai_tab_ask"):
        q = st.session_state.get("ai_tab_textarea", "").strip()
        if q:
            with st.spinner("Thinking..."):
                answer = ask_flair_ai(q, cfg, country)
            st.session_state['ai_result'] = {'question': q, 'answer': answer}
            st.session_state['interactions'].append({
                'timestamp': datetime.now().isoformat(), 'question': q,
                'topic': 'AI', 'country': country
            })
            st.session_state['conversations'].append({
                'question': q, 'answer': answer, 'topic': 'AI',
                'timestamp': datetime.now(), 'country': country
            })
            st.session_state['followups'] = generate_followups(q, answer, cfg, country)
            st.rerun()
        else:
            st.warning("Enter a question!")

    ai = st.session_state.get('ai_result')
    if ai:
        st.markdown(f"""
        <div class="ai-bubble" style="margin-top:16px">
          <p style="color:#71767b;font-size:13px;margin:0 0 8px">
            <strong style="color:#e7e9ea">Q:</strong> {ai['question']}
          </p>
          <p style="color:#e7e9ea;font-size:15px;line-height:1.7;margin:0">{ai['answer']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Clickable follow-up questions — continue the conversation in one click
        _fu = st.session_state.get('followups', [])
        if _fu:
            st.markdown("<p style='color:#71767b;font-size:12px;margin:14px 0 4px'>Ask next:</p>",
                        unsafe_allow_html=True)
            fu_cols = st.columns(len(_fu))
            for i, fq in enumerate(_fu):
                with fu_cols[i]:
                    if st.button(fq, key=f"followup_{i}_{fq[:20]}", use_container_width=True):
                        st.session_state['pending_question'] = fq
                        st.rerun()

    if st.session_state['conversations']:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p style='color:#71767b;font-size:14px;font-weight:600'>History</p>",
                    unsafe_allow_html=True)
        for conv in reversed(st.session_state['conversations'][-5:]):
            with st.expander(f"Q: {conv['question'][:60]}..."):
                st.markdown(f"**A:** {conv['answer']}")
                st.caption(conv['timestamp'].strftime('%H:%M'))

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB: RESEARCH
# ════════════════════════════════════════════════════════════════════════════
elif active == 'research':
    r1, r2 = st.columns([1.2, 1], gap="large")
    with r1:
        st.markdown("""
        <div class="card">
          <p class="card-title">🔬 Research Focus</p>
          <p style="color:#e7e9ea;font-size:15px;line-height:1.7">
            FLAIR is an ongoing research initiative at the intersection of
            <strong style="color:#fff">large language models, financial literacy,
            and behavioral economics</strong> — with a focus on cross-jurisdiction
            financial education.
          </p>
          <div style="margin-top:16px;padding-top:16px;border-top:1px solid #2f3336">
            <p style="color:#71767b;font-size:14px;line-height:1.7;margin:0">
              Research details and findings are currently under preparation for publication.
              This platform serves as the live data collection and validation environment.
            </p>
          </div>
        </div>
        <div class="card">
          <p class="card-title">📚 Areas of Investigation</p>
          <p style="color:#e7e9ea;font-size:14px;line-height:1.8;margin:0">
            <span style="color:#1d9bf0">·</span> LLMs for domain-specific education<br>
            <span style="color:#1d9bf0">·</span> Retrieval-Augmented Generation (RAG)<br>
            <span style="color:#1d9bf0">·</span> Behavioral barriers to financial participation<br>
            <span style="color:#1d9bf0">·</span> Personalization in AI-driven learning<br>
            <span style="color:#1d9bf0">·</span> Cross-jurisdiction financial inclusion<br>
            <span style="color:#1d9bf0">·</span> Human-AI interaction in education
          </p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="card"><p class="card-title">📊 Platform Stats</p>',
                    unsafe_allow_html=True)
        st.metric("Questions Asked", len(st.session_state['interactions']))
        st.metric("Conversations",   len(st.session_state['conversations']))
        st.metric("Status", "Active · Collecting Data")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
          <p class="card-title">📩 Collaborate</p>
          <p style="color:#71767b;font-size:14px;line-height:1.7;margin:0">
            Interested in research collaboration, data partnerships, or
            early access?<br><br>
            <strong style="color:#e7e9ea">steaphen.ai@gmail.com</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB: GLOBAL
# ════════════════════════════════════════════════════════════════════════════
elif active == 'global':
    g1, g2 = st.columns([1, 1], gap="large")
    with g1:
        st.markdown("""
        <div class="card">
          <p class="card-title">🔬 Research Context</p>
          <p style="color:#e7e9ea;font-size:15px;line-height:1.7">
            Financial literacy is a well-studied gap in behavioral economics and education
            research — prior work has linked comprehension barriers to
            <strong style="color:#fff">explanation complexity, language, and cultural framing</strong>
            rather than access alone.
          </p>
          <div style="margin-top:20px">
            <div style="background:#0a0a0a;border:1px solid #2f3336;border-radius:12px;padding:18px;margin-bottom:10px">
              <p style="color:#1d9bf0;font-size:12px;font-weight:700;margin:0 0 6px;text-transform:uppercase;letter-spacing:1px">Open Question</p>
              <p style="color:#e7e9ea;font-size:14px;line-height:1.7;margin:0">
                Does adapting an explanation's framing to an individual's background
                measurably change comprehension outcomes, compared to a single
                fixed explanation for everyone?
              </p>
            </div>
            <div style="background:#0a0a0a;border:1px solid #2f3336;border-radius:12px;padding:18px">
              <p style="color:#1d9bf0;font-size:12px;font-weight:700;margin:0 0 6px;text-transform:uppercase;letter-spacing:1px">Why LLMs</p>
              <p style="color:#e7e9ea;font-size:14px;line-height:1.7;margin:0">
                Prior personalization research was constrained by needing pre-written
                content for every variation. Language models remove that constraint —
                enabling controlled study of explanation-strategy effects at a scale
                that wasn't previously testable.
              </p>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div class="card">
          <p class="card-title">🎯 Scope of This Platform</p>
          <p style="color:#71767b;font-size:14px;line-height:1.7;margin:0">
            FLAIR currently operates as a multi-jurisdiction data collection instrument —
            supporting India, the US, the UK, and the UAE — to study whether
            comprehension effects generalize across regulatory and cultural contexts,
            or are market-specific.<br><br>
            Full methodology is withheld pending publication.
            If you're a researcher, potential collaborator, or lab interested in this direction:<br><br>
            <strong style="color:#e7e9ea">steaphen.ai@gmail.com</strong>
          </p>
        </div>
        <div class="card">
          <p class="card-title">📊 Why This Matters</p>
          <p style="color:#e7e9ea;font-size:14px;line-height:1.7;margin:0">
            Personalized-explanation research has direct implications beyond finance —
            for any domain where comprehension gaps limit good decision-making.
            <strong style="color:#fff">This platform is the experimental testbed for that question.</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="border-top:1px solid #2f3336;margin-top:40px;padding:20px;
            text-align:center;color:#555;font-size:12px">
  <strong style="color:#333">FLAIR</strong> ·
  LLM-Powered Financial Literacy Research ·
  🔬 RAG · Behavioral AI · Multi-Jurisdiction
  <br><span style="font-size:11px;margin-top:4px;display:block">
    ⚠️ Educational only. Not investment advice. Currently viewing: {cfg['flag']} {country} ({cfg['regulator']})
  </span>
</div>
""", unsafe_allow_html=True)
