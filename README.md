\# 🌍 Geopolitical Risk \& Portfolio Tail Dashboard



> \*\*"How much money could you lose if a war breaks out tomorrow — and is your portfolio ready for it?"\*\*



Most financial tools assume the world is always calm. This one doesn't.



This dashboard uses \*\*live news\*\*, \*\*10 years of market data\*\*, and a \*\*smarter statistical model\*\* to show you the honest answer to that question — even if you've never studied finance or statistics.



\---



\## 🧠 The problem this solves (in plain English)



Imagine you own a mix of investments — stocks, bonds, gold, maybe Bitcoin. A financial advisor might tell you: \*"Don't worry, they're diversified — when one falls, the others hold up."\*



That's mostly true. \*\*Except during crises.\*\*



When wars start, pandemics hit, or major political shocks happen, something strange occurs in markets: \*\*everything falls together at the same time\*\*. The diversification you were counting on disappears exactly when you need it most.



Standard risk tools (used by most banks and apps) completely miss this. They assume crashes are independent — like rolling dice. This dashboard uses a better model that captures the "everything crashes together" effect, and adjusts it \*\*based on what's actually happening in the news right now\*\*.



\---



\## 📺 Live demo



👉 \*\*\[Open the dashboard](https://your-streamlit-url-here.streamlit.app)\*\*



\*(Replace this link after deploying to Streamlit Cloud)\*



\---



\## 📖 Key terms explained — with real examples



\*\*GPR Index (Geopolitical Risk Index)\*\*

A daily score that counts how many newspaper articles mention war, terrorism, and conflict.

\- Score of 100 = normal day in 2015

\- Score of 380 = February 2022, the day Russia invaded Ukraine

\- Score today = shown live on the dashboard



\*\*Calm vs Crisis regime\*\*

We split all history into two buckets based on GPR:

\- \*Calm\* = GPR below the 75th percentile — like most of 2017 or 2019

\- \*Crisis\* = GPR above the 75th percentile — like March 2020 or Feb–Mar 2022

We fit a separate risk model for each bucket.



\*\*VaR (Value at Risk)\*\*

"On the worst 1 in 100 trading days, I will lose \*at least\* this much."

\- Example: VaR = −2.1% on a $100,000 portfolio means on a bad day you lose at least $2,100.

\- It's a floor, not an average — actual losses on those days could be worse.



\*\*CVaR (Conditional VaR / Expected Shortfall)\*\*

"On those worst days, what is my \*average\* loss?"

\- Example: CVaR = −2.8% means that across all the worst 1% of days, you lost 2.8% on average.

\- More useful than VaR because it tells you how bad "bad" actually gets.



\*\*Copula model\*\*

A way of modelling how assets move \*together\*, not just individually.

\- Standard model (Gaussian): assumes SPY and QQQ crashes are independent coin flips.

\- Copula model (Clayton): knows that if SPY drops 5%, QQQ is very likely to drop too — especially in a crisis.

\- Real example: on March 16, 2020, SPY fell 12%, QQQ fell 12.3%, GLD fell 3.1% — \*all at once\*.



\*\*Tail dependence λL\*\*

The probability that Asset B crashes hard \*given that\* Asset A already crashed hard.

\- λL = 0.05 → 5% chance they crash together (good diversification)

\- λL = 0.40 → 40% chance they crash together (your diversification is largely an illusion in a crisis)

\- On this dashboard: crisis-regime λL is always higher than calm-regime λL.



\*\*Clayton copula θ (theta)\*\*

A single number measuring crash co-movement strength.

\- θ = 0.5 → mild co-movement (calm markets)

\- θ = 2.0 → strong co-movement (crisis markets, assets tend to crash together)

\- Our dashboard fits θ separately for calm and crisis — and shows you the difference.



\## 🗺️ What you'll see



\### Tab 1 — 🌍 Live Geopolitical Risk

\- A timeline of the \*\*GPR index\*\* (explained below) over your chosen period

\- Red zones show historical "crisis" periods — Ukraine 2022, COVID 2020, etc.

\- \*\*Today's live news headlines\*\*, fetched in real time and scored for risk

\- A gauge showing whether today's news is calm or alarming



\### Tab 2 — 📉 Regime Risk Comparison

\- \*\*Side-by-side comparison\*\* of your portfolio's risk in calm vs crisis times

\- Dollar amounts showing exactly how much more you could lose in a crisis

\- The key insight: your assets are X% more likely to crash \*together\* during a geopolitical event



\### Tab 3 — 🔗 Tail Dependence

\- A chart showing how crash correlations between your assets \*\*changed over time\*\*

\- You can literally see the lines spike during Ukraine 2022 and COVID 2020

\- This is the visual proof that diversification is unreliable during crises



\### Tab 4 — 📊 Full Model Detail

\- The full statistical output for anyone who wants to dig deeper

\- Copula model vs standard Gaussian comparison



\---



\## 📖 Key terms explained (no jargon)



You don't need to understand statistics to use this dashboard. But here's what the words mean if you're curious:



| Term | What it actually means |

|------|------------------------|

| \*\*GPR Index\*\* | A score invented by two Federal Reserve economists. It reads 10 major newspapers every day since 1985 and counts how many articles mention war, terrorism, and conflict. Higher score = scarier world. |

| \*\*Calm regime\*\* | A period when the GPR is in its normal range — roughly 75% of all historical days |

| \*\*Crisis regime\*\* | A period when the GPR spikes into the top 25% — wars, 9/11, Ukraine invasion, COVID crash |

| \*\*Copula model\*\* | A statistical model that captures how assets tend to crash \*together\*. Unlike the standard model, it knows that correlations change during crises. |

| \*\*Clayton copula\*\* | A specific type of copula that's especially good at modelling downside crashes — used here for crisis regime |

| \*\*VaR (Value at Risk)\*\* | "On the worst 1% of trading days, I will lose \*at least\* this much" — think of it as the floor of bad outcomes |

| \*\*CVaR (Conditional VaR)\*\* | "On those worst days, my \*average\* loss will be this much" — more useful than VaR because it captures how bad the bad days really get |

| \*\*Tail dependence λL\*\* | The probability that two assets crash at the same time. A value of 0.3 means: if Asset A crashes hard, there's a 30% chance Asset B crashes at the same time |

| \*\*GARCH\*\* | A model that accounts for the fact that volatile days cluster together — after a big crash, more big moves tend to follow |

| \*\*Gaussian model\*\* | The textbook standard model used by most banks. Assumes crashes are random and independent. Consistently underestimates crisis-period losses. |



\---



\## 🔬 What makes this project unique



Almost every portfolio risk project on GitHub does the same thing: download stock data, compute standard VaR, show a chart. This project is different in three ways:



\*\*1. It uses the Geopolitical Risk Index as a regime signal\*\*

The \[Caldara-Iacoviello GPR index](https://www.matteoiacoviello.com/gpr.htm) is a peer-reviewed measure built by Federal Reserve economists. It's been cited in hundreds of academic papers. No open-source portfolio dashboard uses it as a live risk input.



\*\*2. It fits separate crash models for calm vs crisis periods\*\*

Most copula tools fit one model to all history. We fit two — one on calm-regime data, one on crisis-regime data — then switch between them based on today's GPR reading. This means the risk estimate you see is conditioned on the current geopolitical environment, not just a historical average.



\*\*3. It shows the crash correlation shift in real time\*\*

The rolling tail dependence chart shows you, over time, how the probability of joint crashes between your assets changed. You can see it spike in February 2022 when Russia invaded Ukraine. That's not a model assumption — that's what the data shows happened.



\---



\## 🛠️ How to run it yourself



\### What you need first

\- Python 3.10 or newer (\[download here](https://python.org))

\- A free NewsAPI key from \[newsapi.org](https://newsapi.org) (optional but recommended — takes 2 minutes, no credit card)



\### Step 1 — Download the project

```bash

git clone https://github.com/harshinireddy2204/copula-risk-dashboard.git

cd copula-risk-dashboard

```



\### Step 2 — Create a virtual environment

```bash

\# Windows

python -m venv venv

venv\\Scripts\\activate



\# Mac / Linux

python3 -m venv venv

source venv/bin/activate

```



\### Step 3 — Install dependencies

```bash

pip install -r requirements.txt

```



\### Step 4 — Run the dashboard

```bash

streamlit run app.py

```



Your browser will open automatically at `http://localhost:8501`



\### Step 5 — (Optional) Add your NewsAPI key

In the sidebar, paste your free key from \[newsapi.org](https://newsapi.org). Without it, the dashboard shows demo headlines but everything else works fully.



\---



\## 📁 Project structure



```

copula-risk-dashboard/

│

├── app.py                    # Main dashboard — all tabs and layout

├── config.py                 # Asset tickers, default dates, settings

├── requirements.txt          # All Python packages needed

│

├── data/

│   ├── loader.py             # Downloads stock prices from Yahoo Finance

│   └── gpr\_loader.py        # Downloads GPR index + fetches live news

│

├── models/

│   ├── marginals.py          # GARCH volatility model for each asset

│   ├── copula.py             # Standard copula fitting (Gaussian, t, Clayton)

│   ├── gpr\_copula.py        # NEW: regime-conditioned copula model

│   └── risk.py               # VaR, CVaR, tail dependence calculations

│

├── viz/

│   ├── distribution.py       # PnL histogram chart

│   ├── heatmap.py            # Tail dependence heatmap

│   └── gpr\_charts.py        # NEW: GPR timeline, rolling tail dep, gauge

│

└── .streamlit/

&#x20;   └── config.toml           # Forces dark theme for all users

```



\---



\## 📊 The data sources



| Data | Source | Update frequency |

|------|--------|-----------------|

| Stock prices | Yahoo Finance via `yfinance` | Daily |

| GPR index | \[matteoiacoviello.com](https://www.matteoiacoviello.com/gpr.htm) | Monthly |

| News headlines | \[NewsAPI.org](https://newsapi.org) | Live |



All data is free and publicly available. No paid subscriptions required.



\---



\## 🧪 The statistical pipeline (for the curious)



If you want to understand what's happening under the hood, here's the full process in plain steps:



1\. \*\*Download prices\*\* — daily closing prices for your chosen assets from Yahoo Finance

2\. \*\*Compute log returns\*\* — the daily percentage change in price (log scale)

3\. \*\*Fit GARCH models\*\* — model the volatility of each asset separately, accounting for volatility clustering

4\. \*\*Extract residuals\*\* — remove the volatility structure to get the "pure" co-movement signal

5\. \*\*Probability Integral Transform (PIT)\*\* — convert residuals to uniform \[0,1] values, preserving rank structure

6\. \*\*Label regimes\*\* — match each day to its GPR regime (calm or crisis) using the GPR index

7\. \*\*Fit regime copulas\*\* — fit a separate Clayton copula on calm days and crisis days

8\. \*\*Simulate\*\* — draw 50,000 random scenarios from the copula matching today's regime

9\. \*\*Invert\*\* — map simulated uniform values back to real return space using the empirical distribution

10\. \*\*Compute risk\*\* — calculate VaR and CVaR from the simulated portfolio returns



\---



\## 📚 Academic references



This project implements ideas from:



\- Caldara, D. \& Iacoviello, M. (2022). \*Measuring Geopolitical Risk.\* American Economic Review.

\- Sklar, A. (1959). \*Fonctions de répartition à n dimensions et leurs marges.\* (Original copula theorem)

\- McNeil, A., Frey, R. \& Embrechts, P. (2015). \*Quantitative Risk Management.\* Princeton University Press.

\- Engle, R. (1982). \*Autoregressive Conditional Heteroscedasticity.\* Econometrica. (GARCH foundation)



\---



\## ⚠️ Disclaimer



This dashboard is for \*\*educational and research purposes only\*\*. It is not financial advice. Past risk patterns do not guarantee future outcomes. Do not make investment decisions based solely on this tool.



\---



\## 👩‍💻 Built by



\*\*Harshinireddy2204\*\* — combining quantitative finance, real-time data, and accessible design to make institutional-grade risk tools available to everyone.



\*If you found this useful, consider giving it a ⭐ on GitHub.\*

