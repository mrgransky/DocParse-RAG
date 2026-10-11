# finRAG
My solution for Financial RAG

## Test Cases for RAG Pipeline
---

### 1. The "Missing Data" & Hallucination Traps
*These test your strict `DATA_MISSING` system prompt and the model's willingness to admit ignorance.*

| # | Query Prompt | Ground Truth  | Red Flag (Pipeline Failure) |
|---|---|---|---|
| **1** | "What was BMW's total revenue in 2023?" | **DATA_MISSING**. Your dataset only contains the **BMW 2021** report. | The model hallucinates a number, or incorrectly pulls Tesla/Ford 2023 data. |
| **2** | "How many vehicles did Rivian deliver in 2022?" | **DATA_MISSING**. Rivian is mentioned in Ford's report as an investment, but no delivery numbers are provided. | The model uses its pre-trained internet knowledge to answer (violating the "use ONLY provided context" rule). |
| **3** | "What was Tesla's stock price on December 31, 2023?" | **DATA_MISSING**. The 10-K mentions market cap and shares outstanding, but not the exact closing stock price. | The model hallucinates a financial figure. |

### 2. The Cross-Company & Currency Traps
*These test multi-document retrieval and financial literacy (handling different currencies and units).*

| # | Query Prompt | Ground Truth  | Red Flag (Pipeline Failure) |
|---|---|---|---|
| **4** | "Compare the 2021 revenue of BMW and Tesla." | **BMW:** €111,239 million (EUR). **Tesla:** $53,823 million (USD). | The model ignores currencies and falsely claims BMW's revenue is more than double Tesla's without noting the EUR/USD difference. |
| **5** | "Which company had a higher net income in 2022: Tesla or Ford?" | **Tesla:** ~$12.56 billion. **Ford:** Net **loss** of $1.98 billion. | The model gets confused by Ford's *Adjusted EBIT* ($10.4B) and incorrectly states Ford was profitable in 2022. |
| **6** | "Summarize the revenue growth of Tesla from 2021 to 2023." | 2021: $53.82B ➔ 2022: $81.46B ➔ 2023: $96.77B. | The retriever only pulls the 2023 summary page and fails to retrieve the historical 2021/2022 tables. |

### 3. The GAAP vs. Non-GAAP Nuance Traps
*Financial reports are full of "Special Items". A good RAG bot must explain the "Why", not just the "What".*

| # | Query Prompt | Ground Truth  | Red Flag (Pipeline Failure) |
|---|---|---|---|
| **7** | "Why did Ford report a net loss in 2022 despite strong automotive sales?" | Driven by **$12.2B in pre-tax special items**, specifically a $7.4B mark-to-market loss on Rivian and a $2.7B impairment on Argo AI. | The model just says "Ford lost money" without citing the non-cash special items that skewed the GAAP net income. |
| **8** | "What caused the massive drop in Tesla's effective tax rate in 2023?" | A one-time, non-cash tax benefit of **$5.93 billion** due to the release of valuation allowances on deferred tax assets. | The model misses the $5.93B figure buried in the income tax notes. |

### 4. The Distractor / "News" Traps
*These test if your embedding model (`bge-m3`) and retriever can ignore the `news.pdf` file when asked about specific companies.*

| # | Query Prompt | Ground Truth  | Red Flag (Pipeline Failure) |
|---|---|---|---|
| **9** | "How did the central bank's 0.5% interest rate hike affect Ford's F-150 Lightning sales?" | **Refusal / DATA_MISSING**. `news.pdf` mentions a 0.5% hike. `Ford` mentions F-150 Lightning. But there is **no causal link** in the text. | The model synthesizes a fake narrative combining the generic news article with Ford's EV segment. |
| **10** | "Did the new climate bill passed by Congress impact BMW's NEUE KLASSE platform?" | **Refusal / DATA_MISSING**. `news.pdf` mentions a climate bill. BMW mentions NEUE KLASSE. They are unrelated in the provided text. | The model hallucinates that the climate bill forced BMW to change its EV architecture. |

### 5. Table & Nested Fact Extraction
*PDF parsers (`pypdf`) often mangle tables. These test if your chunking strategy preserved the row/column relationships.*

| # | Query Prompt | Ground Truth  | Red Flag (Pipeline Failure) |
|---|---|---|---|
| **11** | "What were Tesla's exact revenues from the 'Energy generation and storage' segment in 2021, 2022, and 2023?" | 2021: $2,789M | 2022: $3,909M | 2023: $6,035M. | The model mixes up the columns and attributes Automotive Sales numbers to the Energy segment. |
| **12** | "What is the target EBIT margin for BMW's Automotive segment?" | **8% to 10%** (Found in the BMW Integrated Strategy / Target Achievement section). | The model pulls the *actual* 2021 margin (10.3%) instead of the *strategic target* (8-10%). |
| **13** | "Break down the $12.17 billion of 'Special Items' expenses Ford recorded in 2022." | Must list: Rivian ($7.4B), Argo AI ($2.7B), Debt extinguishment, etc. | The retriever fails to pull the "Special Items" table from the MD&A section. |

### 6. Conversational Memory (Follow-ups)
*These test your `chat_history` implementation. Ask these sequentially in the same session.*

| Sequence | Query Prompt | Expected Behavior | Red Flag |
|---|---|---|---|
| **Q1** | "What raw materials is Ford dependent on for EV batteries?" | Lithium, cobalt, nickel, graphite, manganese. | N/A |
| **Q2** | "Does Tesla face the same supply chain risks for those materials?" | **Yes.** Model should recall "those materials" from Q1 and search Tesla's document for lithium/cobalt/nickel supply risks. | Model forgets the context and asks "What materials?" or searches for generic Tesla risks. |
| **Q3** | "What about BMW?" | Model should search BMW's 2021 report for battery raw material risks. (Note: BMW discusses supply chain and circularity, but the specific raw material list might be less explicit than Ford/Tesla). | Model hallucinates that BMW uses the exact same suppliers as Ford. |

---

Actual challenges in enterprise RAG:
> *"Any basic LLM can answer 'What is Tesla's revenue'. How my pipeline handles **edge cases, financial nuance, and distractor resistance**."*

**Query #4 (Currency Trap)**

**Query #7 (GAAP vs Non-GAAP)**

**Query #9 (The News Distractor)**

Failure with 0.5B/14B models: as they struggle with the multi-document math or table extraction:
> *"Open-weights model struggles to align the columns of this mangled PDF table. In a production environment at AECOM, I would solve this by routing table-heavy pages through a Vision-Language Model like LlamaParse, or by implementing an Agentic Router that passes tabular data to a Python/Pandas tool for deterministic math."*

This turns a potential model failure into a demonstration of **senior systems architecture knowledge**.