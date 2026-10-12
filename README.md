# finRAG
My solution for Financial RAG


## Task:
---
Analysis of Annual Reports within the automotive sector

Extracting key ﬁnancial metrics such as revenue, EBITDA, growth numbers, and 
conducting comparative analyses across diﬀerent car companies, 
as well as comparing them to other sectors.

## Proposed Solution:

AI-based RAG pipeline to extract and analyze key financial metrics from annual reports of car companies.
The pipeline will be able to answer questions about the financial performance of car companies, e.g., Tesla, Ford, BMW, etc.

## Test Cases for RAG Pipeline
---

from the instruction:

What was BMW's total revenue in 2023?

How much revenue did Tesla generate in 2023?

What was Ford's revenue for the year 2020?

Can you provide the revenue ﬁgures for BMW in 2017?

What key economic factors inﬂuenced Ford's performance in 2021?

Which Tesla product is currently in the development stage?

What were BMW's proﬁt ﬁgures for 2020 and 2023?

Between Tesla and Ford, which company achieved higher proﬁts in 2022?

What were Tesla's proﬁt numbers for 2022 and 2023?

Which company recorded better proﬁtability in 2022 overall?

Provide a summary of revenue ﬁgures for Tesla, BMW, and Ford over the past three years.

What were the growth trends for BMW's ﬁnancial performance from 2020 to 2023?


personal discovery:

How many vehicles did Rivian deliver in 2022?

Tesla's stock price on December 31, 2023

Compare the 2021 revenue of BMW and Tesla.

Which company had a higher net income in 2022: Tesla or Ford?


Summarize the revenue growth of Tesla from 2021 to 2023.

Why did Ford report a net loss in 2022 despite strong automotive sales


What caused the massive drop in Tesla's effective tax rate in 2023?

Distractor / news.pdf file:

How did the central bank's 0.5% interest rate hike affect Ford's F-150 Lightning sales?

Did the new climate bill passed by Congress impact BMW's NEUE KLASSE platform?

What were Tesla's exact revenues from the 'Energy generation and storage' segment in 2021, 2022, and 2023?

What is the target EBIT margin for BMW's Automotive segment?

Break down the $12.17 billion of 'Special Items' expenses Ford recorded in 2022.

Memory (follow-up questions)

What raw materials is Ford dependent on for EV batteries?

Does Tesla face the same supply chain risks for those materials?

What about BMW?

## Required Packages:
- Transformers
- Torch
- Sentence-transformers
- Docling
- Rich

