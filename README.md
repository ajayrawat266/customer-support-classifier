# Customer Support Ticket Classifier & Routing Engine

## 📊 Dashboard
**[View on Tableau Public →](https://public.tableau.com/views/CustomerSupportTickets_17773573358470/CustomerSupportTicketClassifierRoutingEngine)**

---

## Problem Statement
Support teams manually triage thousands of tickets daily — a slow, inconsistent, and unscalable process. This project automates ticket classification and intelligent routing using NLP and a fine-tuned BERT model, reducing manual triage burden and enabling real-time priority escalation across 6 support categories.

---

## Results

| Metric | Value |
|---|---|
| Tickets Routed | 80,000 |
| Routing Accuracy | **98.63%** |
| Avg BERT Confidence | **99.34%** |
| Escalation Rate | **7.22%** |
| Billing & Payments Accuracy | **100%** |
| Baseline Macro F1 (Random Forest) | 0.9858 |
| BERT Macro F1 | 0.9774 |
| GPU Training Time (T4) | 189 seconds |
| Batch Routing Time (80K tickets) | 314 seconds |

---

## Dataset
**Kaggle: Customer Support on Twitter** (thoughtvector)
- 2.8M tweets from brands including Apple, Amazon, Uber, Spotify, Comcast
- Filtered to inbound customer messages: **1.5M tickets**
- Sampled 80,000 for modeling
- **Labels engineered via TF-IDF + K-Means clustering** — no pre-existing labels

---

## Tech Stack
| Layer | Tools |
|---|---|
| Language | Python 3.9 |
| ML / NLP | scikit-learn, HuggingFace Transformers (BERT), VADER Sentiment |
| Data Engineering | pandas, SQLite |
| Visualization | Tableau Public |
| Infrastructure | Google Colab T4 GPU |

---

## Architecture

1. Raw Tweets (2.8M) 

2. Filter Inbound (1.5M customer messages) 

3. Clean Text (remove @mentions, URLs, punctuation) 

4. TF-IDF + K-Means → Label Engineering (6 categories) 

5. SMOTE → Baseline Models (Logistic Regression, Random Forest) 

6. Fine-tune BERT (bert-base-uncased, T4 GPU, 3 epochs) 

7. route_ticket() → Category + Confidence + Team + Priority + Escalation Flag 

8. SQLite Database → SQL Analysis → Tableau Dashboard

---

## 6 Support Categories (Engineer-Labeled via K-Means)

| Category | Team Routed To | SLA |
|---|---|---|
| account_access | Account & Security | 2h |
| billing_payment | Billing & Payments | 4h |
| technical_issue | Technical Support | 6h |
| general_complaint | Customer Relations | 12h |
| shipping_delivery | Logistics & Fulfilment | 8h |
| general_inquiry | General Support | 24h |

---

---

## Notebooks Walkthrough

### 01 — EDA
Loaded 2.8M tweets, filtered to 1.5M inbound customer messages. Explored text length distributions, daily volume trends, and top keywords to understand the raw data before modeling.

### 02 — Label Engineering
No labels existed in this dataset — we created them. Cleaned tweet text, ran TF-IDF vectorization on 80K tweets, applied K-Means clustering (K=6 via elbow method) to discover 6 natural issue categories. Added priority scores 1-3 using VADER sentiment and urgency keyword detection.

### 03 — Baseline Models
TF-IDF features + Logistic Regression and Random Forest classifiers. Applied SMOTE to handle class imbalance (76% account_access). Best baseline: **Random Forest — Macro F1: 0.9858**. This is the benchmark BERT must beat.

### 04 — BERT Fine-Tuning
Fine-tuned `bert-base-uncased` on 15,000 labeled tickets using HuggingFace Trainer API. 3 epochs, batch size 32, T4 GPU. Training completed in **189 seconds**. BERT Macro F1: **0.9774** — comparable to baseline, with stronger contextual understanding for production use.

### 05 — Routing Engine
Built `route_ticket(text)` — a full routing pipeline that takes raw tweet text and returns: predicted category, confidence score, assigned team, SLA window, priority level (1-3), and escalation flag. Batch routed **80,000 tickets in 314 seconds** on T4 GPU. Results written to SQLite, exported 3 Tableau-ready CSVs.

---

## Key SQL Insights

```sql
-- Overall routing performance
SELECT
    COUNT(*)                                        AS total_routed,     -- 80,000
    ROUND(SUM(correctly_routed)*100.0/COUNT(*), 2)  AS routing_accuracy, -- 98.63%
    ROUND(SUM(escalate)*100.0/COUNT(*), 2)          AS escalation_rate,  -- 7.22%
    ROUND(AVG(confidence), 4)                       AS avg_confidence    -- 0.9934
FROM routing_results;
```

**Routing accuracy by team:**
| Team | Accuracy |
|---|---|
| Billing & Payments | 100.0% |
| Account & Security | 99.7% |
| Logistics & Fulfilment | 98.4% |
| Technical Support | 95.9% |
| General Support | 94.9% |
| Customer Relations | 91.9% |

---

## Business Impact
- **Eliminates manual first-level triage** — estimated 60% reduction in ops time
- **Routes fraud/account tickets in real time** with 99.7% accuracy
- **Flags only 7.22% of tickets for human review** — the ones that actually need it
- **Live Tableau ops dashboard** — ticket health, priority queue, routing accuracy at a glance
- **Scales to millions of tickets** — 80K routed in 5 minutes on GPU

---

## Known Limitations
- `account_access` dominates labels (75.4%) — edge case misrouting expected on minority categories
- Labels are self-engineered via clustering, not human-verified ground truth
- Dataset covers October 2017 only — model may need retraining for other time periods
- Future improvement: balanced resampling across categories before BERT fine-tuning
