# Enterprise Causal AI Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/Google%20GenAI-Gemini-orange.svg)](https://ai.google.dev/)
[![Causal Inference](https://img.shields.io/badge/Engine-Custom%20Hajek--IPTW%20%2B%20DoWhy-green.svg)](https://github.com/py-why/dowhy)
[![Frontend](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Database-SQLite%20(Olist%20E--Commerce)-lightgrey.svg)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

An autonomous Causal AI system that translates natural-language business questions into multi-table SQL queries using semantic Retrieval-Augmented Generation (RAG), retrieves transactional data, and estimates the **Average Treatment Effect (ATE)** using Hajek-normalized Inverse Probability of Treatment Weighting (IPTW) and statistical refutation tests.

---

## Executive Summary

Traditional machine learning identifies associations in data, but associations do not necessarily establish causal relationships. Business decisions based only on correlations can therefore lead to misleading conclusions.

The Enterprise Causal AI Engine integrates natural-language processing, relational database querying, and statistical causal inference into a single workflow.

The system provides the following capabilities:

1. **Natural Language to Causal Parameters:** Converts business questions into structured treatment, outcome, and confounder variables.
2. **Multi-Table Relational SQL:** Retrieves relevant database schemas using semantic embeddings and executes SQL queries across relational tables.
3. **Mathematical Causal Estimation:** Estimates treatment effects using Hajek-normalized propensity weighting and bootstrap confidence intervals.
4. **Automated Causal Discovery:** Compares standardized variable impact scores to identify relationships that require further investigation.
5. **Statistical Stress Testing:** Evaluates model robustness using placebo treatment tests and synthetic common-cause injections.

---

## System Architecture

```text
User Business Question
         |
         v
+-------------------------------------------------------------+
| Tier 1: Intelligent AI and RAG Router                       |
|                                                             |
| - Schema embeddings using text-embedding-004                |
| - Cosine similarity for semantic table retrieval            |
| - Structured parameter extraction                           |
| - Treatment, outcome, confounders, and SQL generation       |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Tier 2: Relational Database Execution Layer                 |
|                                                             |
| - SQLite relational database                                |
| - Multi-table SQL JOIN operations                           |
| - Data extraction and preprocessing                         |
| - Read-only SQL execution safeguards                        |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Tier 3: Mathematical Causal Engine                          |
|                                                             |
| - Smart median continuous-treatment binarization            |
| - Logistic regression propensity-score estimation           |
| - Hajek-normalized IPTW ATE estimation                      |
| - Bootstrap confidence intervals and p-values               |
| - Standardized variable-impact analysis                     |
| - DoWhy placebo and synthetic-confounder refutation tests   |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Tier 4: Streamlit Analytics Dashboard                       |
|                                                             |
| - ATE and confidence intervals                              |
| - Statistical significance                                  |
| - Covariate-balance diagnostics                              |
| - Propensity-model diagnostics                              |
| - Refutation results and analytical insights                |
+-----------------------------+-------------------------------+
                              |
                              v
                    BUSINESS INSIGHTS
```

---

## Mathematical Formulation

### 1. The Fundamental Problem of Causal Inference

In observational data, the same observation cannot simultaneously reveal its outcome under treatment and its outcome without treatment.

The observed outcome is represented as:

$$
Y_i = T_iY_i(1) + (1-T_i)Y_i(0)
$$

Where:

- \(Y_i\) is the observed outcome.
- \(T_i \in \{0,1\}\) is the treatment assignment.
- \(Y_i(1)\) is the potential outcome under treatment.
- \(Y_i(0)\) is the potential outcome without treatment.

Causal inference estimates the difference between these potential outcomes under the assumptions required by the selected method.

### 2. Custom Hajek-Normalized IPTW Estimator

The Inverse Probability of Treatment Weighting method estimates propensity scores to account for differences between treatment and control groups.

The propensity score is defined as:

$$
e(X_i)=P(T_i=1\mid X_i)
$$

Where \(X_i\) represents the selected confounders.

The treatment and control weights are:

$$
w_i^{(1)}=\frac{T_i}{e(X_i)}
$$

$$
w_i^{(0)}=\frac{1-T_i}{1-e(X_i)}
$$

The project uses a Hajek-normalized estimator:

$$
\widehat{ATE} =
\frac{\sum_{i=1}^{N}\frac{T_iY_i}{e(X_i)}}
{\sum_{i=1}^{N}\frac{T_i}{e(X_i)}}
-
\frac{\sum_{i=1}^{N}\frac{(1-T_i)Y_i}{1-e(X_i)}}
{\sum_{i=1}^{N}\frac{1-T_i}{1-e(X_i)}}
$$

Where:

- \(T_i\) represents treatment assignment.
- \(Y_i\) represents the observed outcome.
- \(e(X_i)\) is the propensity score estimated using logistic regression.
- The first term represents the weighted treatment-group mean.
- The second term represents the weighted control-group mean.

The difference between these weighted means estimates the Average Treatment Effect under the required causal assumptions.

The documented implementation clips propensity scores to the interval \([0.01,0.99]\) to limit extreme weights. This does not, by itself, guarantee that the positivity assumption is satisfied.

### 3. Bootstrap Statistical Inference

The engine uses bootstrap resampling to estimate uncertainty in the treatment-effect estimate.

The documented configuration uses \(B=50\) bootstrap samples drawn with replacement.

The bootstrap standard error is calculated as:

$$
\widehat{SE}(\widehat{ATE}) =
\sqrt{
\frac{1}{B-1}
\sum_{b=1}^{B}
\left(
\widehat{ATE}^{*(b)}-\overline{ATE}^{*}
\right)^2
}
$$

The normal-approximation 95% confidence interval is:

$$
95\%\ CI =
\left[
\widehat{ATE}-1.96\widehat{SE},
\widehat{ATE}+1.96\widehat{SE}
\right]
$$

The two-sided p-value under a normal approximation is:

$$
p =
2\left[
1-\Phi\left(
\frac{|\widehat{ATE}|}{\widehat{SE}}
\right)
\right]
$$

Where \(\Phi\) represents the standard normal cumulative distribution function.

A p-value below 0.05 is commonly treated as statistically significant under the specified test. Statistical significance alone does not establish causality.

---

## Benchmark Comparison

The following comparison reproduces the benchmark figures provided in the project documentation. The baseline definitions and evaluation procedure should be checked against the actual experiment code before interpreting these figures as independently verified improvements.

| Metric / Evaluation Criterion | Baseline | Proposed Causal Engine | Reported Comparison |
| :--- | :---: | :---: | :--- |
| Covariate Balance (SMD) | `0.342` | `0.021` | 93.8% reduction |
| Benchmark Estimation Error (PEHE) | `0.1420` | `0.0149` | 89.5% reduction |
| Placebo Refutation Effect | `0.0480` | `0.0020` | 24x smaller |
| Average Refutation Pass Rate | `78.4%` | `83.3%` | +4.9 percentage points |
| Average Confounder-Related AUC | `0.612` | `0.726` | 18.6% increase |
| Autonomous Causal Discovery | Manual | Automated | Automated variable-impact ranking |

The reported improvements depend on using comparable datasets, treatment definitions, evaluation metrics, and validation procedures for both methods.

---

## Empirical Results Across 6 Real-World Test Cases

The project documentation reports experiments using the Brazilian E-Commerce Public Dataset by Olist, which contains multiple relational tables covering orders, products, payments, shipping, and customer reviews.

The following results preserve the six test cases and their reported values.

| Run | Business Question | Sample Size | Estimated ATE | 95% Confidence Interval | Covariate Balance (SMD) | Propensity AUC | Refutation Pass Rate | Key Empirical Finding |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | Installments → Total Payment Value (Control: `payment_sequential`) | 103,886 | `+59.7865` | `[-20.60, +140.17]` | `0.001` | `0.500` | `100%` | High basket variance; the interval includes zero, so the direction is uncertain. |
| 2 | Freight Value → Review Rating (Controls: `price`, `product_weight_g`) | 112,372 | `-0.1326` | `[-0.1647, -0.1005]` | `0.021` | `0.716` | `100%` | Negative estimated effect with a confidence interval excluding zero. |
| 3 | Product Photos → Review Rating (Control: `price`) | 110,774 | `+0.0745` | `[+0.0579, +0.0911]` | `0.003` | `0.504` | `100%` | Positive estimated effect with reported covariate balance below 0.10. |
| 4 | Description Length → Review Rating (Controls: `price`, `product_photos_qty`) | 110,774 | `+0.0451` | `[+0.0249, +0.0653]` | `0.015` | `0.604` | `50%` | Variable-impact analysis reports greater standardized impact for product photos than description length. |
| 5 | Product Weight → Freight Cost (Control: `price`) | 112,632 | `+4.6150` | `[+4.1750, +5.0550]` | `0.019` | `0.736` | `50%` | Positive estimated relationship between product weight and freight cost. |
| 6 | Product Length → Freight Cost (Control: `product_weight_g`) | 112,632 | `-2.4567` | `[-3.0807, -1.8327]` | `0.024` | `0.825` | `50%` | Variable-impact analysis reports a substantially greater impact score for product weight than product length. |

These values are reported experimental outputs, not independently reproduced results. The causal interpretations depend on treatment definitions, confounder selection, covariate balance, overlap, and other identification assumptions.

A refutation pass rate describes the proportion of configured tests that meet their acceptance criteria. It should not be interpreted as a direct probability that a causal estimate is correct.

---

## Project Structure

```text
Causal-LLM-Engine/
|
├── ai_engine/
|   ├── __init__.py
|   ├── gemini_router.py
|   └── schema_embedder.py
|
├── database/
|   ├── __init__.py
|   ├── db_setup.py
|   └── olist.db
|
├── math_engine/
|   ├── __init__.py
|   └── causal_model.py
|
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

### File Descriptions

- `ai_engine/gemini_router.py`: Processes natural-language questions and extracts structured analysis parameters.
- `ai_engine/schema_embedder.py`: Creates schema embeddings and performs semantic table retrieval.
- `database/db_setup.py`: Initializes the SQLite database and configures database safeguards.
- `database/olist.db`: Stores the imported Olist relational data.
- `math_engine/causal_model.py`: Implements propensity weighting, ATE estimation, statistical evaluation, and configured refutation tests.
- `app.py`: Runs the Streamlit analytics dashboard.
- `requirements.txt`: Lists the project's Python dependencies.
- `.env`: Stores local environment configuration.
- `README.md`: Documents the project and its methodology.

The structure above should be adjusted if the actual repository contains different files or modules.

---

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Viper777/Causal-LLM-Engine.git
cd Causal-LLM-Engine
```

Replace the repository URL if your current GitHub repository uses a different address.

### 2. Set Up Virtual Environment

**Linux/macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

Ensure the environment variable name matches the one used by the application.

Do not commit your actual API key to GitHub.

### 5. Build Database (One-Time Setup)

```bash
python database/db_setup.py
```

Ensure the Olist source dataset is available at the location expected by the database initialization script.

### 6. Run the Application

```bash
streamlit run app.py
```

Open the local address displayed in the terminal, usually:

`http://localhost:8501`

---

## Academic Citation and References

```bibtex
@article{kiciman2023causal,
  title={Causal Reasoning and Large Language Models: Opening a New Frontier for Causality},
  author={Kiciman, Emre and Ness, Robert and Sharma, Amit and Tan, Chenhao},
  journal={arXiv preprint arXiv:2304.15044},
  year={2023}
}

@article{sharma2020dowhy,
  title={DoWhy: An End-to-End Library for Causal Inference},
  author={Sharma, Amit and Kiciman, Emre},
  journal={arXiv preprint arXiv:2011.04216},
  year={2020}
}
```

Additional resources:

- DoWhy Documentation: https://www.pywhy.org/dowhy/
- Olist Brazilian E-Commerce Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Google AI for Developers: https://ai.google.dev/
- Streamlit Documentation: https://docs.streamlit.io/
