# Student Budget Runway

**Track**: AI + Finance | **Status**: Prototype (Educational, not financial advice)

## Research Question
Can a simple trained classifier (TF-IDF + Logistic Regression) accurately categorize a student's messy bank transactions, and can a category-aware forecast warn a student at least 3 days before they run out of money before payday, outperforming a naive "spend-as-usual" baseline?

---

## Solution Overview
1. **Transaction Categorization**: Cleans and classifies raw bank transaction descriptions into 15 spending categories using TF-IDF vectorization and a Logistic Regression classifier.
2. **Deterministic Runway Simulation**: Separates fixed recurring bills (rent, utilities, subscriptions, insurance) from flexible discretionary spending.
3. **Pre-Payday Warning Engine**: Runs probabilistic Monte Carlo simulations to estimate the exact risk of running out of money before payday and computes the "smallest fix" (minimal flexible spending cut) needed to avoid overdraft.

---

## Datasets & Credits
- **Data Calibration**: Synthetic Canadian student banking dataset calibrated to official Canadian institutional benchmarks:
  * Statistics Canada (Survey of Household Spending [SHS], Food Services and Drinking Places Survey [FSDSP], Labour Force Survey [LFS])
  * Canada Mortgage and Housing Corporation (CMHC - Rental Market Report)
  * Financial Consumer Agency of Canada (FCAC)
  * Payments Canada
  * Canadian Institute for Health Information (CIHI)
- **Libraries**: `scikit-learn`, `pandas`, `numpy`, `streamlit`, `joblib`
- **AI & Collaboration Tools**: Built with Python and assisted by LLM tooling (Claude & Gemini) for code structuring and documentation review, alongside custom Canadian, UK, and Indian messy test transaction benchmarks.

---

## Definitions
- **Run Out**: Account balance falls below the safety buffer before the next payday.
- **Early Warning**: Triggered at least 3 days prior to an anticipated overdraft event.
- **Sorting Baseline**: Always predicting the most common category in the dataset.
- **Forecast Baseline ("Spend as Usual")**: Projecting future balances using simple 30-day moving-average daily spending without separating fixed bills.

---

## Evaluation & Results
- **Transaction Sorting**: 
  * Baseline ("always guess most common"): **35.2% accuracy**
  * TF-IDF + Logistic Regression (30% test split): **99.7% accuracy** (Macro-F1: **99.5%**)
  * TF-IDF + Logistic Regression (Custom messy test lines): **100.0% accuracy**
- **Early-Warning Forecast**: 
  * Tested on **522 evaluation days** across **28 synthetic Canadian student ledgers** with **34 actual run-out cases**.
  * **Category-Aware Forecast**: Flagged 35 students; correct 30 times (**Precision: 0.86, Recall: 0.88**).
  * **Spend-as-Usual Baseline**: Flagged 65 students; correct 31 times (**Precision: 0.48, Recall: 0.91**).
  * *Note: With a limited number of actual run-out cases in the synthetic test set, these results serve as an encouraging prototype proof-of-concept rather than clinical proof.*

---

## Installation & Running Locally

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/emshowah3-commits/student-budget-runway.git](https://github.com/emshowah3-commits/student-budget-runway.git)
   cd student-budget-runway