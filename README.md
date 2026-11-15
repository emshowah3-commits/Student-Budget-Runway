# Runway: Student Budget Runway

**Track:** AI + Finance | **Status:** educational prototype, not financial advice | **Data:** synthetic only

Runway sorts a student's messy bank lines with a trained classifier, then uses a category-aware Monte Carlo forecast to warn them at least 3 days before they would run out of money before payday. It was tested against a "spend as usual" rule on synthetic data.

## The problem

A student has rent due on the 25th and payday on the 28th. Their balance looks fine today, but in between it hits zero. Most budgeting tools show where money went. Runway shows whether the money will last until payday, early enough to do something about it.

## What it does

1. **Sorts the bank lines.** A TF-IDF + logistic regression model labels each transaction (Groceries, Housing, Restaurants, Utilities, and so on, 15 categories).
2. **Splits spending into fixed bills and flexible spending.** Housing, utilities, insurance, subscriptions, education and fees are treated as fixed bills. This step is plain code, not AI.
3. **Looks ahead to payday.** It projects the balance day by day, runs 1,000 simulated futures drawn from the student's own recent spending, and reports:
   - the chance of running out before payday
   - a safe-to-spend amount per day
   - the "smallest fix": the smallest cut to flexible spending that brings the risk under the chosen level
4. **Lets the user correct labels.** If the classifier gets a line wrong, fixing the category updates the forecast straight away.

## Run it locally

```
git clone <this repository's URL>
cd <repository folder>
pip install -r requirements.txt
streamlit run app.py
```

- The app needs `models/categorizer.joblib`. It is included in the repo. To recreate it, run `notebooks/01_categorization.ipynb`.
- With no upload, the app loads the synthetic demo students in `data/demo/demo_student.csv`. Select `sam` to see a warning.
- To use your own file, upload a CSV with the columns `date`, `description` and `amount`. The columns `user_id`, `running_balance` and `days_to_payday` are optional.
- The notebooks run top to bottom with Restart and Run All.

## Repository layout

```
app.py                      Streamlit app
src/runway.py               forecasting code: fixed vs flexible spending, simulation, safe-to-spend, smallest fix
models/categorizer.joblib   trained sorting model
notebooks/                  01_categorization (sorting), 02_forecast (warning evaluation)
data/demo/                  synthetic demo students
data/labeled/               my 160 hand-labelled messy test lines
results/                    result tables and charts
docs/                       write-up and sources
```

## Definitions

- **Run out:** the balance falls below the safety buffer before the next payday.
- **Early warning:** raised at least 3 days before the projected run-out.
- **Sorting baseline:** always predict the most common category.
- **Forecast baseline ("spend as usual"):** project the balance using the 30-day average daily spending, without separating fixed bills from flexible spending.

## Data and credits

- **Training and test ledgers:** synthetic Canadian student ledgers (`canadian_student_bank_forecasting_dataset.csv`). [FILL: say exactly how they were generated (your script or the AI tool used) and whether you have a license.] Spending levels were informed by rough public benchmarks from Statistics Canada, CMHC, FCAC, Payments Canada and CIHI. No data was taken directly from those sources.
- **My own test set:** 160 messy bank lines in Canadian, UK, Indian and US formats, drafted with Claude's help and reviewed and labelled by me.
- **Libraries:** scikit-learn, pandas, NumPy, Streamlit, joblib.
- **AI tools:** Claude (planning, writing and debugging code, drafting the test lines) and Gemini (code structure review). I reviewed everything and can explain every part.

## Results

All results come from synthetic data, so read them as a prototype check, not proof.

**Sorting bank lines**

| Test | Accuracy |
|---|---|
| Always guess the most common category | 35.2% |
| TF-IDF + logistic regression, 30% split of the training ledgers | 99.7% (macro-F1 99.5%). This is inflated, because the generated ledgers repeat the same templates. |
| TF-IDF + logistic regression, my 160 messy lines | [FILL: accuracy on the lines that do not appear in the training data, with the number of lines] |

**Early-warning forecast** (522 evaluation days, 21 actual run-out cases)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Spend as usual | 0.280 | 0.667 | 0.394 |
| Category-aware (Runway) | 0.471 | 0.762 | 0.582 |


With only 21 real cases, the gap is encouraging but not conclusive. The forecast and the data generator share assumptions about fixed and flexible spending, so real bank data could give lower scores.

## Where it fails

- **Mislabelled student purchases.** A campus bookstore line can be labelled Restaurants, a ride home Housing, and a campus café Groceries. Because fixed bills and flexible spending are split by category, these errors flow into the forecast. The in-app label review exists for this.
- **False alarms and misses.** A short dip right after rent can trigger a warning even though the student recovers, and a spending spike in the last two days before payday can be missed.
- **Irregular income.** Co-op terms and one-off e-transfers make payday timing hard to infer, so the app falls back to a payday you enter.
- **Fixed bills are guessed** from repeated payments, so a one-off or changed bill can be missed.

## Privacy

When you run the app on your own machine, parsing, classification and simulation happen locally in memory. The demo uses only synthetic data. Please don't upload real bank statements to a copy of this app hosted on someone else's server.

## Out of scope

Live bank linking, multi-account workflows, multi-currency support, and automatic actions such as pausing subscriptions.

## Disclaimer

Runway is an educational prototype built for a hackathon. It is not financial advice.
