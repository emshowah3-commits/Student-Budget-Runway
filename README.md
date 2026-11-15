# Student Budget Runway 

**Track**: AI + Finance. Prototype only, not financial advice.

**Question**: Can a simple model sort a student's messy bank lines into
spending categories, and can a forecast built on those categories warn
a student at least 3 days before they run out of money ahead of payday,
better than a "keeps spending as usual" rule?

**Solutions**

**Datasets & Credits**:
synthetic Canadian student transaction sets
Statistics Canada benchmarks

**Disclaimer & Privacy**
Educational tool, not financial advice. Uses synthetic data only.

**AI tools used**
CLaude and gemini

**Definitions**
- Run out: balance falls below $0 before the next payday.
- Early: at least 3 days before the day it happens.
- Sorting baseline: always predict the most common category.
- Forecast baseline: assume average daily spending continues.

**Metrics**
- Sorting: accuracy and macro-F1 on 150 to 200 test lines I wrote.
- Forecast: warning precision and recall, and median days of warning.

Out of scope: bank logins, real user data, receipts, voice, mobile app.
