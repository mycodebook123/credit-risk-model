# Credit Risk Probability Model

An end-to-end credit risk modeling pipeline for Bati Bank's buy-now-pay-later service, built on transaction data from the Xente eCommerce platform.

## Project Structure
credit-risk-model/
├── .github/workflows/ci.yml    # CI/CD pipeline
├── data/raw/                   # Raw data (not committed)
├── data/processed/             # Processed data (not committed)
├── notebooks/eda.ipynb         # Exploratory analysis
├── src/
│   ├── data_processing.py      # Feature engineering pipeline
│   ├── train.py                # Model training + MLflow tracking
│   ├── predict.py              # Inference
│   └── api/
│       ├── main.py             # FastAPI application
│       └── pydantic_models.py  # Request/response schemas
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

## Setup

```bash
pip install -r requirements.txt
```

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord influence the need for an interpretable and well-documented model?

The Basel II Capital Accord requires banks to hold capital reserves proportional to their risk exposure. To calculate this, banks must demonstrate to regulators that their risk models are accurate, transparent, and auditable. This directly creates three requirements for any credit scoring model:

- **Interpretability**: Regulators and internal risk officers must be able to explain why a customer was denied credit. A black-box model that cannot produce a reason code fails this requirement. This is why Logistic Regression with Weight of Evidence (WoE) transformations remains the industry standard — each coefficient has a direct, explainable relationship to the outcome.
- **Documentation**: Every modeling choice — feature selection, proxy variable definition, threshold selection — must be documented and defensible. Basel II Pillar 2 requires internal capital adequacy assessment, meaning the bank's own risk assessment process must be rigorous and reproducible.
- **Validation**: Models must be periodically validated against new data. MLflow experiment tracking directly supports this by maintaining a full audit trail of every model version, its parameters, and its performance metrics.

### 2. Why is a proxy variable necessary, and what business risks does it introduce?

The Xente transaction dataset contains no direct "default" label — there is no record of customers failing to repay loans because this is a buy-now-pay-later product being launched for the first time. A proxy variable is therefore necessary to create a supervised learning target from unsupervised behavioral data.

We use RFM (Recency, Frequency, Monetary) analysis to engineer this proxy: customers who transact infrequently, have long gaps since their last transaction, and spend little are labeled "high risk" (is_high_risk = 1). This is grounded in the empirical observation that disengaged customers are more likely to default on credit obligations.

**Business risks introduced by proxy-based prediction:**
- **Label noise**: The proxy may mislabel some customers. A customer who stopped transacting because they switched to a competitor is not necessarily a credit risk.
- **Concept drift**: The behavioral patterns used to define the proxy may not remain stable over time, requiring periodic revalidation.
- **Regulatory scrutiny**: Proxy labels must be explicitly documented and justified to regulators. Using a proxy without disclosure could constitute a compliance violation.
- **Self-fulfilling bias**: If the model denies credit to proxy-labeled customers, it never observes their actual repayment behavior, making it impossible to validate whether the proxy was correct.

### 3. Trade-offs between interpretable and high-performance models in a regulated context

| Criterion | Logistic Regression + WoE | Gradient Boosting (XGBoost) |
|-----------|--------------------------|------------------------------|
| Interpretability | High — coefficients are directly explainable | Low — requires SHAP for post-hoc explanation |
| Regulatory acceptance | High — industry standard for scorecards | Moderate — acceptable with SHAP documentation |
| Predictive performance | Moderate | High |
| Auditability | Easy — monotonic WoE bins are inspectable | Hard — thousands of trees are opaque |
| Development time | Low | High (hyperparameter tuning) |
| Maintenance | Simple | Complex — more prone to overfitting |

**Recommendation**: In a regulated context like Bati Bank, a Logistic Regression baseline with WoE features provides the compliance foundation. A Gradient Boosting model can be developed in parallel for performance benchmarking, but the deployed scoring model should be the interpretable version unless SHAP-based documentation is formally accepted by the risk committee.

## How to Run

```bash
# Feature engineering
python src/data_processing.py

# Model training
python src/train.py

# Start API
uvicorn src.api.main:app --reload

# Run tests
pytest tests/ -v
```

## Data Pipeline

Raw transaction data → RFM feature engineering → K-Means clustering → is_high_risk label → sklearn Pipeline → model training → MLflow registry → FastAPI deployment