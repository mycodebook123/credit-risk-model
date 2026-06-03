import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating aggregate features per customer")
    agg = df.groupby('CustomerId').agg(
        total_transaction_amount=('Amount', 'sum'),
        avg_transaction_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_transaction_amount=('Amount', 'std'),
        total_value=('Value', 'sum'),
        avg_value=('Value', 'mean'),
    ).reset_index()
    agg['std_transaction_amount'] = agg['std_transaction_amount'].fillna(0)
    return agg


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extracting time features")
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding categorical features")
    df = df.copy()
    cat_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating RFM metrics")
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerId').agg(
        recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        frequency=('TransactionId', 'count'),
        monetary=('Amount', 'sum')
    ).reset_index()
    return rfm


def assign_risk_label(rfm: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    logger.info("Clustering customers to assign risk labels")
    rfm = rfm.copy()

    # Cap extreme outliers before scaling
    for col in ['recency', 'frequency', 'monetary']:
        upper = rfm[col].quantile(0.99)
        lower = rfm[col].quantile(0.01)
        rfm[f'{col}_capped'] = rfm[col].clip(lower, upper)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency_capped', 'frequency_capped', 'monetary_capped']])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
    logger.info(f"Cluster summary:\n{cluster_summary}")

    # High risk = highest recency (disengaged), lowest frequency, lowest monetary
    # We score each cluster: high recency is bad, low frequency is bad, low monetary is bad
    high_risk_cluster = (
        cluster_summary['recency'] * 1.0
        - cluster_summary['frequency'] * 0.5
        - cluster_summary['monetary'] * 0.0001
    ).idxmax()

    logger.info(f"High-risk cluster identified: {high_risk_cluster}")
    logger.info(f"High-risk cluster profile:\n{cluster_summary.loc[high_risk_cluster]}")

    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

    # Drop capped columns
    rfm = rfm.drop(columns=['recency_capped', 'frequency_capped', 'monetary_capped'])
    return rfm[['CustomerId', 'recency', 'frequency', 'monetary', 'is_high_risk']]


def build_model_dataset(raw_filepath: str, output_filepath: str) -> pd.DataFrame:
    df = load_data(raw_filepath)

    # Time features on full transaction data
    df = extract_time_features(df)
    df = encode_categorical(df)

    # Aggregate to customer level
    agg = create_aggregate_features(df)

    # Time feature aggregates
    time_agg = df.groupby('CustomerId').agg(
        avg_hour=('transaction_hour', 'mean'),
        avg_day=('transaction_day', 'mean'),
        avg_month=('transaction_month', 'mean'),
    ).reset_index()

    # RFM + risk label
    rfm = calculate_rfm(df)
    rfm = assign_risk_label(rfm)

    # Merge everything
    final = agg.merge(time_agg, on='CustomerId', how='left')
    final = final.merge(rfm, on='CustomerId', how='left')

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    feature_cols = [c for c in final.columns if c not in ['CustomerId', 'is_high_risk']]
    final[feature_cols] = imputer.fit_transform(final[feature_cols])

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    final.to_csv(output_filepath, index=False)
    logger.info(f"Saved processed dataset to {output_filepath} with shape {final.shape}")
    logger.info(f"Risk label distribution:\n{final['is_high_risk'].value_counts()}")

    return final


if __name__ == "__main__":
    build_model_dataset(
        raw_filepath="data/raw/data.csv",
        output_filepath="data/processed/processed_data.csv"
    )