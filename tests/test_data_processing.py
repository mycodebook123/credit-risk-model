import pandas as pd
from src.data_processing import create_aggregate_features, extract_time_features, calculate_rfm


def test_create_aggregate_features_columns():
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionId': ['T1', 'T2', 'T3'],
        'Amount': [100, 200, 300],
        'Value': [100, 200, 300],
    })
    result = create_aggregate_features(df)
    expected_cols = ['CustomerId', 'total_transaction_amount', 'avg_transaction_amount',
                     'transaction_count', 'std_transaction_amount', 'total_value', 'avg_value']
    assert list(result.columns) == expected_cols


def test_create_aggregate_features_values():
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1'],
        'TransactionId': ['T1', 'T2'],
        'Amount': [100, 200],
        'Value': [100, 200],
    })
    result = create_aggregate_features(df)
    assert result.loc[0, 'total_transaction_amount'] == 300
    assert result.loc[0, 'transaction_count'] == 2
    assert result.loc[0, 'avg_transaction_amount'] == 150.0


def test_extract_time_features_columns():
    df = pd.DataFrame({
        'TransactionStartTime': ['2018-11-15T02:18:49Z', '2018-12-20T10:00:00Z']
    })
    result = extract_time_features(df)
    for col in ['transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year']:
        assert col in result.columns


def test_calculate_rfm_columns():
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionId': ['T1', 'T2', 'T3'],
        'Amount': [100, 200, 300],
        'TransactionStartTime': ['2018-11-15T02:18:49Z', '2018-11-20T10:00:00Z', '2018-10-01T10:00:00Z']
    })
    result = calculate_rfm(df)
    assert 'recency' in result.columns
    assert 'frequency' in result.columns
    assert 'monetary' in result.columns
    assert len(result) == 2