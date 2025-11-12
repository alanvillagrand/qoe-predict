import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw metrics into QoE features."""
    df['latency_quality_ratio'] = df['points_delivered'] / (df['latency_ms'] + 1e-6)
    df['stall_density'] = df['stall_time'] / (df['session_duration'])
    df['spatial_completeness'] = df['points_delivered'] / df['points_total']
    df['pcqm_var'] = df['pcqm'].rolling(5, min_periods=1).var()
    return df
