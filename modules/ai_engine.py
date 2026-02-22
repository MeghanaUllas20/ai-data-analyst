def generate_ai_insights(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    insights = []

    # Identifier detection
    for col in df.columns:
        if df[col].nunique() == len(df):
            insights.append(f"'{col}' has unique values and may behave like an identifier.")

    # Sequential detection
    for col in numeric_cols:
        if df[col].is_monotonic_increasing:
            insights.append(f"'{col}' appears sequential and may introduce modeling leakage.")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        worst = missing.sort_values(ascending=False).index[0]
        insights.append(f"'{worst}' contains the highest missing values.")

    # Correlation intelligence
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        corr.values[[range(len(corr))]*2] = 0
        pair = corr.unstack().sort_values(ascending=False).idxmax()
        insights.append(f"Strong relationship detected between '{pair[0]}' and '{pair[1]}'.")

    if not insights:
        insights.append("Dataset appears structurally clean with no major anomalies detected.")

    return insights