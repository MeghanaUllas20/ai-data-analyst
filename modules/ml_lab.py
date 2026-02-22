from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def run_auto_ml(df, target):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    X = df[numeric_cols].drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    r2 = model.score(X_test, y_test)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return r2, importance