import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score

def detectar_fraude(df: pd.DataFrame) -> dict:

    df = df.copy()

    if df['pais'].dtype == object:
        le = LabelEncoder()
        df['pais'] = le.fit_transform(df['pais'].astype(str))

    features = ['monto', 'hora', 'pais', 'frecuencia_transacciones']
    X = df[features].values
    y = df['fraude'].values

    n_legit = int((y == 0).sum())
    n_fraud = int((y == 1).sum())
    ratio   = round(n_fraud / n_legit, 4) if n_legit > 0 else float('inf')
    desbalance = {
        'legítimas':             n_legit,
        'fraudulentas':          n_fraud,
        'ratio_fraude/legítima': ratio,
        'desbalanceado':         ratio < 0.2
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=200, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    modelo.fit(X_train, y_train)

    preds = modelo.predict(X_test)
    cm    = confusion_matrix(y_test, preds)
    f1    = round(f1_score(y_test, preds, zero_division=0), 4)
    f1_w  = round(f1_score(y_test, preds, average='weighted', zero_division=0), 4)

    return {
        'modelo':           modelo,
        'confusion_matrix': cm,
        'f1':               f1,
        'f1_weighted':      f1_w,
        'desbalance':       desbalance,
        'predicciones':     preds
    }