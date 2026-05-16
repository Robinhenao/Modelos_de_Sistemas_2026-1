import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def clasificar_biomoleculas(dataset: pd.DataFrame, features: list, target: str) -> dict:

    cols_momento = ['momento_x', 'momento_y', 'momento_z']
    df_limpio = dataset.dropna(subset=cols_momento + [target]).copy()
    mascara = (
        (df_limpio['momento_x'] >= 0) &
        (df_limpio['momento_y'] >= 0) &
        (df_limpio['momento_z'] >= 0)
    )
    df_limpio = df_limpio[mascara].reset_index(drop=True)

    if df_limpio.empty:
        raise ValueError("El DataFrame quedó vacío tras la limpieza.")

    mx = df_limpio['momento_x'].values
    my = df_limpio['momento_y'].values
    mz = df_limpio['momento_z'].values
    numerador   = (mx - my)**2 + (my - mz)**2 + (mz - mx)**2
    denominador = (mx + my + mz)**2
    df_limpio['indice_asimetria'] = np.where(denominador > 0, numerador / denominador, 0.0)

    le = LabelEncoder()
    y_encoded = le.fit_transform(df_limpio[target])

    features_model = features + ['indice_asimetria']
    X = df_limpio[features_model].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    modelo = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)

    return {
        'df_procesado': df_limpio,
        'target_vector': y_encoded,
        'clases_mapeadas': dict(zip(le.classes_, le.transform(le.classes_)))
    }
