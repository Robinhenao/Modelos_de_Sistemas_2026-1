import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


def generar_caso_de_uso_entrenar_clasificador_clientes():
    n = random.randint(200, 600)
    segmentos = random.choice([
        ["bajo", "medio", "alto"],
        ["riesgo_bajo", "riesgo_alto"],
        ["A", "B", "C", "D"],
    ])
    regiones = random.sample(["norte", "sur", "este", "oeste", "centro"], k=random.randint(3, 5))
    canales = random.sample(["online", "tienda", "telefono", "app"], k=random.randint(2, 4))
    df = pd.DataFrame({
        "edad":             np.random.randint(18, 75, n),
        "ingresos_anuales": np.random.uniform(15000, 120000, n).round(2),
        "num_productos":    np.random.randint(1, 10, n),
        "antiguedad_meses": np.random.randint(1, 120, n),
        "region":           random.choices(regiones, k=n),
        "canal_preferido":  random.choices(canales, k=n),
        "segmento":         random.choices(segmentos, k=n),
    })
    target_col = "segmento"

    input_data = {"df": df.copy(), "target_col": target_col}

    # --- Replicar lógica de entrenar_clasificador_clientes ---
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred),
    }

    output_data = (pipeline, metrics)

    print(f"Clientes: {n} | Target: 'segmento' {segmentos} | Columnas: {list(df.columns)}")
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_entrenar_clasificador_clientes()
    pipeline, metrics = salida
    print("\nInput (head):")
    print(entrada["df"].head())
    print("\nMétricas esperadas:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
