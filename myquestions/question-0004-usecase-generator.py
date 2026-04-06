import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def generar_caso_de_uso_predecir_consumo_energetico():
    n = random.randint(300, 800)
    horas = np.arange(n)
    temperatura  = 20 + 10 * np.sin(2 * np.pi * horas / 24) + np.random.normal(0, 1.5, n)
    ocupacion    = np.random.randint(0, 200, n).astype(float)
    humedad      = np.random.uniform(30, 80, n)
    luminosidad  = np.random.uniform(0, 1000, n)
    consumo = (2.5 * temperatura + 0.8 * ocupacion + 0.3 * humedad
               + 0.01 * luminosidad + np.random.normal(0, 20, n))
    n_out = random.randint(10, 30)
    idx_out = random.sample(range(n), n_out)
    consumo[idx_out] += np.random.choice([-1, 1], n_out) * np.random.uniform(300, 600, n_out)

    df = pd.DataFrame({
        "hora_del_dia":           horas % 24,
        "dia_semana":             (horas // 24) % 7,
        "temperatura_exterior":   temperatura.round(2),
        "ocupacion_personas":     ocupacion,
        "humedad_relativa":       humedad.round(2),
        "luminosidad_lux":        luminosidad.round(2),
        "consumo_kwh":            consumo.round(3),
    })
    target_col = "consumo_kwh"

    input_data = {"df": df.copy(), "target_col": target_col}

    # --- Replicar lógica de predecir_consumo_energetico ---
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Split 75/25 ANTES de IsolationForest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Escalar con MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # IsolationForest solo sobre train
    iso = IsolationForest(contamination=0.05, random_state=42)
    mask = iso.fit_predict(X_train_scaled) == 1
    n_outliers = int((~mask).sum())
    X_train_clean = X_train_scaled[mask]
    y_train_clean = y_train[mask]

    # Entrenar regresor
    reg = GradientBoostingRegressor(random_state=42)
    reg.fit(X_train_clean, y_train_clean)

    # Predecir sobre test
    y_pred = reg.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    output_data = {
        "predicciones":          y_pred,
        "rmse":                  rmse,
        "r2":                    r2,
        "n_outliers_eliminados": n_outliers,
    }

    print(f"Registros: {n} | Outliers inyectados: {n_out} | Target: 'consumo_kwh'")
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_predecir_consumo_energetico()
    print("\nInput (head):")
    print(entrada["df"].head())
    print("\nOutput esperado:")
    print(f"  RMSE: {salida['rmse']:.4f}")
    print(f"  R2:   {salida['r2']:.4f}")
    print(f"  Outliers eliminados: {salida['n_outliers_eliminados']}")
    print(f"  Predicciones shape: {salida['predicciones'].shape}")
