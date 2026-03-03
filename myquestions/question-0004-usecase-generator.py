import pandas as pd
import numpy as np
import random


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
        "hora_del_dia":       horas % 24,
        "dia_semana":         (horas // 24) % 7,
        "temperatura_exterior": temperatura.round(2),
        "ocupacion_personas": ocupacion,
        "humedad_relativa":   humedad.round(2),
        "luminosidad_lux":    luminosidad.round(2),
        "consumo_kwh":        consumo.round(3),
    })
    print(f"Registros: {n} | Outliers inyectados: {n_out} | Target: 'consumo_kwh'")
    return df, "consumo_kwh"


if __name__ == "__main__":
    df_input, target = generar_caso_de_uso_predecir_consumo_energetico()
    print("\nDatos de entrada (primeras filas):")
    print(df_input.head())