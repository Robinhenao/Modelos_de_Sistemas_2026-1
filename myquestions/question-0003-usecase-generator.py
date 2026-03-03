import pandas as pd
import numpy as np
import random


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
    print(f"Clientes: {n} | Target: 'segmento' {segmentos} | Columnas: {list(df.columns)}")
    return df, "segmento"

if __name__ == "__main__":
    df_input, target = generar_caso_de_uso_entrenar_clasificador_clientes()
    print("\nDatos de entrada (primeras filas):")
    print(df_input.head())