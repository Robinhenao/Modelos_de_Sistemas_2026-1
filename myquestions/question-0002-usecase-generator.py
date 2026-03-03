import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


def generar_caso_de_uso_calcular_metricas_mensuales():
    categorias = random.sample(
        ["alimentación", "transporte", "ocio", "salud", "tecnología", "hogar"],
        k=random.randint(3, 5),
    )
    n = random.randint(60, 200)
    start = datetime(2024, 1, 1)
    fechas = [start + timedelta(days=random.randint(0, 364)) for _ in range(n)]
    df = pd.DataFrame({
        "fecha": pd.to_datetime(fechas),
        "categoria": random.choices(categorias, k=n),
        "importe": [round(random.uniform(5, 800), 2) for _ in range(n)],
    })
    print(f"Transacciones: {n} | Categorías: {categorias} | Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    return df

if __name__ == "__main__":
    df_input = generar_caso_de_uso_calcular_metricas_mensuales()
    print("\ndf_input:")
    print(df_input.head())
    