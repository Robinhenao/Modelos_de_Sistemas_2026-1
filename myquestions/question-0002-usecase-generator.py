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

    input_data = {"df": df.copy()}

    # --- Replicar lógica de calcular_metricas_mensuales ---
    df_work = df.copy()
    df_work["mes"] = df_work["fecha"].dt.to_period("M")

    grouped = df_work.groupby(["mes", "categoria"])["importe"].agg(
        suma_total="sum",
        media="mean",
        maximo="max",
        minimo="min",
        n_transacciones="count",
    ).reset_index()

    # % sobre total del mes
    total_mes = grouped.groupby("mes")["suma_total"].transform("sum")
    grouped["pct_sobre_total_mes"] = (grouped["suma_total"] / total_mes * 100).round(2)

    # Ordenar por mes asc, suma_total desc
    grouped = grouped.sort_values(["mes", "suma_total"], ascending=[True, False]).reset_index(drop=True)

    output_data = grouped

    print(f"Transacciones: {n} | Categorías: {categorias} | Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_calcular_metricas_mensuales()
    print("\nInput (head):")
    print(entrada["df"].head())
    print("\nOutput esperado (head):")
    print(salida.head())
