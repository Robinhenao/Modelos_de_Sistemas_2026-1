import pandas as pd
import numpy as np
import random


def generar_caso_de_uso_limpiar_dataset_ventas():
    n = random.randint(30, 100)
    productos = random.sample(
        ["  Laptop", "TABLET", "monitor ", "Teclado", "RATON", "auriculares"],
        k=random.randint(3, 6),
    )
    df = pd.DataFrame({
        "producto": random.choices(productos, k=n),
        "precio": [round(random.uniform(10, 2000), 2) if random.random() > 0.15 else None for _ in range(n)],
        "unidades": [random.randint(1, 50) if random.random() > 0.1 else None for _ in range(n)],
        "fecha": [
            f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            if random.random() > 0.05 else None
            for _ in range(n)
        ],
    })
    dupes = df.sample(n=random.randint(2, 6))
    df = pd.concat([df, dupes], ignore_index=True)

    input_data = {"df": df.copy()}

    # --- Replicar lógica de limpiar_dataset_ventas ---
    df_out = df.copy()

    # 1. Eliminar duplicados
    df_out = df_out.drop_duplicates()

    # 2. Imputar nulos numéricos con mediana
    for col in df_out.select_dtypes(include=[np.number]).columns:
        median = df_out[col].median()
        df_out[col] = df_out[col].fillna(median)

    # 3. Normalizar strings: minúsculas y sin espacios extra
    for col in df_out.select_dtypes(include="object").columns:
        if col != "fecha":
            df_out[col] = df_out[col].str.strip().str.lower()

    # 4. Convertir fecha a datetime
    df_out["fecha"] = pd.to_datetime(df_out["fecha"])

    output_data = df_out

    print(f"Dataset generado: {len(df)} filas, {df.isnull().sum().sum()} nulos, {df.duplicated().sum()} duplicados")
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_limpiar_dataset_ventas()
    print("\nInput (head):")
    print(entrada["df"].head())
    print("\nOutput esperado (head):")
    print(salida.head())
    print(f"\nTipo fecha: {salida['fecha'].dtype}")
