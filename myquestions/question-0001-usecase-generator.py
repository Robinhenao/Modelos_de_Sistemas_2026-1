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
    print(f"Dataset generado: {len(df)} filas, {df.isnull().sum().sum()} nulos, {df.duplicated().sum()} duplicados")
    return df


if __name__ == "__main__":
    df_input = generar_caso_de_uso_limpiar_dataset_ventas()
    print("\ndf_input:")
    print(df_input.head())
  