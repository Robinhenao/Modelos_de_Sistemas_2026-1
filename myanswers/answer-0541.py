import pandas as pd
import numpy as np

def calcular_ratio_alto_valor(df):
    df = df.copy()
    df['valor_total'] = df['precio'] * df['cantidad']
    percentil_75 = df['valor_total'].quantile(0.75)
    alto_valor = (df['valor_total'] > percentil_75).sum()
    return alto_valor / len(df)