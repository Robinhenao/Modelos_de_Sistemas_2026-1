import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


# =========================================================
# SOLUCIÓN
# =========================================================
def detectar_fraude(df):

    # ==========================================
    # Copia del DataFrame
    # ==========================================
    data = df.copy()

    # ==========================================
    # Codificación categórica
    # ==========================================
    le = LabelEncoder()

    data['pais'] = le.fit_transform(data['pais'])

    # ==========================================
    # Features y target
    # ==========================================
    features = [
        'monto',
        'hora',
        'frecuencia_transacciones',
        'pais'
    ]

    X = data[features].values

    y = data['fraude'].values

    # ==========================================
    # Desbalance
    # ==========================================
    n_legit = int((y == 0).sum())

    n_fraud = int((y == 1).sum())

    ratio = (
        round(n_fraud / n_legit, 4)
        if n_legit > 0 else float('inf')
    )

    # ==========================================
    # División train/test
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # ==========================================
    # Modelo
    # ==========================================
    modelo = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    # ==========================================
    # Predicciones
    # ==========================================
    preds = modelo.predict(X_test)

    # ==========================================
    # Métricas
    # ==========================================
    cm = confusion_matrix(y_test, preds)

    f1 = round(
        f1_score(
            y_test,
            preds,
            zero_division=0
        ),
        4
    )

    f1_w = round(
        f1_score(
            y_test,
            preds,
            average='weighted',
            zero_division=0
        ),
        4
    )

    # ==========================================
    # Output
    # ==========================================
    output = {

        'modelo': modelo,

        'confusion_matrix': cm,

        'f1': f1,

        'f1_weighted': f1_w,

        'desbalance': {

            'legítimas': n_legit,

            'fraudulentas': n_fraud,

            'ratio_fraude/legítima': ratio,

            'desbalanceado': ratio < 0.2
        },

        'predicciones': preds
    }

    return output


# =========================================================
# GENERADOR DE CASOS DE USO
# =========================================================
def generar_caso_de_uso_transacciones(
    n=1500,
    seed=42
):

    np.random.seed(seed)

    monto = np.random.exponential(200, n)

    hora = np.random.randint(0, 24, n)

    pais = np.random.choice(
        ['CO', 'US', 'MX', 'BR', 'AR'],
        n
    )

    frecuencia_transacciones = np.random.poisson(3, n)

    fraude = (
        (monto > 500) &
        (hora < 5)
    ).astype(int)

    df = pd.DataFrame({
        "monto": monto,
        "hora": hora,
        "pais": pais,
        "frecuencia_transacciones": frecuencia_transacciones,
        "fraude": fraude
    })

    # ==========================================
    # Inputs para la solución
    # ==========================================
    input_dict = {
        "df": df
    }

    # ==========================================
    # Resultado esperado
    # ==========================================
    expected_output = detectar_fraude(df)

    return input_dict, expected_output
