import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def entrenar_y_evaluar_knn(X, y, train_pct, n_neighbors):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_pct, random_state=42
    )

    modelo = KNeighborsClassifier(n_neighbors=n_neighbors)
    modelo.fit(X_train, y_train)

    preds = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    return {
        "Xts":      X_test,
        "yts":      y_test,
        "preds":    preds,
        "accuracy": accuracy
    }