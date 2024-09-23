import pandas as pd
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import joblib
from collections import Counter
from imblearn.over_sampling import SMOTE

from books_analysis.CustomLogisticRegression import CustomLogisticRegression
from books_analysis.linear_regression import custom_train_test_split


# Funkcija koja kategorizuje cene knjiga u definisane opsege
def categorize_book_price(price):
    if price <= 500:
        return 0  # manje ili jednako od 500 dinara
    elif price <= 1500:
        return 1  # između 501 i 1500 dinara
    elif price <= 3000:
        return 2  # između 1501 i 3000 dinara
    elif price <= 5000:
        return 3  # između 3001 i 5000 dinara
    elif price <= 10000:
        return 4  # između 5001 i 10000 dinara
    else:
        return 5  # više od 10000 dinara


def train_and_evaluate_logistic_regression_models():
    # Kreiranje konekcije ka PostgreSQL bazi podataka
    db_url = "postgresql+psycopg2://postgres:postgres@localhost:5432/books_database"
    db_engine = create_engine(db_url)

    # Učitavanje podataka iz tabele
    books_df = pd.read_sql_table("preprocessed_books", con=db_engine)

    # Uklanjanje kolona koje nece biti koriscene prilikom treniranja
    X = books_df.drop(
        columns=[
            "code",
            "title",
            "author",
            "price",
            "format",
            "category",
            "publisher",
            "description",
        ]
    )
    y = books_df["price"]

    # Pretvaranje 'year' i 'pages' u integer vrednosti
    X["year"] = X["year"].astype(int)
    X["pages"] = X["pages"].astype(int)

    # One Hot Encoding za kolonu 'binding' koja samo sadrži vrednosti 'Tvrd' i 'Broš'
    X = pd.get_dummies(X, columns=["binding"], dtype=int)

    # Konverzija kolone 'price' u numerički tip (float)
    y = books_df["price"].astype(float)

    # Kategorizacija cene knjige u opsege na osnovu funkcije categorize_price
    y_categorical = y.apply(categorize_book_price)

    # Podela podataka na trening i test set
    X_train, X_test, y_train, y_test = custom_train_test_split(
        X, y_categorical, test_size=0.2
    )
    # Prikaz distribucije klasa pre balansiranja
    print("Distribucija klasa pre balansiranja:")
    print(Counter(y_train))

    # Primena SMOTE za balansiranje klasa
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = custom_balance_classes(X_train, y_train)
    # Prikaz distribucije klasa nakon balansiranja
    print("Distribucija klasa nakon balansiranja:")
    print(Counter(y_train_balanced))

    # One-vs-Rest Logistička regresija
    ovr_model = CustomLogisticRegression(
        max_iterations=10, eta0=0.05, alpha=0.005, multi_class="ovr"
    )
    print("Treniram ovr")
    ovr_model.fit(X_train, y_train)
    ovr_predictions = ovr_model.predict(X_test)

    # Multinomijalna Logistička regresija
    multi_model = CustomLogisticRegression(
        max_iterations=15, eta0=0.1, alpha=0.001, multi_class="multinomial"
    )
    print("Treniram multi")
    multi_model.fit(X_train, y_train)
    multi_predictions = multi_model.predict(X_test)

    # Evaluacija modela
    print("One-vs-Rest Logistic Regression:")
    print(f"Accuracy: {accuracy_score(y_test, ovr_predictions):.2f}")
    print(
        f"F1 Score (Weighted): {f1_score(y_test, ovr_predictions, average='weighted'):.2f}"
    )

    print("\nMultinomial Logistic Regression:")
    print(f"Accuracy: {accuracy_score(y_test, multi_predictions):.2f}")
    print(
        f"F1 Score (Weighted): {f1_score(y_test, multi_predictions, average='weighted'):.2f}"
    )

    # Čuvanje modela
    # joblib.dump(ovr_model, 'ovr_logistic_regression_model.pkl')
    # joblib.dump(multi_model, 'multinomial_logistic_regression_model.pkl')

    print("Models have been stored successfully.")

    # Prikaz konfuzione matrice za oba modela
    oVr_conf_matrix = ConfusionMatrixDisplay.from_predictions(y_test, ovr_predictions)
    oVr_conf_matrix.ax_.set_title(
        "Confusion Matrix for One-vs-Rest Logistic Regression"
    )

    multi_conf_matrix = ConfusionMatrixDisplay.from_predictions(
        y_test, multi_predictions
    )
    multi_conf_matrix.ax_.set_title(
        "Confusion Matrix for Multinomial Logistic Regression"
    )

    plt.show()


# Balansira klase koristeći tehniku oversampling-a, sa ograničenjem na ciljani broj uzoraka ili originalni broj
# uzoraka, koji god je manji.
def custom_balance_classes(X, y, target_samples=1800):

    # Spojimo X i y za lakšu manipulaciju
    df = pd.concat([X, y], axis=1)
    target_column = y.name

    # Inicijalizujmo liste za balansirane podatke
    balanced_dfs = []

    for class_label, count in Counter(y).items():
        # Izdvojimo sve uzorke ove klase
        class_df = df[df[target_column] == class_label]

        if count < target_samples:
            # Ako je broj uzoraka manji od ciljanog, zadržimo sve originalne uzorke
            balanced_dfs.append(class_df)
        else:
            # Ako je broj uzoraka veći od ciljanog, smanjimo na ciljani broj
            balanced_dfs.append(class_df.sample(n=target_samples, random_state=42))

    # Spojanje balansiranih klasa
    balanced_df = (
        pd.concat(balanced_dfs, axis=0)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Razdvajanje feature i target
    X_balanced = balanced_df.drop(columns=[target_column])
    y_balanced = balanced_df[target_column]

    return X_balanced, y_balanced
