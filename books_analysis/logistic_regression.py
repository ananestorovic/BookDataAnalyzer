import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib


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
    db_url = 'postgresql+psycopg2://postgres:postgres@localhost:5432/books_database'
    db_engine = create_engine(db_url)

    # Učitavanje podataka iz tabele
    books_df = pd.read_sql_table('preprocessed_books', con=db_engine)

    # Uklanjanje kolona koje nece biti koriscene prilikom treniranja
    X = books_df.drop(columns=['code', 'title', 'author', 'price', 'format', 'category', 'publisher', 'description'])
    y = books_df['price']

    # One Hot Encoding za kolonu 'binding' koja samo sadrži vrednosti 'Tvrd' i 'Broš'
    X = pd.get_dummies(X, columns=['binding'], dtype=int)

    # Konverzija kolone 'price' u numerički tip (float)
    y = books_df['price'].astype(float)

    # Kategorizacija cene knjige u opsege na osnovu funkcije categorize_price
    y_categorical = y.apply(categorize_book_price)

    # Podela podataka na trening i test set
    X_train, X_test, y_train_categorical, y_test = train_test_split(X, y_categorical, test_size=0.2)
    X_train = X_train.to_numpy().reshape(-1, X_train.shape[1])
    X_test = X_test.to_numpy().reshape(-1, X_test.shape[1])
    y_train_categorical = y_train_categorical.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()

    # Skaliranje podataka kako bi se svi podaci doveli u isti raspon vrednosti
    scaling = MinMaxScaler()
    X_train_scaled = scaling.fit_transform(X_train)
    X_test_scaled = scaling.transform(X_test)

    # One-vs-Rest Logisticka regresija
    oVr_logistic_model = OneVsRestClassifier(LogisticRegression())
    oVr_logistic_model.fit(X_train_scaled, y_train_categorical)

    # Predikcija
    oVr_predicted_labels = oVr_logistic_model.predict(X_test_scaled)

    print("One-vs-Rest (OvR) Logistic Regression:")
    print(oVr_predicted_labels)

    # Multinomijalna Logisticka  regresija
    multi_logistic_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    multi_logistic_model.fit(X_train_scaled, y_train_categorical)
    # Predikcija
    multi_predicted_labels = multi_logistic_model.predict(X_test_scaled)

    print("Multinomial Logistic Regression:")
    print(multi_predicted_labels)

    # Izračunavanje preciznosti za oba modela
    oVr_accuracy_result = accuracy_score(y_test, oVr_predicted_labels)
    multi_accuracy_result = accuracy_score(y_test, multi_predicted_labels)

    # Izračunavanje F1 skora za oba modela
    oVr_f1_result = f1_score(y_test, oVr_predicted_labels, average='weighted')
    multi_f1_result = f1_score(y_test, multi_predicted_labels, average='weighted')

    # Prikaz rezultata za One-vs-Rest logističku regresiju
    print("Results for One-vs-Rest Logistic Regression:")
    print(f"Accuracy: {oVr_accuracy_result:.2f}")
    print(f"F1 Score (Weighted): {oVr_f1_result:.2f}")

    # Prikaz rezultata za Multinomial logističku regresiju
    print("\nResults for Multinomial Logistic Regression:")
    print(f"Accuracy: {multi_accuracy_result:.2f}")
    print(f"F1 Score (Weighted): {multi_f1_result:.2f}")

    # Čuvanje modela u fajlove za kasniju upotrebu
    joblib.dump(oVr_logistic_model, 'ovr_logistic_regression_model.pkl')
    joblib.dump(multi_logistic_model, 'multinomial_logistic_regression_model.pkl')

    print("Models have been stored successfully.")


