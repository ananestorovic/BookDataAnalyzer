import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import joblib
from sklearn.metrics import mean_squared_error
from CustomLinearRegression import CustomLinearRegression
from sklearn.linear_model import LinearRegression


# Deli podatke na trening i test skupove, radi sa pandas DataFrame-ovima i Series.
def custom_train_test_split(X, y, test_size=0.2):

    if len(X) != len(y):
        raise ValueError("X i y moraju imati isti broj uzoraka")

    num_samples = len(X)
    num_test = int(num_samples * test_size)

    # Generisanje permutacije indeksa
    indices = np.random.permutation(num_samples)

    # Podela indeksa na trening i test skupove
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    # Kreiranje trening i test skupova koristeći .iloc za DataFrame i Series
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


# Ova klasa implementira prilagođeno min-max skaliranje podataka.
class CustomMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.array(X)
        self.min_ = X.min(axis=0)
        self.scale_ = X.max(axis=0) - self.min_
        # Izbegavamo deljenje sa nulom
        self.scale_[self.scale_ == 0] = 1
        return self

    def transform(self, X):
        X = np.array(X)
        X_scaled = (X - self.min_) / self.scale_
        X_scaled = (
            X_scaled * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )
        return X_scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.array(X)
        X_original = (X - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )
        X_original = X_original * self.scale_ + self.min_
        return X_original


def process_and_train_linear_regression_model():
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

    # One Hot Encoding za kolonu 'binding' koja samo sadrži vrednosti 'Tvrd' i 'Broš'
    X = pd.get_dummies(X, columns=["binding"], dtype=int)

    # Pretvaranje 'year' i 'pages' u integer vrednosti
    X["year"] = X["year"].astype(int)
    X["pages"] = X["pages"].astype(int)

    # Podela podataka na trening i test set
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2)
    X_train = X_train.to_numpy().reshape(-1, X_train.shape[1])
    X_test = X_test.to_numpy().reshape(-1, X_test.shape[1])
    y_train = y_train.to_numpy().reshape(-1, 1)

    # Konverzija y_train u float
    y_train = y_train.astype(float)

    # Skaliranje podataka kako bi se sve vrednosti u datasetu dovele u isti raspon
    scaling = CustomMinMaxScaler()
    X_train_scaled = scaling.fit_transform(X_train)
    X_test_scaled = scaling.transform(X_test)

    joblib.dump(scaling, "scaler.pkl")  # sacuvati skaler u fajl scaler.pkl

    # Treniranje modela pomocu custom linearne regresije
    custom_model = CustomLinearRegression()
    custom_model.fit(X_train_scaled, y_train)

    # Treniranje modela pomocu linearne regresije iz sklearn biblioteke
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train_scaled, y_train)

    # Izračunavanje srednje kvadratne greške za oba modela
    mse_custom_model = mean_squared_error(y_test, custom_model.predict(X_test_scaled))
    mse_linear_regression_model = mean_squared_error(
        y_test, linear_regression_model.predict(X_test_scaled)
    )

    print(f"MSE za custom linear regression model: {mse_custom_model}")
    print(f"MSE za linear regression model: {mse_linear_regression_model}")

    # Cuvanje modela
    # joblib.dump(custom_model, "custom_linear_regression_model.pkl")
    # joblib.dump(linear_regression_model, "linear_regression_model.pkl")

    print("Models have been stored successfully.")
