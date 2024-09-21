import pandas as pd
from sqlalchemy import create_engine
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from CustomLinearRegression import CustomLinearRegression
from sklearn.linear_model import LinearRegression


def process_and_train_linear_regression_model():
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

    # Pretvaranje 'year' i 'pages' u integer vrednosti
    X['year'] = X['year'].astype(int)
    X['pages'] = X['pages'].astype(int)

    # Podela podataka na trening i test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train.to_numpy().reshape(-1, X_train.shape[1])
    X_test = X_test.to_numpy().reshape(-1, X_test.shape[1])
    y_train = y_train.to_numpy().reshape(-1, 1)

    # Konverzija y_train u float
    y_train = y_train.astype(float)

    # Skaliranje podataka kako bi se sve vrednosti u datasetu dovele u isti raspon
    scaling = MinMaxScaler()
    X_train_scaled = scaling.fit_transform(X_train)
    X_test_scaled = scaling.transform(X_test)

    joblib.dump(scaling, 'scaler.pkl') #sacuvati skaler u fajl scaler.pkl

    # Treniranje modela pomocu custom linearne regresije
    custom_model = CustomLinearRegression()
    custom_model.fit(X_train_scaled, y_train)

    # Treniranje modela pomocu linearne regresije iz sklearn biblioteke
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train_scaled, y_train)

    # Izračunavanje srednje kvadratne greške za oba modela
    mse_custom_model = mean_squared_error(y_test, custom_model.predict(X_test_scaled))
    mse_linear_regression_model = mean_squared_error(y_test, linear_regression_model.predict(X_test_scaled))

    print(f'MSE za custom linear regression model: {mse_custom_model}')
    print(f'MSE za linear regression model: {mse_linear_regression_model}')

    # Cuvanje modela
    joblib.dump(custom_model, 'trained_custom_linear_regression_model.pkl')
    joblib.dump(linear_regression_model, 'trained_linear_regression_model.pkl')

    print("Modeli su uspešno sačuvani.")
