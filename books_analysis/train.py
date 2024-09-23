from books_analysis.kmeans import kmeans_run
from books_analysis.linear_regression import process_and_train_linear_regression_model
from books_analysis.logistic_regression import (
    train_and_evaluate_logistic_regression_models,
)


def main():
    # Pozivanje funkcije za procesiranje i treniranje linearnog regresionog modela
    # process_and_train_linear_regression_model()

    # Pozivanje funkcije za treniranje i evaluaciju logistickog regresivnog modela
    #train_and_evaluate_logistic_regression_models()

    # Pozivanje funkcije za k-Means algoritam
     kmeans_run()


if __name__ == "__main__":
    main()
