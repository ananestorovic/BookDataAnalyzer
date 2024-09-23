import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import json
from collections import defaultdict
from random import uniform
from math import sqrt
import pandas as pd
import numpy as np


# Funkcija za izračunavanje proseka svih tačaka u jednoj dimenziji (centara klastera)
def point_avg(points):
    dimensions = len(points[0])
    new_center = []
    for dimension in range(dimensions):
        dim_sum = sum(p[dimension] for p in points)
        new_center.append(dim_sum / float(len(points)))
    return new_center


# Ažurira centre klastera na osnovu srednje vrednosti tačaka koje pripadaju svakom klasteru
def update_centers(data_set, assignments):
    new_means = defaultdict(list)
    centers = []
    # Grupisanje tačaka po klasterima
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
    # Računanje novih centara za svaki klaster
    for points in new_means.values():
        centers.append(point_avg(points))
    return centers


# Funkcija za dodeljivanje svake tačke najbližem centru klastera
def assign_points(data_points, centers):
    assignments = []
    for point in data_points:
        shortest = float("inf")
        shortest_index = 0
        for i, center in enumerate(centers):
            val = distance(point, center)
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


# Euklidovsko rastojanje između dve tačke
def distance(a, b):
    return sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


# Generiše k nasumičnih centara u opsegu podataka
def generate_k(data_set, k):
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = f"min_{i}"
            max_key = f"max_{i}"
            min_max[min_key] = min(min_max.get(min_key, val), val)
            min_max[max_key] = max(min_max.get(max_key, val), val)

    # Generiše nasumične centre unutar opsega minimuma i maksimuma
    for _ in range(k):
        rand_point = [
            uniform(min_max[f"min_{i}"], min_max[f"max_{i}"]) for i in range(dimensions)
        ]
        centers.append(rand_point)

    return centers


def k_means(dataset, k):
    # Generiše inicijalne centre
    k_points = generate_k(dataset, k)
    # Dodeljuje tačke inicijalnim klasterima
    assignments = assign_points(dataset, k_points)
    old_assignments = None

    # Iterativno ažuriranje centara dok dodeljivanja ne budu stabilna
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)

    # Računa inerciju (suma kvadrata rastojanja od centara)
    inertia = sum(
        distance(point, new_centers[assignment]) ** 2
        for point, assignment in zip(dataset, assignments)
    )

    return assignments, new_centers, inertia


# Čuva centre klastera u JSON fajlu
def save_model(centers, filename="kmeans_model.json"):
    with open(filename, "w") as f:
        json.dump(centers, f)
    print(f"Model saved to {filename}")


# Učitava centre klastera iz JSON fajla
def load_model(filename="kmeans_model.json"):
    with open(filename, "r") as f:
        centers = json.load(f)
    print(f"Model loaded from {filename}")
    return centers


# Predviđa kom klasteru pripada nova tačka
def predict(point, centers):
    assignments = assign_points([point], centers)
    return assignments[0]


# Kategorizuje cene knjiga u opsege
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


# Klasa za skaliranje podataka
class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=1)

        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.array(X)
        return X * self.scale_ + self.mean_


def categorize_book_price(price):
    if price < 1000:
        return 0  # Jeftino
    elif price < 2000:
        return 1  # Srednje
    else:
        return 2  # Skupo


# Priprema podataka za K-means klasterizaciju
def preprocess_data():
    # Kreiranje konekcije ka PostgreSQL bazi podataka
    db_url = "postgresql+psycopg2://postgres:postgres@localhost:5432/books_database"
    db_engine = create_engine(db_url)

    # Učitavanje podataka iz tabele
    books_df = pd.read_sql_table("preprocessed_books", con=db_engine)

    X = books_df[["year", "pages"]]

    # One Hot Encoding za kolonu 'binding'
    binding_encoded = pd.get_dummies(books_df["binding"], prefix="binding", dtype=int)
    X = pd.concat([X, binding_encoded], axis=1)

    # Pretvaranje 'year' i 'pages' u integer vrednosti
    X["year"] = X["year"].astype(int)
    X["pages"] = X["pages"].astype(int)

    # Standardizacija features-a
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)

    books_df["price"] = pd.to_numeric(books_df["price"])

    # Priprema ciljne varijable (kategorije cena)
    y = books_df["price"].apply(categorize_book_price)

    return X_scaled.tolist(), y.tolist(), scaler, books_df


def evaluate_clustering(assignments, true_labels):
    cluster_labels = defaultdict(list)
    for assignment, label in zip(assignments, true_labels):
        cluster_labels[assignment].append(label)

    purity = sum(
        max(labels.count(i) for i in set(labels)) for labels in cluster_labels.values()
    ) / len(assignments)
    return purity


def visualize_clusters(X, assignments, centers, scaler):
    X_inv = scaler.inverse_transform(X)
    df = pd.DataFrame(X_inv, columns=["year", "pages", "binding_Broš", "binding_Tvrd"])
    df["cluster"] = assignments

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        df["year"], df["pages"], df["binding_Tvrd"], c=df["cluster"], cmap="viridis"
    )

    centers_inv = scaler.inverse_transform(centers)
    ax.scatter(
        centers_inv[:, 0],
        centers_inv[:, 1],
        centers_inv[:, 2],
        c="red",
        s=200,
        alpha=0.8,
        marker="*",
    )

    ax.set_xlabel("Godina")
    ax.set_ylabel("Broj stranica")
    ax.set_zlabel("Povez (0=Broš, 1=Tvrd)")
    ax.set_title("3D vizualizacija klastera knjiga")

    plt.colorbar(scatter)
    #plt.savefig(fname="kmeans.png")
    plt.show()


def analyze_clusters(X, assignments, books_df):
    df = pd.DataFrame(X, columns=["year", "pages", "binding_Broš", "binding_Tvrd"])
    df["cluster"] = assignments
    df["price"] = books_df["price"]

    for i in range(max(assignments) + 1):
        cluster_data = df[df["cluster"] == i]
        print(f"\nKlaster {i}:")
        print(cluster_data.describe())
        print(f"\nProsečna cena: {cluster_data['price'].mean():.2f}")


def kmeans_run():
    X, y, scaler, books_df = preprocess_data()
    # Broj klastera
    k = 11

    # Trening modela
    assignments, centers, inertia = k_means(X, k)
    print("Trening završen")

    # Evaluacija klasterizacije
    purity = evaluate_clustering(assignments, y)
    print(f"Čistoća klasterizacije: {purity:.2f}")

    # Vizualizacija klastera
    visualize_clusters(X, assignments, centers, scaler)

    # Analiza klastera
    analyze_clusters(X, assignments, books_df)

    # Čuvanje modela (centri i scaler)
    model_data = {
        "centers": centers,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    #save_model(model_data)

    # Učitavanje modela
    loaded_model = load_model()
    loaded_centers = loaded_model["centers"]
    loaded_scaler = CustomStandardScaler()
    loaded_scaler.mean_ = np.array(loaded_model["scaler_mean"])
    loaded_scaler.scale_ = np.array(loaded_model["scaler_scale"])

    # Predikcija za novu knjigu
    new_book = [250, 1000, 0, 1]  # Primer: godina 2022, 300 stranica, tvrdi povez
    new_book_scaled = loaded_scaler.transform([new_book])[0].tolist()
    cluster = predict(new_book_scaled, loaded_centers)
    print(f"Nova knjiga pripada klasteru {cluster}")

    # Analiza inercije (Metod lakta)
    inertia_values = []

    # for k in range(1, 20):
    #     _, _, inertia = k_means(X, k)
    #     inertia_values.append(inertia)
    #     print(f"K = {k}, Inertia = {inertia}")
    #
    # # Vizualizacija inercije
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, 20), inertia_values, 'bo-')
    # plt.title("Metod lakta za određivanje optimalnog broja klastera")
    # plt.xlabel("Broj klastera (K)")
    # plt.ylabel("Inercija")
    # plt.show()
