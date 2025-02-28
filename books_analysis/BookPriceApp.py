import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

from books_analysis.kmeans import CustomStandardScaler, load_model, predict


class BookPricePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BookPriceApp")

        # Nazivi za klase kojoj pripada predikcija cene knjige
        self.class_descriptions = {
            0: "Cena <= 500 dinara",
            1: "501 - 1500 dinara",
            2: "1501 - 3000 dinara",
            3: "3001 - 5000 dinara",
            4: "5001 - 10000 dinara",
            5: "Cena > 10000 dinara",
        }

        # Forma za unos podataka
        self.create_form()

    def create_form(self):
        # Labele za unos potrebnih polja
        tk.Label(self.root, text="Author").grid(row=0, column=0)
        self.author_entry = tk.Entry(self.root)
        self.author_entry.grid(row=0, column=1)

        tk.Label(self.root, text="Category").grid(row=1, column=0)
        self.category_entry = tk.Entry(self.root)
        self.category_entry.grid(row=1, column=1)

        tk.Label(self.root, text="Publisher").grid(row=2, column=0)
        self.publisher_entry = tk.Entry(self.root)
        self.publisher_entry.grid(row=2, column=1)

        tk.Label(self.root, text="Year").grid(row=3, column=0)
        self.year_entry = tk.Entry(self.root)
        self.year_entry.grid(row=3, column=1)

        tk.Label(self.root, text="Pages").grid(row=4, column=0)
        self.pages_entry = tk.Entry(self.root)
        self.pages_entry.grid(row=4, column=1)

        tk.Label(self.root, text="Binding (Tvrd/Broš)").grid(row=5, column=0)
        # Padajući meni za odabir tipa poveza
        self.binding_entry = ttk.Combobox(self.root, values=["Tvrd", "Broš"])
        self.binding_entry.grid(row=5, column=1)

        tk.Label(self.root, text="Format").grid(row=6, column=0)
        self.format_entry = tk.Entry(self.root)
        self.format_entry.grid(row=6, column=1)

        # Padajući meni za izbor modela
        tk.Label(self.root, text="Select Model").grid(row=7, column=0)
        self.model_selection = ttk.Combobox(
            self.root,
            values=[
                "Linear Regression",  # Linearna regresija
                "Logistic Regression (OvR)",  # Logistička regresija OvR (jedan naspram svih)
                "Logistic Regression (Multinomial)",  # Multinomialna logistička regresija
                "kmeans",  # K-means klasterizacija
            ],
        )
        self.model_selection.grid(row=7, column=1)

        # Dugme za predikciju cene
        self.predict_button = tk.Button(
            self.root, text="Predict Price", command=self.predict_price
        )
        self.predict_button.grid(row=8, column=1)

    def predict_price(self):
        try:
            # Prikupljanje podataka koje je uneo korisnik
            year = int(self.year_entry.get())
            pages = int(self.pages_entry.get())
            binding = self.binding_entry.get().strip()

            # One-hot kodiranje za tip poveza
            binding_hard = 1 if binding == "Tvrd" else 0
            binding_soft = 1 if binding == "Broš" else 0

            # Kreiramo niz feature-a za predikciju
            input_features = np.array([[year, pages, binding_soft, binding_hard]])

            # Učitavamo scaler model za normalizaciju input podataka
            scaler = joblib.load("scaler.pkl")
            input_scaled = scaler.transform(input_features)

            # Provera koji je model korisnik izabrao
            selected_model = self.model_selection.get()

            if selected_model == "Linear Regression":
                # Učitavanje modela linearne regresije i predikcija cene
                model = joblib.load("custom_linear_regression_model.pkl")
                predicted_price = model.predict(input_scaled)
                messagebox.showinfo(
                    "Predicted Price",
                    f"The predicted price of the book is: {predicted_price[0][0]:.2f}",
                )

            elif (
                # Učitavanje OvR logističkog regresionog modela
                selected_model == "Logistic Regression (OvR)"
                or selected_model == "Logistic Regression (Multinomial)"
            ):
                # Učitavanje multinomialnog logističkog regresionog modela
                if selected_model == "Logistic Regression (OvR)":
                    model = joblib.load("ovr_logistic_regression_model.pkl")
                else:
                    model = joblib.load("multinomial_logistic_regression_model.pkl")

                predicted_class = model.predict(input_scaled)[0]
                class_description = self.class_descriptions.get(
                    predicted_class, "Unknown class"
                )

                messagebox.showinfo(
                    "Predicted Class",
                    f"The predicted class of the book is: {predicted_class} ({class_description})",
                )
            elif selected_model == "kmeans":
                # Učitavanje modela K-means klasterizacije i skalera
                loaded_model = load_model()
                loaded_centers = loaded_model["centers"]
                loaded_scaler = CustomStandardScaler()
                loaded_scaler.mean_ = np.array(loaded_model["scaler_mean"])
                loaded_scaler.scale_ = np.array(loaded_model["scaler_scale"])
                # Skaliranje novih podataka
                new_book_scaled = loaded_scaler.transform(input_features.tolist())[
                    0
                ].tolist()
                # Predikcija klastera za novu knjigu
                cluster = predict(new_book_scaled, loaded_centers)
                messagebox.showinfo(
                    "Predicted Cluster",
                    f"The predicted cluster of the book is: {cluster} ",
                )

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BookPricePredictionApp(root)
    root.mainloop()
