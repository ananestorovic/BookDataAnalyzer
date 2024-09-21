import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib


class BookPricePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BookPriceApp")


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

        # Dugme za predikciju cene
        self.predict_button = tk.Button(self.root, text="Predict Price", command=self.predict_price)
        self.predict_button.grid(row=7, column=1)

    def predict_price(self):
        try:
            # Prikupljanje podataka iz unosa
            year = int(self.year_entry.get())
            pages = int(self.pages_entry.get())
            binding = self.binding_entry.get().strip()
            # One-hot kodiranje za tip poveza
            binding_hard = 1 if binding == 'Tvrd' else 0
            binding_soft = 1 if binding == 'Broš' else 0

            # Kreiramo niz feature-a za predikciju
            input_features = np.array([[year, pages, binding_hard, binding_soft]])

            # Učitavanje treniranog modela
            model = joblib.load('trained_custom_linear_regression_model.pkl')

            # Učitavanje skalera za normalizaciju unosa
            scaler = joblib.load('scaler.pkl')
            input_scaled = scaler.transform(input_features)

            # Predikcija cene knjige
            predicted_price = model.predict(input_scaled)
            messagebox.showinfo("Predicted Price", f"The predicted price of the book is: {predicted_price[0][0]:.2f}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BookPricePredictionApp(root)
    root.mainloop()
