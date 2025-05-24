#le code est valide en classe#
import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score
import pickle

class ImageSearchEngine:
    def __init__(self):
        self.index = {}
        self.methods = {
            'Histogramme Couleur': self.color_histogram,
            'Histogramme Niveaux de Gris': self.gray_histogram,
            'Corrélogramme': self.correlogram,
            'VGG16': self.vgg16_features
        }
        self.model_vgg = None

    def load_vgg16(self):
        """Charge le modèle VGG16 pré-entraîné"""
        base_model = VGG16(weights='imagenet')
        self.model_vgg = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def vgg16_features(self, image_path):
        """Extrait les features VGG16 d'une image"""
        if self.model_vgg is None:
            self.load_vgg16()
        img = Image.open(image_path).resize((224, 224))
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        img = preprocess_input(img[np.newaxis, ...])
        features = self.model_vgg.predict(img)
        return features.flatten()

    def color_histogram(self, image_path):
        image = cv2.imread(image_path)
        histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                                 [0, 256, 0, 256, 0, 256])
        return histogram.flatten()

    def gray_histogram(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        return histogram.flatten()

    def correlogram(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (256, 256))
        glcm = graycomatrix(img, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2],
                            levels=256, symmetric=True, normed=True)
        features = []
        for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
            features.extend(graycoprops(glcm, prop).flatten())
        return np.array(features)

    def index_images(self, directory, method_name):
        method = self.methods[method_name]
        self.index = {}
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    features = method(filepath)
                    self.index[filename] = features
                except Exception as e:
                    print(f"Erreur lors du traitement de {filename}: {e}")
        return self.index

    def search_similar_images(self, query_path, method_name, top_n=5):
        if method_name not in self.methods:
            raise ValueError("Méthode non reconnue")
        method = self.methods[method_name]
        query_features = method(query_path)
        distances = []
        for filename, features in self.index.items():
            if features is not None and query_features is not None:
                try:
                    distance = np.linalg.norm(query_features - features)
                    distances.append((filename, distance))
                except Exception as e:
                    print(f"Erreur comparaison avec {filename}: {e}")
        distances.sort(key=lambda x: x[1])
        return distances[:top_n]

    def save_index(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.index, f)

    def load_index(self, filename):
        with open(filename, 'rb') as f:
            self.index = pickle.load(f)


class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Moteur de Recherche d'Images")
        self.engine = ImageSearchEngine()
        self.method_var = tk.StringVar(value='Histogramme Couleur')
        self.db_path_var = tk.StringVar()
        self.query_path_var = tk.StringVar()
        self.results = []
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main_frame, text="Méthode d'indexation:").grid(row=0, column=0, sticky=tk.W)
        method_combo = ttk.Combobox(main_frame, textvariable=self.method_var,
                                    values=list(self.engine.methods.keys()))
        method_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        ttk.Label(main_frame, text="Base d'images:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.db_path_var, width=40).grid(row=1, column=1, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Parcourir", command=self.browse_db).grid(row=1, column=2, padx=5)

        ttk.Button(main_frame, text="Indexer les images", command=self.index_images).grid(row=2, column=1, pady=10)

        ttk.Label(main_frame, text="Image requête:").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.query_path_var, width=40).grid(row=3, column=1, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Parcourir", command=self.browse_query).grid(row=3, column=2, padx=5)

        ttk.Button(main_frame, text="Rechercher images similaires", command=self.search_images).grid(row=4, column=1, pady=10)

        self.results_frame = ttk.Frame(main_frame)
        self.results_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))

        main_frame.columnconfigure(1, weight=1)

    def browse_db(self):
        folder = filedialog.askdirectory(title="Sélectionner le répertoire de la base d'images")
        if folder:
            self.db_path_var.set(folder)

    def browse_query(self):
        file = filedialog.askopenfilename(title="Sélectionner l'image requête",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if file:
            self.query_path_var.set(file)

    def index_images(self):
        db_path = self.db_path_var.get()
        method = self.method_var.get()
        if not db_path:
            messagebox.showerror("Erreur", "Veuillez sélectionner un répertoire de base d'images")
            return
        try:
            self.engine.index_images(db_path, method)
            messagebox.showinfo("Succès", f"Indexation terminée avec la méthode {method}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'indexation: {str(e)}")

    def search_images(self):
        query_path = self.query_path_var.get()
        method = self.method_var.get()
        if not query_path:
            messagebox.showerror("Erreur", "Veuillez sélectionner une image requête")
            return
        if not self.engine.index:
            messagebox.showerror("Erreur", "Veuillez d'abord indexer la base d'images")
            return
        try:
            self.results = self.engine.search_similar_images(query_path, method)
            self.display_results()
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la recherche: {str(e)}")

    def display_results(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        if not self.results:
            ttk.Label(self.results_frame, text="Aucun résultat trouvé").pack()
            return

        ttk.Label(self.results_frame, text="Image requête:").pack(anchor=tk.W)
        self.show_image(self.query_path_var.get(), self.results_frame)

        ttk.Label(self.results_frame, text="Images similaires trouvées:").pack(anchor=tk.W)

        for i, (filename, distance) in enumerate(self.results):
            frame = ttk.Frame(self.results_frame)
            frame.pack(fill=tk.X, pady=5)
            ttk.Label(frame, text=f"{i+1}. {filename} (distance: {distance:.2f})").pack(side=tk.LEFT)
            db_path = self.db_path_var.get()
            img_path = os.path.join(db_path, filename)
            ttk.Button(frame, text="Afficher",
                       command=lambda path=img_path: self.show_large_image(path)).pack(side=tk.RIGHT)

    def show_image(self, image_path, parent):
        try:
            img = Image.open(image_path)
            img.thumbnail((100, 100))
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(parent, image=photo)
            label.image = photo
            label.pack()
        except Exception as e:
            print(f"Erreur affichage image: {e}")

    def show_large_image(self, image_path):
        try:
            img = Image.open(image_path)
            top = tk.Toplevel(self.root)
            top.title(os.path.basename(image_path))
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(top, image=photo)
            label.image = photo
            label.pack()
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher l'image: {str(e)}")


def main():
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
