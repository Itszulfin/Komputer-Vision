import cv2
import pytesseract
import numpy as np
import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from PIL import Image, ImageTk
import joblib
import re
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate
from Klasifikasi import classify_text_with_ml, extract_title_text, save_classification_results

# Mengatur jalur Tesseract yang sudah terinstal
pytesseract.pytesseract.tesseract_cmd = r'D:\App\Tesseract\tesseract.exe'

# Fungsi untuk memuat data KBK dari file dataKBK.csv
def load_kbk_data(file_path):
    kbk_data = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                judul = row['judul'].strip()
                abstract = row['abstract'].strip()
                kbk = row['kbk'].strip()
                kbk_data.append({'judul': judul, 'abstract': abstract, 'kbk': kbk})
    except Exception as e:
        print(f"Gagal memuat data KBK: {e}")
    return kbk_data

# Fungsi untuk memuat data pelatihan dari file kbk_data.csv
def load_training_data(file_path):
    train_data = []
    train_labels = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['judul'].strip()
                label = row['kbk'].strip()
                train_data.append(text)
                train_labels.append(label)
    except Exception as e:
        print(f"Gagal memuat data pelatihan: {e}")
    return train_data, train_labels

# Memuat data KBK dari file dataKBK.csv
kbk_data = load_kbk_data("datakbk.csv")

# Memuat data pelatihan dari file kbk_data.csv
train_data, train_labels = load_training_data("datakbk.csv")

# Model pipeline menggunakan TfidfVectorizer dan MultinomialNB
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Latih model
model.fit(train_data, train_labels)

# Fungsi untuk memuat model dan memperbarui data pelatihan
def load_model():
    global model, train_data, train_labels
    try:
        train_data, train_labels = load_training_data("kbk_data.csv")
        model.fit(train_data, train_labels)
        messagebox.showinfo("Info", "Model berhasil dimuat dan diperbarui dengan data terbaru.")
    except Exception as e:
        messagebox.showerror("Error", f"Gagal memuat model: {e}")

# Fungsi untuk menyimpan model dan hasil klasifikasi ke file CSV
def save_model():
    try:
        joblib.dump(model, "model_ml.pkl")
        with open("kbk_data.csv", mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["judul", "abstract", "kbk"])
            writer.writeheader()
            writer.writerows(kbk_data)
        messagebox.showinfo("Info", "Model dan data hasil klasifikasi berhasil disimpan ke file CSV.")
    except Exception as e:
        messagebox.showerror("Error", f"Gagal menyimpan model dan data: {e}")

# Fungsi untuk memusatkan GUI di tengah layar
def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')

# Fungsi untuk menampilkan GUI dengan scrollable window dan center window
def create_gui():
    global eval_text

    # Inisialisasi GUI
    root = tk.Tk()
    root.title("OCR dan Klasifikasi Teks")
    root.geometry("800x600")

    # Memusatkan jendela GUI
    center_window(root)

    # Membuat canvas dan frame dengan scrollbar
    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="n", width=780)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Fungsi untuk memuat gambar
    def upload_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;.jpeg")])
        if file_path:
            img = Image.open(file_path)
            img.thumbnail((400, 400))
            img = ImageTk.PhotoImage(img)
            panel.configure(image=img)
            panel.image = img
            result, processed_image = extract_title_text(file_path)
            classification, description = classify_text_with_ml(result)
            result_text.set(f"Teks Terdeteksi: {result}\nKlasifikasi: {classification}\nDeskripsi: {description}")
            kbk_data.append({"judul": result, "abstract": description, "kbk": classification})
            save_classification_results([result, classification, description])

    # Fungsi untuk mengunggah banyak gambar dan menghitung evaluasi model
    def upload_multiple_images():
        folder_path = filedialog.askdirectory()  # Pilih folder yang berisi gambar
        if folder_path:
            true_labels = []
            predicted_labels = []
            predictions_details = []  # Menyimpan detail prediksi setiap gambar
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Cek apakah itu gambar
                    try:
                        # Mengambil nama folder untuk true label
                        folder_name = os.path.basename(folder_path)  # Nama folder menjadi true label
                        true_labels.append(folder_name)

                        # Proses OCR dan klasifikasi
                        result, processed_image = extract_title_text(file_path)
                        classification, description = classify_text_with_ml(result)
                        predicted_labels.append(classification)

                        # Simpan detail prediksi
                        predictions_details.append( 
                            f"File: {file_name} | True Label: {folder_name} | Prediksi: {classification} | Teks: {result}"
                        )

                        # Tambahkan data ke KBK
                        kbk_data.append({"judul": result, "abstract": description, "kbk": classification})
                        save_classification_results([result, classification, description])
                    except Exception as e:
                        predictions_details.append(f"File: {file_name} | Error: {e}")

            # Hitung metrik evaluasi
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
            recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

            # Tampilkan hasil evaluasi
            eval_text.delete(1.0, tk.END)
            eval_text.insert(tk.END, f"Jumlah Foto yang Diunggah: {len(true_labels)}\n\n")
            eval_text.insert(tk.END, f"Prediksi AI untuk setiap gambar:\n")
            for detail in predictions_details:
                eval_text.insert(tk.END, detail + "\n")

            eval_text.insert(tk.END, f"\nAkurasi: {accuracy:.2f}\nPresisi: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}\n")

    # Membuat widget GUI di dalam frame yang dapat di-scroll
    panel = tk.Label(scrollable_frame)
    panel.pack(padx=10, pady=10, anchor="center")

    result_text = tk.StringVar()
    result_label = tk.Label(scrollable_frame, textvariable=result_text, wraplength=700)
    result_label.pack(padx=10, pady=10, anchor="center")

    # Menggunakan widget Text untuk menampilkan hasil evaluasi
    eval_text = tk.Text(scrollable_frame, wrap="word", height=15, width=90)
    eval_text.pack(padx=10, pady=10, anchor="center")
    eval_text.insert(tk.END, "Hasil evaluasi model akan tampil di sini.\n")

    upload_btn = tk.Button(scrollable_frame, text="Unggah Gambar", command=upload_image)
    upload_btn.pack(pady=5, anchor="center")

    upload_multiple_btn = tk.Button(scrollable_frame, text="Tes Presisi", command=upload_multiple_images)
    upload_multiple_btn.pack(pady=5, anchor="center")

    load_btn = tk.Button(scrollable_frame, text="Muat Model", command=load_model)
    load_btn.pack(pady=5, anchor="center")

    save_btn = tk.Button(scrollable_frame, text="Simpan Model", command=save_model)
    save_btn.pack(pady=5, anchor="center")

    root.mainloop()

# Menjalankan fungsi utama
create_gui()
