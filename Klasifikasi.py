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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report
from tabulate import tabulate

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

    load_model_btn = tk.Button(scrollable_frame, text="Muat Model", command=load_model)
    load_model_btn.pack(pady=5, anchor="center")

    save_model_btn = tk.Button(scrollable_frame, text="Simpan Model dan Data", command=save_model)
    save_model_btn.pack(pady=5, anchor="center")

    eval_model_btn = tk.Button(scrollable_frame, text="Evaluasi Model", command=evaluate_model)
    eval_model_btn.pack(pady=5, anchor="center")

    root.mainloop()

# Memuat data dari CSV
def load_data_from_csv(datakbk_csv):
    train_data = ['judul']
    train_labels = ['kbk']
    
    with open(datakbk_csv, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            train_data.append(row['judul'])
            train_labels.append(row['kbk']) 
    
    return train_data, train_labels

# Fungsi untuk praproses teks yang terdeteksi dari OCR agar lebih bersih dan konsisten
def preprocess_text(text):
    text = text.lower()  # Ubah menjadi huruf kecil
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Hanya pertahankan alfanumerik dan spasi
    text = text.strip()  # Hilangkan spasi berlebih di awal/akhir
    return text

# Fungsi untuk mengukur akurasi dan presisi model pengklasifikasian teks
def evaluate_model():
    global eval_text
    
    # Memuat data dari file CSV
    train_data, train_labels = load_data_from_csv('datakbk.csv')
    
    # Split data training dan testing (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
    
    # Melatih model dengan data train
    model.fit(X_train, y_train)
    
    # Prediksi pada data test
    y_pred_raw = model.predict(X_test)
    
    # Praproses prediksi dan label untuk konsistensi
    y_pred = [preprocess_text(text) for text in y_pred_raw]  # Prediksi diproses agar seragam
    y_test_clean = [preprocess_text(text) for text in y_test]  # Label tes juga diproses agar konsisten
    
    # Mengambil metrik evaluasi
    accuracy = accuracy_score(y_test_clean, y_pred)
    precision = precision_score(y_test_clean, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test_clean, y_pred, output_dict=True, zero_division=0)
    
    # Memformat hasil klasifikasi menggunakan tabulate
    headers = ["", "precision", "recall", "f1-score", "support"]
    rows = [[label] + list(metrics.values()) for label, metrics in report.items() if label != 'accuracy']
    rows.append(["accuracy", "", "", "", report["accuracy"]])
    table = tabulate(rows, headers=headers, tablefmt="grid")
    
    # Kosongkan Text widget sebelum menambahkan hasil baru
    eval_text.delete(1.0, tk.END)
    eval_text.insert(tk.END, f"Akurasi: {accuracy:.2f}\nPresisi: {precision:.2f}\n\nLaporan Klasifikasi:\n{table}")
# Menyimpan hasil klasifikasi ke dalam file kbk_data.csv
def save_classification_results(result, filename="kbk_data.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Extracted Text", "Classification", "Description"])
        writer.writerow(result)
    print(f"Hasil klasifikasi disimpan di: {filename}")

# Melakukan praproses pada gambar dengan mengubahnya menjadi grayscale dan menerapkan threshold binarization.
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Menemukan area teks dengan menggunakan kontur
def detect_text_area(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Membuat bounding box dari kontur yang paling besar (diasumsikan sebagai area teks terbesar)
    if text_contours:
        x, y, w, h = cv2.boundingRect(text_contours[0])
        return x, y, w, h
    return None

# Mengekstrak teks judul dari gambar menggunakan OCR Tesseract dengan cropping ROI manual
def extract_title_text(image_path):
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)
    
    # Manual cropping of the ROI: define the area to scan (adjust these values as needed)
    # These values should focus on the title area, excluding logos or covers
    crop_x, crop_y, crop_w, crop_h = 180, 27, 1000, 200  # Example of a cropped area
    roi = processed_image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    # Apply OCR to the cropped ROI
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(roi, config=custom_config)
    
    # Display the cropped ROI for verification
    cv2.imshow("Cropped ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Extract text until the first slash if any
    text_until_slash = extracted_text.split('/')[0] 
    return text_until_slash.strip(), processed_image

# Klasifikasi teks menggunakan model machine learning
def classify_text_with_ml(text):
    classification = model.predict([text])[0]
    return classification, f"Klasifikasi sebagai {classification}"

if __name__ == "__main__":
    create_gui()
