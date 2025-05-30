# Prediksi Harga Rumah di Tebet Menggunakan Regresi Linier

Proyek ini dibuat sebagai tugas dari mata kuliah **Sistem Cerdas** di Universitas Brawijaya. Tujuan utama dari proyek ini adalah memprediksi harga rumah di daerah Tebet menggunakan metode **Regresi Linier** berdasarkan berbagai fitur properti seperti luas bangunan, luas tanah, jumlah kamar tidur, kamar mandi, dan garasi.

## 🚀 Tujuan Proyek

Menyediakan model prediksi harga rumah yang dapat membantu:
- Pembeli dan penjual properti dalam membuat keputusan yang lebih baik.
- Developer dan investor untuk merencanakan pembangunan atau investasi secara strategis.
- Menentukan harga properti yang kompetitif dan akurat.

## 🧠 Metode

Proyek ini menggunakan metode **Regresi Linier**, yaitu pendekatan statistik yang mencari hubungan antara variabel target (harga rumah) dengan fitur-fitur input seperti:
- Luas Bangunan
- Luas Tanah
- Jumlah Kamar Tidur
- Jumlah Kamar Mandi
- Garasi

## 🛠️ Tools & Library

- Python
- Pandas
- Matplotlib
- Scikit-learn

## 🧹 Data Preparation

1. Menghapus data yang hilang (NaN).
2. Memisahkan fitur dan target.
3. Membagi data menjadi training (80%) dan testing (20%).

## 📊 Evaluasi Model

Model dievaluasi menggunakan dua metrik:
- **Mean Squared Error (MSE)**: untuk mengukur kesalahan prediksi.
- **R² Score**: untuk mengukur seberapa baik model menjelaskan data.

## 📈 Hasil

Model menunjukkan performa yang cukup baik dengan nilai MSE rendah dan R² mendekati 1, yang berarti hubungan antara fitur dan harga rumah cukup kuat.

Visualisasi hubungan nilai aktual dan prediksi juga ditampilkan dengan scatter plot.

## 💡 Fitur Aplikasi

- Melatih model regresi linier.
- Visualisasi hasil prediksi.
- Input manual untuk memprediksi harga rumah berdasarkan fitur-fitur tertentu.

## 🔧 Cara Menjalankan Program

1. Pastikan Python sudah terinstal.
2. Install dependensi:
    ```bash
    pip install pandas matplotlib scikit-learn openpyxl
    ```
3. Jalankan file Python:
    ```bash
    python prediksi_harga_rumah.py
    ```

## 👥 Kontributor

- Daffa Raihan Dwi Ari Putra (Pemrogram)
- Daffa Aprilian Herdikaputra (Pemrogram, Pengujian)
- Fahmi Robbani (Pencarian dataset, jurnal, slide)
- Miftahul Fikri Ramadhan (Penulisan laporan, PPT)
- Perlita Veda Fitrianingrum (Penulisan laporan, pencarian jurnal)

## 📚 Referensi

- [Kaggle - House Prediction in Tebet](https://www.kaggle.com/code/sateasinpedas/house-prediction-in-tebet-linear-regression)
---

