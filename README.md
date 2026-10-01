# PCD: Perbandingan Deteksi Wajah MTCNN vs RetinaFace

Aplikasi berbasis Streamlit dan skrip eksperimen untuk membandingkan performa model **MTCNN** (Multi-task Cascaded Convolutional Networks) dan **RetinaFace** dalam deteksi serta cropping wajah pada dataset citra.

---

## 📌 Fitur Utama
1. **Interactive Dashboard (Streamlit)**:
   - Visualisasi perbandingan metrik performa (Inference Time, Deteksi, Confidence Score).
   - Mode interaktif pengujian single image (Upload foto custom atau pilih dari dataset).
   - Galeri perbandingan bounding box dan hasil crop wajah (160x160 px).
2. **Experiment Script (`run_experiment.py`)**:
   - Menjalankan pipeline evaluasi otomatis pada dataset.
   - Menyimpan hasil visualisasi dan rekapan perbandingan ke format CSV.

---

## 🚀 Cara Menjalankan

### 1. Buat & Aktifkan Virtual Environment (Opsional tapi Direkomendasikan)
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Dashboard Streamlit
```bash
streamlit run app.py
```

### 4. Menjalankan Uji Eksperimen
```bash
python run_experiment.py
```

---

## 📁 Struktur Direktori
```text
├── app.py                 # Aplikasi Web Streamlit
├── run_experiment.py      # Script eksperimen batch MTCNN vs RetinaFace
├── requirements.txt       # Daftar dependensi Python
├── .gitignore             # File abaian Git (ZIP/Cache/Virtualenv)
├── dataset_selected/      # Sampel dataset terpilih
├── foto_asli_100/         # Folder gambar asli pengujian
└── results/               # Output deteksi, crop, dan CSV komparasi
```
