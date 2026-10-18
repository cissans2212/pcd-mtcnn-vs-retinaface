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

### 4. Menjalankan Uji Eksperimen Baseline
```bash
python run_experiment.py
```

### 5. Menjalankan Eksperimen Preprocessing (Brightness, Contrast, CLAHE)
```bash
python run_preprocessing_experiment.py
```

---

## 🔬 Eksperimen Preprocessing Citra
Eksperimen ini mengevaluasi 4 skenario prapemrosesan pada 100 citra uji:
1. **Original**: Citra baseline tanpa transformasi.
2. **Brightness**: Peningkatan kecerahan linier ($\alpha = 1.0, \beta = 30$).
3. **Contrast**: Peningkatan kontras global ($\alpha = 1.3, \beta = 0$).
4. **CLAHE**: Peningkatan kontras adaptif lokal pada kanal Luminance (L) ruang warna CIELAB (`clipLimit = 2.0`, `tileGridSize = (8, 8)`), menjaga integritas warna kulit asli.

---

## 📁 Struktur Direktori
```text
├── app.py                                          # Aplikasi Web Streamlit (5 Modul Interaktif)
├── run_experiment.py                               # Script eksperimen batch MTCNN vs RetinaFace (Baseline)
├── run_preprocessing_experiment.py                 # Script eksperimen 4 skenario preprocessing
├── preprocessing_utils.py                          # Modul utilitas transformasi citra (Brightness, Contrast, CLAHE)
├── requirements.txt                                # Daftar dependensi Python
├── RINGKASAN_PROYEK_DAN_PROGRES.md                 # Ringkasan proyek, hasil, dan pencatat progres (AI handoff)
├── HASIL_EKSPERIMEN_DAN_LAPORAN_PROYEK.md          # Laporan komprehensif data eksperimen & template prompt jurnal
├── BAB_4_HASIL_DAN_PEMBAHASAN_LENGKAP.md           # Naskah lengkap Bab IV (Hasil & Pembahasan) skripsi/jurnal
├── Draf_Jurnal_MTCNN_RetinaFace_ITFORMAN_Kelompok1.docx # Naskah publikasi ilmiah (format ITFORMAN)
├── dataset_selected/                               # Sampel dataset terpilih (100 citra LFW)
├── foto_asli_100/                                  # Folder gambar asli pengujian
└── results/                                        # Output baseline dan folder preprocessing/
    ├── comparison.csv                              # Rekap hasil baseline
    ├── preprocessing_comparison.csv                # Rekap 400 pengujian preprocessing
    ├── preprocessing_scenario_summary.csv          # Rangkuman performa 4 skenario
    └── preprocessing/                              # Crops, deteksi, dan visualisasi skenario
```
