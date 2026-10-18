# BAB IV: HASIL DAN PEMBAHASAN
## DOKUMEN REKAPITULASI DATA EMPIRIS LENGKAP & ANALISIS AKADEMIS PENYUSUNAN LAPORAN SKRIPSI / JURNAL

> **Panduan Penggunaan Dokumen Ini:**  
> Dokumen ini disusun secara sistematis sesuai dengan format baku **BAB IV (Hasil dan Pembahasan)** untuk laporan Tugas Akhir / Skripsi / Naskah Jurnal Ilmiah (Template ITFORMAN/JOINS). Dokumen ini memuat seluruh data empiris aktual, tabel statistik, formulasi matematis, analisis komparatif kuantitatif dan kualitatif dari:
> 1. **Eksperimen Baseline**: Evaluasi kinerja deteksi dan face cropping model MTCNN vs. RetinaFace pada 100 citra LFW.
> 2. **Eksperimen Preprocessing Citra**: Analisis pengaruh manipulasi kecerahan (*Brightness*), peregangan kontras (*Contrast*), dan perataan histogram adaptif lokal (*CLAHE* pada ruang warna CIELAB) terhadap performa detektor.

---

## 4.1 LINGKUNGAN PENGUJIAN DAN KONFIGURASI SISTEM

Pengujian dilakukan pada lingkungan perangkat keras dan perangkat lunak yang terkontrol guna memastikan validitas komparasi (*controlled benchmark environment*):

### 4.1.1 Spesifikasi Perangkat Keras dan Perangkat Lunak
- **Processor (CPU)**: Central Processing Unit x86_64 dengan set instruksi AVX2/FMA aktif.
- **Sistem Operasi**: Microsoft Windows 11 64-bit.
- **Bahasa Pemrograman**: Python 3.10.
- **Pustaka Utama**:
  - `facenet-pytorch` (Implementasi PyTorch MTCNN)
  - `retina-face` (Implementasi RetinaFace berbasis TensorFlow/Keras)
  - `opencv-python` v4.x (Operasi prapemrosesan citra, CLAHE, manipulasi matriks piksel, dan I/O)
  - `numpy` & `pandas` (Analisis statistik dan agregasi data tabular)
  - `matplotlib` (Pembuatan visualisasi grid dan grafik perbandingan)
  - `streamlit` (Platform dashboard interaktif pengujian real-time)

### 4.1.2 Prosedur Benchmark & Model Warm-Up
Untuk mengeliminasi bias latensi akibat *cold-start* (alokasi memori bobot jaringan saat pemanggilan pertama), sistem menjalankan fase pemanasan (*warm-up phase*) menggunakan matriks citra tiruan berdimensi $250 \times 250 \times 3$ piksel sebelum pencatatan waktu dimulai. Seluruh durasi waktu diukur menggunakan *high-resolution performance counter* (`time.perf_counter()`).

### 4.1.3 Karakteristik Dataset Uji
- **Dataset**: *Labeled Faces in the Wild* (LFW Funneled Dataset).
- **Jumlah Sampel**: 100 citra digital format RGB, resolusi asli $250 \times 250$ piksel.
- **Stratifikasi**: 20 figur publik internasional representatif, masing-masing 5 citra terurut:
  *Andre Agassi, Ariel Sharon, Arnold Schwarzenegger, Colin Powell, Donald Rumsfeld, George W. Bush, Gerhard Schroeder, Gloria Macapagal Arroyo, Hans Blix, Hugo Chavez, Jacques Chirac, Jennifer Capriati, Junichiro Koizumi, Kofi Annan, Laura Bush, Luiz Inacio Lula da Silva, Megawati Sukarnoputri, Serena Williams, Tony Blair, Vladimir Putin.*
- **Variasi Kondisi Citra**: Orientasi wajah frontal vs miring (*yaw/pitch/roll*), aksesoris optik (kacamata bingkai tebal/refleksi lensa), variasi ekspresi emosi (senyum lebar, mulut terbuka), oklusi parsial rambut, serta pencahayaan *indoor* dan *outdoor*.

---

## 4.2 PARAMETER DAN FORMULASI METODE PREPROCESSING

Empat skenario prapemrosesan citra diuji secara independen pada 100 citra yang sama (total 400 evaluasi):

```mermaid
flowchart LR
    In["Citra Asli (LFW 250x250)"] --> S1["1. Original (Baseline)"]
    In --> S2["2. Brightness (beta = +30)"]
    In --> S3["3. Contrast (alpha = 1.3)"]
    In --> S4["4. CLAHE (Kanal L LAB)"]
    
    S1 & S2 & S3 & S4 --> Det["Deteksi Wajah: MTCNN vs RetinaFace"]
    Det --> Eval["Metrik: Detection Rate, Crop Rate, Latensi, Inter-Model IoU"]
```

### 1. Skenario 1: Original (Baseline)
Citra input diproses langsung tanpa modifikasi intensitas piksel ($I_{\text{out}} = I_{\text{in}}$). Digunakan sebagai tolok ukur acuan (*ground benchmark*).

### 2. Skenario 2: Penyesuaian Kecerahan (Brightness Adjustment)
Meningkatkan intensitas pencahayaan secara linier untuk mensimulasikan kondisi pencahayaan berlebih (*over-exposure*):
$$I_{\text{bright}}(x, y) = \text{clip}\left(1.0 \cdot I(x, y) + \beta, \, 0, \, 255\right)$$
dengan parameter terdokumentasi $\alpha = 1.0$ dan $\beta = +30$. Pemotongan saturasi (*clipping*) dilakukan menggunakan fungsi saturasi `cv2.convertScaleAbs` untuk menjamin tidak terjadi *overflow* modular `uint8`.

### 3. Skenario 3: Peningkatan Kontras Global (Contrast Adjustment)
Meregangkan rentang dinamis intensitas piksel (*dynamic range stretching*) untuk mempertegas batas tepi (*edge gradient*):
$$I_{\text{contrast}}(x, y) = \text{clip}\left(\alpha \cdot I(x, y) + \beta, \, 0, \, 255\right)$$
dengan parameter terdokumentasi $\alpha = 1.3$ dan $\beta = 0$.

### 4. Skenario 4: Contrast Limited Adaptive Histogram Equalization (CLAHE) pada Ruang Warna CIELAB
Pemerataan histogram adaptif lokal untuk meningkatkan kontras detail bayangan tanpa memperkuat derau latar belakang (*noise amplification*). Agar warna kulit alami subjek tidak mengalami distorsi kroma (*color fading* / *hue shift*), transformasi diterapkan khusus pada kanal luminansi ($L$) dalam ruang warna CIELAB ($L^*a^*b^*$):
1. Konversi ruang warna: $\text{RGB} \xrightarrow{\mathcal{T}} \text{CIELAB}$.
2. Pemisahan kanal menjadi komponen $L$, $a$, dan $b$.
3. Penerapan CLAHE pada kanal $L$:
   $$L_{\text{enhanced}} = \text{CLAHE}(L, \, \text{clipLimit} = 2.0, \, \text{tileGridSize} = (8, 8))$$
4. Penggabungan kembali kanal: $(L_{\text{enhanced}}, a, b)$.
5. Konversi balik ke ruang warna tampilan: $\text{CIELAB} \xrightarrow{\mathcal{T}^{-1}} \text{RGB}$.

---

## 4.3 METRIK EVALUASI KINERJA

Evaluasi dilakukan menggunakan parameter kuantitatif terstandarisasi:

1. **Tingkat Keberhasilan Deteksi (Detection Success Rate)**:
   $$\text{Detection Rate} = \left(\frac{N_{\text{detected}}}{N_{\text{total}}}\right) \times 100\%$$
2. **Tingkat Keberhasilan Cropping (Face Cropping Rate)**:
   Persentase citra yang berhasil dipotong sesuai koordinat bounding box dan diubah ukurannya secara seragam ke dimensi target $160 \times 160$ piksel menggunakan interpolasi bilinear (`cv2.INTER_LINEAR`).
3. **Waktu Komputasi (Latency Decomposition)**:
   Diukur secara terpisah untuk setiap sampel:
   - **Waktu Preprocessing ($t_{\text{prep}}$)**: Durasi komputasi manipulasi citra digital.
   - **Waktu Inferensi ($t_{\text{infer}}$)**: Durasi ekstraksi fitur dan lokalisasi wajah oleh jaringan neural.
   - **Total Waktu Pemrosesan ($t_{\text{total}}$)**: $t_{\text{total}} = t_{\text{prep}} + t_{\text{infer}}$.
4. **Derajat Kesepakatan Spasial Antar-Model (Inter-Model Intersection over Union / IoU)**:
   Mengukur derajat tumpang tindih spasial antara bounding box yang dihasilkan oleh RetinaFace ($B_{\text{RF}}$) dan MTCNN ($B_{\text{MT}}$):
   $$\text{IoU}(B_{\text{RF}}, B_{\text{MT}}) = \frac{\text{Area}(B_{\text{RF}} \cap B_{\text{MT}})}{\text{Area}(B_{\text{RF}} \cup B_{\text{MT}})} = \frac{\text{Area}(B_{\text{RF}} \cap B_{\text{MT}})}{\text{Area}(B_{\text{RF}}) + \text{Area}(B_{\text{MT}}) - \text{Area}(B_{\text{RF}} \cap B_{\text{MT}})}$$
   > **Catatan Metodologis:**  
   > Dataset LFW tidak menyediakan anotasi bounding box manual ground truth. Oleh karena itu, metrik IoU pada penelitian ini merepresentasikan **Inter-Model Spatial Agreement** (derajat konsistensi spasial antara dua arsitektur detektor state-of-the-art), bukan akurasi mutlak terhadap ground truth manual.

---

## 4.4 HASIL EKSPERIMEN KUANTITATIF

### 4.4.1 Tabel Komparasi Kinerja Utama Antar Skenario (400 Uji Coba)

Tabel berikut menyajikan rekapitulasi performa lengkap pada 100 citra untuk keempat skenario pengujian:

| Skenario Pengujian | Jumlah Citra | Waktu Preproc Rata-rata | MTCNN Deteksi Sukses (%) | MTCNN Crop Sukses (%) | MTCNN Rata-rata Inferensi | MTCNN Rata-rata Total Waktu | RetinaFace Deteksi Sukses (%) | RetinaFace Crop Sukses (%) | RetinaFace Rata-rata Inferensi | RetinaFace Rata-rata Total Waktu | Rata-rata Inter-Model IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Original** | 100 | **0.03 ms** | **100/100 (100.0%)** | **100/100 (100.0%)** | **0.0626 s** (62.6 ms) | 0.0626 s | **100/100 (100.0%)** | **100/100 (100.0%)** | **1.2872 s** (1287.2 ms) | 1.2872 s | **0.9083** (90.83%) |
| **2. Brightness** (+30) | 100 | **0.09 ms** | **100/100 (100.0%)** | **100/100 (100.0%)** | **0.0553 s** (55.3 ms) | 0.0554 s | **100/100 (100.0%)** | **100/100 (100.0%)** | **1.3168 s** (1316.8 ms) | 1.3169 s | **0.9033** (90.33%) |
| **3. Contrast** ($\times 1.3$) | 100 | **0.09 ms** | **100/100 (100.0%)** | **100/100 (100.0%)** | **0.0609 s** (60.9 ms) | 0.0610 s | **100/100 (100.0%)** | **100/100 (100.0%)** | **1.3240 s** (1324.0 ms) | 1.3241 s | **0.8985** (89.85%) |
| **4. CLAHE** (LAB) | 100 | **4.93 ms** | **100/100 (100.0%)** | **100/100 (100.0%)** | **0.0752 s** (75.2 ms) | 0.0802 s | **100/100 (100.0%)** | **100/100 (100.0%)** | **1.3237 s** (1323.7 ms) | 1.3286 s | **0.9040** (90.40%) |

---

### 4.4.2 Analisis Statistik Distribusi Waktu Inferensi

| Skenario | Model | Rata-rata (Mean) | Standar Deviasi (Std) | Waktu Tercepat (Min) | Waktu Terlama (Max) | Rasio Kecepatan (Speedup) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Original** | MTCNN | 0.0626 s | ± 0.0285 s | 0.0371 s | 0.1963 s | **MTCNN 20.56× lebih cepat** |
| | RetinaFace | 1.2872 s | ± 0.0968 s | 1.1890 s | 1.7001 s | Acuan Dasar |
| **Brightness** | MTCNN | 0.0553 s | ± 0.0160 s | 0.0365 s | 0.1240 s | **MTCNN 23.81× lebih cepat** |
| | RetinaFace | 1.3168 s | ± 0.1019 s | 1.1920 s | 1.8230 s | Acuan Dasar |
| **Contrast** | MTCNN | 0.0609 s | ± 0.0245 s | 0.0379 s | 0.1650 s | **MTCNN 21.74× lebih cepat** |
| | RetinaFace | 1.3240 s | ± 0.1707 s | 1.1850 s | 1.9540 s | Acuan Dasar |
| **CLAHE** | MTCNN | 0.0752 s | ± 0.0532 s | 0.0450 s | 0.2210 s | **MTCNN 17.60× lebih cepat** |
| | RetinaFace | 1.3237 s | ± 0.1245 s | 1.1820 s | 1.7650 s | Acuan Dasar |

---

### 4.4.3 Analisis Statistik Inter-Model IoU per Skenario

| Skenario | Mean IoU | Median IoU | Std Deviasi | Min IoU | Max IoU | % Sampel IoU $\ge 0.90$ | % Sampel $0.80 \le \text{IoU} < 0.90$ | % Sampel $\text{IoU} < 0.80$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | **0.9083** | 0.9101 | 0.0350 | 0.8055 | 0.9778 | **68.0%** | **32.0%** | **0.0%** |
| **Brightness** | **0.9033** | 0.9065 | 0.0392 | 0.7506 | 0.9745 | **64.0%** | **34.0%** | **2.0%** |
| **Contrast** | **0.8985** | 0.9030 | 0.0401 | 0.7868 | 0.9643 | **59.0%** | **38.0%** | **3.0%** |
| **CLAHE** | **0.9040** | 0.9070 | 0.0383 | 0.7872 | 0.9649 | **65.0%** | **33.0%** | **2.0%** |

---

### 4.4.4 Rekapitulasi Rincian Kinerja per 20 Identitas Tokoh Publik (Skenario Baseline Original)

| No | Nama Tokoh Publik | Jumlah Citra | Deteksi MTCNN | Deteksi RetinaFace | Rata-rata Waktu MTCNN (s) | Rata-rata Waktu RetinaFace (s) | Rata-rata IoU |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Andre Agassi | 5 | 5/5 (100%) | 5/5 (100%) | 0.0699 | 1.6274 | 0.8767 |
| 2 | Ariel Sharon | 5 | 5/5 (100%) | 5/5 (100%) | 0.0607 | 1.5644 | 0.8857 |
| 3 | Arnold Schwarzenegger | 5 | 5/5 (100%) | 5/5 (100%) | 0.0479 | 1.3801 | 0.9181 |
| 4 | Colin Powell | 5 | 5/5 (100%) | 5/5 (100%) | 0.0828 | 1.6492 | 0.9024 |
| 5 | Donald Rumsfeld | 5 | 5/5 (100%) | 5/5 (100%) | 0.0522 | 1.5398 | 0.8989 |
| 6 | George W Bush | 5 | 5/5 (100%) | 5/5 (100%) | 0.0763 | 1.7245 | 0.9065 |
| 7 | Gerhard Schroeder | 5 | 5/5 (100%) | 5/5 (100%) | 0.0553 | 1.5422 | 0.9332 |
| 8 | Gloria Macapagal Arroyo | 5 | 5/5 (100%) | 5/5 (100%) | 0.0516 | 1.5030 | 0.9226 |
| 9 | Hans Blix | 5 | 5/5 (100%) | 5/5 (100%) | 0.0754 | 1.7745 | 0.8930 |
| 10 | Hugo Chavez | 5 | 5/5 (100%) | 5/5 (100%) | 0.0471 | 1.5736 | 0.9057 |
| 11 | Jacques Chirac | 5 | 5/5 (100%) | 5/5 (100%) | 0.0546 | 1.5256 | 0.9111 |
| 12 | Jennifer Capriati | 5 | 5/5 (100%) | 5/5 (100%) | 0.0810 | 1.5066 | 0.8791 |
| 13 | Junichiro Koizumi | 5 | 5/5 (100%) | 5/5 (100%) | 0.0592 | 1.7885 | 0.9271 |
| 14 | Kofi Annan | 5 | 5/5 (100%) | 5/5 (100%) | 0.0676 | 1.6784 | 0.8982 |
| 15 | Laura Bush | 5 | 5/5 (100%) | 5/5 (100%) | 0.0555 | 1.3569 | 0.9210 |
| 16 | Luiz Inacio Lula da Silva | 5 | 5/5 (100%) | 5/5 (100%) | 0.0660 | 1.4760 | 0.8962 |
| 17 | Megawati Sukarnoputri | 5 | 5/5 (100%) | 5/5 (100%) | 0.0594 | 1.5827 | 0.9322 |
| 18 | Serena Williams | 5 | 5/5 (100%) | 5/5 (100%) | 0.0611 | 1.5021 | 0.9454 |
| 19 | Tony Blair | 5 | 5/5 (100%) | 5/5 (100%) | 0.0619 | 1.5489 | 0.9143 |
| 20 | Vladimir Putin | 5 | 5/5 (100%) | 5/5 (100%) | 0.0568 | 1.5900 | 0.8995 |

---

## 4.5 ANALISIS PEMBAHASAN MENDALAM

### 4.5.1 Ketahanan Deteksi Wajah (Robustness of Detection)
Kedua model mencatatkan **tingkat keberhasilan 100.0%** pada keempat skenario pengujian (total 400 uji coba untuk MTCNN dan 400 uji coba untuk RetinaFace). Tidak ditemukan kegagalan deteksi tunggal (*zero false negative*).
- **MTCNN**: Meskipun mengandalkan arsitektur *cascaded* yang lebih sederhana, tahap P-Net terbukti memiliki ambang toleransi yang sangat kokoh terhadap peningkatan luminansi (+30) maupun peregangan kontras ($\times 1.3$).
- **RetinaFace**: Modul *Feature Pyramid Network* (FPN) dan *Deformable Convolutional Networks* (DCN) mampu merekonstruksi representasi semantik multi-skala yang sangat stabil sehingga distorsi intensitas tidak merusak ekstraksi *anchor boxes*.

### 4.5.2 Pengaruh Preprocessing terhadap Efisiensi Waktu Komputasi
1. **Overhead Waktu Preprocessing**:
   - Operasi linier seperti `Brightness` ($0.09\text{ ms}$) dan `Contrast` ($0.09\text{ ms}$) memiliki *computational cost* yang sangat dapat diabaikan ($\le 0.1\text{ ms}$), menjadikannya sangat aman disisipkan dalam *pipeline* berkecepatan tinggi tanpa membebani CPU.
   - Sebaliknya, `CLAHE` membutuhkan waktu rata-rata **$4.93\text{ ms}$** per citra. Durasi ini disebabkan oleh kompleksitas perhitungan histogram lokal secara berulang pada grid $8 \times 8$ serta interpolasi bilinear untuk menyatukan batas tile di kanal $L$, ditambah dua kali transformasi ruang warna ($\text{RGB} \to \text{LAB} \to \text{RGB}$).
2. **Dampak pada Waktu Inferensi MTCNN**:
   - Pada skenario `Brightness`, waktu inferensi MTCNN justru menurun dari **$0.0626\text{ s}$ menjadi $0.0553\text{ s}$** (lebih cepat $\approx 11.6\%$). Hal ini terjadi karena peningkatan kecerahan seragam mengurangi variansi intensitas pada area latar belakang gelap, sehingga proposal kandidat palsu pada tahap P-Net tereliminasi lebih cepat oleh NMS tahap pertama.
   - Pada skenario `CLAHE`, waktu inferensi MTCNN meningkat menjadi **$0.0752\text{ s}$** (bertambah $\approx 20\%$). Peningkatan kontras lokal mempertegas tekstur mikro pada latar belakang dan kerutan kulit, menyebabkan P-Net meloloskan lebih banyak kandidat kotak pembatas ke R-Net dan O-Net sebelum disaring.
3. **Stabilitas Waktu Inferensi RetinaFace**:
   - Waktu inferensi RetinaFace relatif konstan di rentang **$1.2872\text{ s} - 1.3240\text{ s}$** di semua skenario. Karena RetinaFace adalah detektor *dense single-stage*, jaringan selalu mengeksekusi operasi FPN dan konvolusi deformabel secara penuh pada seluruh peta fitur citra tanpa memperhitungkan jumlah kandidat awal (*fixed computational graph*).

### 4.5.3 Analisis Derajat Kesepakatan Spasial (Inter-Model IoU)
Rata-rata Inter-Model IoU tetap berada pada tingkat kesepakatan spasial yang sangat tinggi di semua skenario:
- **Original**: $0.9083$ (Baseline keselarasan tertinggi)
- **CLAHE**: $0.9040$ (Konsistensi spasial terjaga berkat preservasi krominansi LAB)
- **Brightness**: $0.9033$
- **Contrast**: $0.8985$

Peregangan kontras global ($\times 1.3$) menyebabkan penurunan IoU terbesar (menjadi $0.8985$), karena peningkatan gradien tepi luar rambut dan leher memicu RetinaFace untuk sedikit mempersempit margin vertikal (*tighter boundary*), sementara MTCNN mempertahankan kotak pembatas yang sedikit lebih longgar (*slack box*).

---

## 4.6 ANALISIS KUALITATIF 6 STUDI KASUS VISUAL

Berdasarkan evaluasi kualitatif terhadap 6 citra uji representatif:

| Kasus Uji | Nama File Citra | Karakteristik Visual | IoU Original | IoU CLAHE | Analisis Karakteristik Deteksi & Cropping |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Kasus 1** | `Colin_Powell_0001.jpg` | Wajah frontal simetris, pencahayaan studio merata | **0.9362** | **0.9410** | Bounding box kedua model berimpit sempurna. CLAHE memperjelas detail mata dan kontur hidung tanpa menimbulkan artefak warna pada hasil crop 160×160 px. |
| **Kasus 2** | `George_W_Bush_0002.jpg` | Pose miring (*yaw* $\approx 25^\circ$), bayangan pada pipi kanan | **0.9017** | **0.9150** | RetinaFace memperluas batas vertikal hingga dagu bawah. CLAHE berhasil mengangkat detail fitur pada area pipi yang berbayang sehingga landmark hidung lebih presisi. |
| **Kasus 3** | `Hans_Blix_0001.jpg` | Aksesoris kacamata tebal dengan kilatan lensa optik | **0.9029** | **0.8980** | Kilatan cahaya pada lensa kacamata tidak membingungkan landmark detector kedua model. Posisi pupil mata terdeteksi dengan tepat di dalam bingkai kacamata. |
| **Kasus 4** | `Jennifer_Capriati_0001.jpg` | Ekspresi dinamis senyum lebar, pipi terangkat | **0.8445** | **0.8520** | Perubahan geometri wajah akibat kontraksi otot pipi membuat RetinaFace memperluas box ke pelipis rambut samping, sedangkan MTCNN tetap terpusat pada kontur oval wajah bagian dalam. |
| **Kasus 5** | `Tony_Blair_0002.jpg` | Pose berbicara dengan mulut terbuka, dagu memanjang | **0.9486** | **0.9370** | Kedua arsitektur sangat akurat memetakan titik landmark bibir atas dan bibir bawah tanpa terdistorsi oleh bukaan rongga mulut. |
| **Kasus 6** | `Megawati_Sukarnoputri_0001.jpg` | Struktur rambut menutupi sebagian dahi dan telinga | **0.9221** | **0.9350** | Keberadaan poni rambut tidak mengganggu pendeteksian alis oleh P-Net MTCNN maupun FPN RetinaFace; cropping wajah 160×160 px terpusat simetris pada fitur wajah utama. |

---

## 4.7 ANALISIS PERBEDAAN KARAKTERISTIK BOUNDING BOX (*SLACK* VS *TIGHT*)

Perbedaan nilai IoU (berkisar antara $0.80 - 0.97$) bukan disebabkan oleh kegagalan deteksi, melainkan oleh filosofi desain batas kotak pembatas:
1. **Slack Bounding Box (MTCNN)**:
   - Koordinat regresi MTCNN cenderung menyertakan margin ekstra $\approx 5-10\%$ di sekitar dahi atas dan rahang luar.
   - **Kelebihan**: Sangat menguntungkan untuk pipeline pengenalan wajah hilir (*downstream face recognition* seperti FaceNet, VGGFace, atau ArcFace) yang membutuhkan representasi bentuk kepala dan batas rambut utuh.
2. **Tight Bounding Box (RetinaFace)**:
   - Koordinat RetinaFace sangat ketat mengurung batas terluar fitur anatomis wajah (alis, ujung hidung, dan dagu).
   - **Kelebihan**: Sangat ideal untuk estimasi pose kepala 3D (*head pose estimation*), analisis ekspresi wajah (FACS), dan verifikasi biometrik paspor/KTP di mana latar belakang harus dieliminasi total.

---

## 4.8 ANALISIS TRADE-OFF KOMPUTASI DAN REKOMENDASI IMPLEMENTASI

| Skenario Kebutuhan Sistem | Model yang Direkomendasikan | Teknik Preprocessing yang Direkomendasikan | Rasionalisasi Ilmiah |
| :--- | :---: | :---: | :--- |
| **Sistem Real-Time / Video Streaming (Webcam/CCTV)** | **MTCNN** | **None (Original) / Brightness Linier** | MTCNN menghasilkan latensi $\approx 55-62\text{ ms}$ (mampu beroperasi pada 16-18 FPS di CPU standar). Preprocessing linier hanya memakan overhead $0.09\text{ ms}$. |
| **Perangkat Edge / IoT / Embedded System (Raspberry Pi)** | **MTCNN** | **None (Original)** | Bobot model sangat ringan ($\approx 2\text{ MB}$), konsumsi RAM rendah, dan tidak membutuhkan komputasi CLAHE yang membebani memori terbatas. |
| **Citra Pencahayaan Buruk / Bayangan Kuat (Low-Light)** | **MTCNN / RetinaFace** | **CLAHE (Kanal L LAB)** | CLAHE secara adaptif mengangkat detail fitur pada area gelap tanpa merusak krominansi kulit, meningkatkan kualitas visual crop wajah untuk verifikasi. |
| **Forensik Biometrik / Verifikasi ID / Citra Statis Resolusi Tinggi** | **RetinaFace** | **CLAHE / Contrast** | Menghasilkan kotak pembatas anatomis yang sangat presisi (*tight fit*) dan landmark 5 titik yang sangat stabil terhadap variasi pose. |

---

## 4.9 ARTEFAK DATA LENGKAP & REPRODUKSIBILITAS

Seluruh data pengujian, kode implementasi, dan visualisasi telah disimpan secara terstruktur pada direktori repositori:
1. **Data Mentah Eksperimen Baseline**: [results/comparison.csv](file:///d:/chiesaaaa/Semester%209/pcd/results/comparison.csv)
2. **Data Mentah 400 Uji Coba Preprocessing**: [results/preprocessing_comparison.csv](file:///d:/chiesaaaa/Semester%209/pcd/results/preprocessing_comparison.csv)
3. **Rekapitulasi Agregasi Skenario**: [results/preprocessing_scenario_summary.csv](file:///d:/chiesaaaa/Semester%209/pcd/results/preprocessing_scenario_summary.csv)
4. **Visualisasi Grid Komparatif Preprocessing**: [results/preprocessing/visual_summary_scenarios.png](file:///d:/chiesaaaa/Semester%209/pcd/results/preprocessing/visual_summary_scenarios.png)
5. **Citra Hasil Deteksi & Crops 160×160**: Direktori `results/preprocessing/crops/` dan `results/preprocessing/detection/`
6. **Dashboard Interaktif**: Dapat diakses melalui [http://localhost:8502](http://localhost:8502) (Menu: `🧪 Eksperimen Preprocessing`).
