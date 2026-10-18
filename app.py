import os
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from facenet_pytorch import MTCNN
from retinaface import RetinaFace
from preprocessing_utils import (
    PREPROCESSING_PIPELINES,
    apply_original,
    apply_brightness,
    apply_contrast,
    apply_clahe,
    BRIGHTNESS_BETA,
    CONTRAST_ALPHA,
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE
)

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="PCD: Deteksi Wajah MTCNN vs RetinaFace",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SELECTED_DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset_selected")
FOTO_ASLI_DIR = os.path.join(WORKSPACE_DIR, "foto_asli_100")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "comparison.csv")
VISUALIZATION_PATH = os.path.join(RESULTS_DIR, "visual_comparison.png")

PREPROC_RESULTS_DIR = os.path.join(RESULTS_DIR, "preprocessing")
PREPROC_CSV_PATH = os.path.join(RESULTS_DIR, "preprocessing_comparison.csv")
PREPROC_SUMMARY_PATH = os.path.join(RESULTS_DIR, "preprocessing_scenario_summary.csv")
PREPROC_VIS_PATH = os.path.join(PREPROC_RESULTS_DIR, "visual_summary_scenarios.png")

RETINAFACE_DET_DIR = os.path.join(RESULTS_DIR, "retinaface", "detection")
RETINAFACE_CROP_DIR = os.path.join(RESULTS_DIR, "retinaface", "crops")
MTCNN_DET_DIR = os.path.join(RESULTS_DIR, "mtcnn", "detection")
MTCNN_CROP_DIR = os.path.join(RESULTS_DIR, "mtcnn", "crops")

CROP_SIZE = (160, 160)

# Custom CSS untuk tampilan premium dan bersih
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        color: white;
        margin-bottom: 12px;
    }
    .metric-card h4 {
        margin: 0;
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
        color: #38bdf8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHE MODEL DETECTOR
# ==============================================================================
@st.cache_resource(show_spinner="Memuat model deteksi wajah (MTCNN & RetinaFace)...")
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device)
    # Warmup dummy inference
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    _ = mtcnn.detect(dummy)
    _ = RetinaFace.detect_faces(dummy)
    return mtcnn, device


def compute_iou(boxA, boxB):
    """Menghitung Intersection over Union (IoU) antara dua bounding box [x1, y1, x2, y2]."""
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxB_area = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area
    return (inter_area / float(union_area)) if union_area > 0 else 0.0


def select_best_face(candidates, img_w, img_h):
    if not candidates:
        return None
    best_cand = None
    best_score = -1.0
    cx_img, cy_img = img_w / 2.0, img_h / 2.0
    for cand in candidates:
        x1, y1, x2, y2, score, landmarks = cand
        area = max(0, x2 - x1) * max(0, y2 - y1)
        cx_box = (x1 + x2) / 2.0
        cy_box = (y1 + y2) / 2.0
        dist = np.sqrt((cx_box - cx_img) ** 2 + (cy_box - cy_img) ** 2)
        rank_score = area / (dist + 10.0)
        if rank_score > best_score:
            best_score = rank_score
            best_cand = cand
    return best_cand


def run_mtcnn_inference(mtcnn_model, img_rgb):
    h, w, _ = img_rgb.shape
    t0 = time.perf_counter()
    boxes, probs, landmarks = mtcnn_model.detect(img_rgb, landmarks=True)
    infer_time = time.perf_counter() - t0

    candidates = []
    if boxes is not None and len(boxes) > 0:
        for i, (b, p) in enumerate(zip(boxes, probs)):
            if p is not None and p >= 0.5:
                lms = landmarks[i] if landmarks is not None and len(landmarks) > i else []
                candidates.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(p), lms))

    best = select_best_face(candidates, w, h)
    vis_img = img_rgb.copy()
    crop_img = None

    if best:
        x1, y1, x2, y2, score, lms = best
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Gambar Bounding Box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(vis_img, f"MTCNN: {score:.2f}", (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 140, 0), 1, cv2.LINE_AA)

        # Gambar Landmarks (5 titik wajah)
        if len(lms) > 0:
            for pt in lms:
                cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 3, (255, 230, 0), -1)

        # Crop & Resize
        crop_raw = img_rgb[y1:y2, x1:x2]
        if crop_raw.shape[0] > 0 and crop_raw.shape[1] > 0:
            crop_img = cv2.resize(crop_raw, CROP_SIZE, interpolation=cv2.INTER_LINEAR)

        return True, (x1, y1, x2, y2), score, infer_time, vis_img, crop_img
    return False, None, 0.0, infer_time, vis_img, None


def run_retinaface_inference(img_rgb):
    h, w, _ = img_rgb.shape
    t0 = time.perf_counter()
    rf_res = RetinaFace.detect_faces(img_rgb)
    infer_time = time.perf_counter() - t0

    candidates = []
    if isinstance(rf_res, dict) and len(rf_res) > 0:
        for k, val in rf_res.items():
            box = val.get("facial_area")
            score = val.get("score", 1.0)
            lms_dict = val.get("landmarks", {})
            lms = list(lms_dict.values()) if isinstance(lms_dict, dict) else []
            if box:
                candidates.append((int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(score), lms))

    best = select_best_face(candidates, w, h)
    vis_img = img_rgb.copy()
    crop_img = None

    if best:
        x1, y1, x2, y2, score, lms = best
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Gambar Bounding Box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 230, 115), 2)
        cv2.putText(vis_img, f"RetinaFace: {score:.2f}", (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 115), 1, cv2.LINE_AA)

        # Gambar Landmarks (5 titik wajah)
        if len(lms) > 0:
            for pt in lms:
                cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 3, (255, 60, 60), -1)

        # Crop & Resize
        crop_raw = img_rgb[y1:y2, x1:x2]
        if crop_raw.shape[0] > 0 and crop_raw.shape[1] > 0:
            crop_img = cv2.resize(crop_raw, CROP_SIZE, interpolation=cv2.INTER_LINEAR)

        return True, (x1, y1, x2, y2), score, infer_time, vis_img, crop_img
    return False, None, 0.0, infer_time, vis_img, None


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
def main():
    st.title("👤 Sistem Evaluasi Deteksi Wajah: MTCNN vs RetinaFace")
    st.caption("Proyek Mata Kuliah Pengolahan Citra Digital (PCD) — Analisis Performa & Komparasi Deteksi Wajah pada Dataset LFW People.")

    mtcnn_model, device_name = load_models()

    # Sidebar Navigasi
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/200px-Python-logo-notext.svg.png", width=60)
    st.sidebar.title("Navigasi Aplikasi")
    menu = st.sidebar.radio(
        "Pilih Menu:",
        [
            "🔍 Uji Deteksi Interaktif",
            "📊 Hasil Eksperimen 100 Citra",
            "🖼️ Galeri Dataset & Crop Wajah",
            "🧪 Eksperimen Preprocessing",
            "📖 Teori & Analisis Metode PCD"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Info Sistem")
    st.sidebar.info(f"**Device**: {device_name.upper()}\n\n**Ukuran Crop Target**: {CROP_SIZE[0]}×{CROP_SIZE[1]} px\n\n**Format Uji**: RGB 3-Channel")

    # ==========================================================================
    # MENU 1: UJI DETEKSI INTERAKTIF
    # ==========================================================================
    if menu == "🔍 Uji Deteksi Interaktif":
        st.subheader("🔍 Uji Komparasi Deteksi Wajah (Side-by-Side)")
        st.write("Bandingkan kemampuan MTCNN dan RetinaFace secara langsung pada citra pilihan Anda.")

        sumber_opsi = st.radio(
            "Pilih Sumber Citra:",
            ["📁 100 Dataset PCD Terpilih", "📂 4 Subfolder Foto Asli Mentah", "📤 Unggah Foto Sendiri", "📸 Kamera Langsung (Webcam)"],
            horizontal=True
        )

        pil_image = None
        info_label = ""

        # Opsi 1: 100 Dataset Terpilih
        if sumber_opsi == "📁 100 Dataset PCD Terpilih":
            if os.path.exists(SELECTED_DATASET_DIR):
                all_files = sorted([f for f in os.listdir(SELECTED_DATASET_DIR) if f.endswith(".jpg")])
                identities = sorted(list(set(["_".join(f.split("_")[:-1]) for f in all_files])))
                
                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    chosen_id = st.selectbox("Pilih Figur Publik (Identitas):", identities)
                with col_sel2:
                    id_images = [f for f in all_files if f.startswith(chosen_id + "_")]
                    chosen_file = st.selectbox("Pilih Foto:", id_images)

                img_path = os.path.join(SELECTED_DATASET_DIR, chosen_file)
                pil_image = Image.open(img_path).convert("RGB")
                info_label = f"Dataset LFW: {chosen_file}"
            else:
                st.error("Folder dataset terpilih belum ditemukan.")

        # Opsi 2: 4 Subfolder Mentah
        elif sumber_opsi == "📂 4 Subfolder Foto Asli Mentah":
            if os.path.exists(FOTO_ASLI_DIR):
                folders = ["folder_1", "folder_2", "folder_3", "folder_4"]
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    chosen_f = st.selectbox("Pilih Subfolder:", folders)
                with col_f2:
                    sub_path = os.path.join(FOTO_ASLI_DIR, chosen_f)
                    sub_images = sorted(os.listdir(sub_path)) if os.path.exists(sub_path) else []
                    chosen_img = st.selectbox("Pilih Foto Mentah:", sub_images)

                if chosen_img:
                    img_path = os.path.join(FOTO_ASLI_DIR, chosen_f, chosen_img)
                    pil_image = Image.open(img_path).convert("RGB")
                    info_label = f"Foto Mentah: {chosen_f}/{chosen_img}"
            else:
                st.error("Folder `foto_asli_100` tidak ditemukan.")

        # Opsi 3: Upload Foto
        elif sumber_opsi == "📤 Unggah Foto Sendiri":
            uploaded = st.file_uploader("Unggah gambar format JPG atau PNG:", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                pil_image = Image.open(uploaded).convert("RGB")
                info_label = f"Unggahan: {uploaded.name}"

        # Opsi 4: Webcam
        elif sumber_opsi == "📸 Kamera Langsung (Webcam)":
            cam_pic = st.camera_input("Ambil foto dari kamera perangkat Anda:")
            if cam_pic is not None:
                pil_image = Image.open(cam_pic).convert("RGB")
                info_label = "Tangkapan Kamera Langsung"

        # Proses Deteksi
        if pil_image is not None:
            img_rgb = np.array(pil_image)
            h_orig, w_orig, _ = img_rgb.shape

            st.markdown("---")
            run_btn = st.button("🚀 Jalankan Deteksi Wajah (RetinaFace & MTCNN)", type="primary", use_container_width=True)

            if run_btn:
                with st.spinner("Sedang memproses deteksi pada kedua model..."):
                    # MTCNN
                    mt_ok, mt_box, mt_score, mt_time, mt_vis, mt_crop = run_mtcnn_inference(mtcnn_model, img_rgb)
                    # RetinaFace
                    rf_ok, rf_box, rf_score, rf_time, rf_vis, rf_crop = run_retinaface_inference(img_rgb)

                # Layout 3 Kolom
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### 📷 Citra Input Asli")
                    st.image(pil_image, use_container_width=True, caption=f"{info_label} ({w_orig}×{h_orig} px)")
                    st.info(f"**Dimensi**: `{w_orig} × {h_orig}` piksel\n\n**Format**: RGB")

                with col2:
                    st.markdown("### 🔵 MTCNN (Multi-task CNN)")
                    st.image(mt_vis, use_container_width=True, caption="BBox (Oranye) & 5 Facial Landmarks (Kuning)")
                    
                    status_mt = "✅ Terdeteksi" if mt_ok else "❌ Gagal Deteksi"
                    st.markdown(f"**Status**: {status_mt}")
                    st.markdown(f"**Confidence**: `{mt_score:.4f}`")
                    st.markdown(f"**Bounding Box**: `{mt_box}`")
                    st.markdown(f"**Waktu Inferensi**: `{mt_time:.4f} detik`")

                    if mt_crop is not None:
                        st.markdown("**Hasil Crop (160×160 px):**")
                        st.image(mt_crop, width=130)

                with col3:
                    st.markdown("### 🟢 RetinaFace (Single-Stage)")
                    st.image(rf_vis, use_container_width=True, caption="BBox (Hijau) & 5 Facial Landmarks (Merah)")
                    
                    status_rf = "✅ Terdeteksi" if rf_ok else "❌ Gagal Deteksi"
                    st.markdown(f"**Status**: {status_rf}")
                    st.markdown(f"**Confidence**: `{rf_score:.4f}`")
                    st.markdown(f"**Bounding Box**: `{rf_box}`")
                    st.markdown(f"**Waktu Inferensi**: `{rf_time:.4f} detik`")

                    if rf_crop is not None:
                        st.markdown("**Hasil Crop (160×160 px):**")
                        st.image(rf_crop, width=130)

                # Evaluasi Komparasi
                st.markdown("---")
                st.subheader("⚡ Ringkasan Analisis Komparatif Citra Ini")
                m1, m2, m3, m4 = st.columns(4)

                m1.metric("Waktu MTCNN", f"{mt_time:.4f} s")
                m2.metric("Waktu RetinaFace", f"{rf_time:.4f} s")
                
                speed_ratio = (rf_time / mt_time) if mt_time > 0 else 0
                m3.metric("Rasio Kecepatan", f"{speed_ratio:.1f}x", "MTCNN Lebih Cepat" if speed_ratio >= 1 else "RetinaFace Lebih Cepat")

                iou_val = compute_iou(mt_box, rf_box) if (mt_ok and rf_ok) else 0.0
                m4.metric("IoU Bounding Box", f"{iou_val:.3f}", "Tumpang tindih tinggi" if iou_val > 0.75 else "Cukup berbeda")

    # ==========================================================================
    # MENU 2: HASIL EKSPERIMEN 100 CITRA
    # ==========================================================================
    elif menu == "📊 Hasil Eksperimen 100 Citra":
        st.subheader("📊 Laporan Hasil Eksperimen 100 Citra LFW People")
        st.write("Data aktual hasil pengujian menyeluruh pada 100 gambar uji dari 20 figur publik.")

        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            total_n = len(df)
            rf_succ = df["retinaface_detected"].sum()
            mt_succ = df["mtcnn_detected"].sum()
            rf_avg = df["retinaface_time"].mean()
            mt_avg = df["mtcnn_time"].mean()
            rf_tot = df["retinaface_time"].sum()
            mt_tot = df["mtcnn_time"].sum()

            # Kartu Metrik
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Gambar Uji", f"{total_n} Citra")
            c2.metric("Akurasi MTCNN", f"{(mt_succ/total_n)*100:.1f}%", f"{mt_succ}/{total_n} sukses")
            c3.metric("Akurasi RetinaFace", f"{(rf_succ/total_n)*100:.1f}%", f"{rf_succ}/{total_n} sukses")
            speed_mult = rf_avg / mt_avg if mt_avg > 0 else 0
            c4.metric("Kecepatan Rata-rata", f"{mt_avg:.4f}s vs {rf_avg:.4f}s", f"MTCNN {speed_mult:.1f}x lebih cepat")

            st.markdown("---")
            st.markdown("#### 📋 Tabel Rekapitulasi Metrik Kinerja")
            rekap_df = pd.DataFrame([
                {
                    "Metode": "MTCNN",
                    "Jumlah Gambar": total_n,
                    "Berhasil Deteksi": mt_succ,
                    "Gagal": total_n - mt_succ,
                    "Crop Berhasil": df["mtcnn_crop"].sum(),
                    "Total Waktu": f"{mt_tot:.2f} detik",
                    "Rata-rata Waktu / Citra": f"{mt_avg:.4f} detik"
                },
                {
                    "Metode": "RetinaFace",
                    "Jumlah Gambar": total_n,
                    "Berhasil Deteksi": rf_succ,
                    "Gagal": total_n - rf_succ,
                    "Crop Berhasil": df["retinaface_crop"].sum(),
                    "Total Waktu": f"{rf_tot:.2f} detik",
                    "Rata-rata Waktu / Citra": f"{rf_avg:.4f} detik"
                }
            ])
            st.table(rekap_df)

            # Chart Perbandingan Waktu
            st.markdown("---")
            st.markdown("#### 📈 Grafik Perbandingan Waktu Inferensi per Gambar (Detik)")
            chart_data = df[["image_name", "mtcnn_time", "retinaface_time"]].copy()
            chart_data = chart_data.rename(columns={
                "mtcnn_time": "MTCNN (detik)",
                "retinaface_time": "RetinaFace (detik)"
            })
            st.line_chart(chart_data.set_index("image_name"))

            # Tampilan Visual Comparison PNG
            if os.path.exists(VISUALIZATION_PATH):
                st.markdown("---")
                st.markdown("#### 🖼️ Grid Komparasi Visual 6 Sampel Representatif")
                st.image(VISUALIZATION_PATH, use_container_width=True, caption="Grid Komparasi Visual (Asli vs RetinaFace vs MTCNN)")

            # Tabel Detail & Filter
            st.markdown("---")
            st.markdown("#### 🔍 Eksplorasi Data Tabel 100 Citra")
            identities_list = ["Semua Identitas"] + sorted(df["identity"].unique().tolist())
            chosen_filter = st.selectbox("Filter berdasarkan Nama Tokoh:", identities_list)

            filtered_data = df if chosen_filter == "Semua Identitas" else df[df["identity"] == chosen_filter]
            st.dataframe(filtered_data, use_container_width=True)

            # Tombol Download CSV
            csv_bytes = filtered_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Unduh Data Hasil Evaluasi (CSV)",
                data=csv_bytes,
                file_name="hasil_evaluasi_pcd_mtcnn_vs_retinaface.csv",
                mime="text/csv"
            )
        else:
            st.warning("File `results/comparison.csv` belum ditemukan. Silakan jalankan `run_experiment.py` terlebih dahulu.")

    # ==========================================================================
    # MENU 3: GALERI DATASET & CROP
    # ==========================================================================
    elif menu == "🖼️ Galeri Dataset & Crop Wajah":
        st.subheader("🖼️ Galeri 100 Citra & Komparasi Crop Wajah")
        st.write("Jelajahi ke-100 citra hasil deteksi dan perbandingan ukuran bounding box serta pemotongan wajah (160×160).")

        if os.path.exists(SELECTED_DATASET_DIR):
            all_files = sorted([f for f in os.listdir(SELECTED_DATASET_DIR) if f.endswith(".jpg")])
            identities = sorted(list(set(["_".join(f.split("_")[:-1]) for f in all_files])))

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                sel_id = st.selectbox("Pilih Identitas / Tokoh:", identities, key="gal_id")
            with col_g2:
                sel_files = [f for f in all_files if f.startswith(sel_id + "_")]
                sel_file = st.selectbox("Pilih Foto:", sel_files, key="gal_file")

            # Load path masing-masing hasil
            orig_p = os.path.join(SELECTED_DATASET_DIR, sel_file)
            rf_det_p = os.path.join(RETINAFACE_DET_DIR, sel_file)
            rf_crop_p = os.path.join(RETINAFACE_CROP_DIR, sel_file)
            mt_det_p = os.path.join(MTCNN_DET_DIR, sel_file)
            mt_crop_p = os.path.join(MTCNN_CROP_DIR, sel_file)

            st.markdown("---")
            col_a, col_b, col_c, col_d, col_e = st.columns(5)

            with col_a:
                st.markdown("**1. Citra Asli**")
                if os.path.exists(orig_p):
                    st.image(orig_p, use_container_width=True, caption="LFW 250×250")
            with col_b:
                st.markdown("**2. RetinaFace BBox**")
                if os.path.exists(rf_det_p):
                    st.image(rf_det_p, use_container_width=True, caption="Deteksi BBox")
            with col_c:
                st.markdown("**3. RetinaFace Crop**")
                if os.path.exists(rf_crop_p):
                    st.image(rf_crop_p, use_container_width=True, caption="Crop 160×160")
            with col_d:
                st.markdown("**4. MTCNN BBox**")
                if os.path.exists(mt_det_p):
                    st.image(mt_det_p, use_container_width=True, caption="Deteksi BBox")
            with col_e:
                st.markdown("**5. MTCNN Crop**")
                if os.path.exists(mt_crop_p):
                    st.image(mt_crop_p, use_container_width=True, caption="Crop 160×160")

            # Cari data foto ini di CSV jika ada
            if os.path.exists(CSV_PATH):
                df = pd.read_csv(CSV_PATH)
                match = df[df["image_name"] == sel_file]
                if len(match) > 0:
                    row = match.iloc[0]
                    st.markdown("---")
                    st.markdown("#### Detail Citra Terpilih:")
                    st.write(f"- **RetinaFace**: Bounding Box: `{row['retinaface_bbox']}` | Waktu: `{row['retinaface_time']} s`")
                    st.write(f"- **MTCNN**: Bounding Box: `{row['mtcnn_bbox']}` | Waktu: `{row['mtcnn_time']} s`")
        else:
            st.error("Folder dataset belum ditemukan.")

    # ==========================================================================
    # MENU 4: EKSPERIMEN PREPROCESSING (BRIGHTNESS, CONTRAST, CLAHE)
    # ==========================================================================
    elif menu == "🧪 Eksperimen Preprocessing":
        st.subheader("🧪 Eksperimen Preprocessing Citra: Brightness, Contrast & CLAHE")
        st.caption("Mengevaluasi pengaruh teknik prapemrosesan citra digital terhadap performa deteksi dan cropping wajah (MTCNN vs RetinaFace) pada 100 citra LFW.")

        sub_tab1, sub_tab2 = st.tabs(["📊 Hasil Benchmark 4 Skenario", "🔬 Uji Preprocessing Interaktif (Live)"])

        with sub_tab1:
            st.markdown("### 📋 Hasil Evaluasi Kuantitatif 4 Skenario pada 100 Citra")
            st.info("💡 **Catatan Metrik**: Dataset LFW tidak menyediakan anotasi bounding box manual (ground truth). Metrik IoU yang dilaporkan adalah **Inter-Model IoU** (derajat kesepakatan spasial antara MTCNN dan RetinaFace), bukan akurasi mutlak terhadap ground truth.")

            if os.path.exists(PREPROC_SUMMARY_PATH) and os.path.exists(PREPROC_CSV_PATH):
                sum_df = pd.read_csv(PREPROC_SUMMARY_PATH)
                detail_df = pd.read_csv(PREPROC_CSV_PATH)

                # Tampilkan KPI Cards ringkasan per skenario
                kpi_cols = st.columns(4)
                for idx, scen in enumerate(["Original", "Brightness", "Contrast", "CLAHE"]):
                    scen_row = sum_df[sum_df["Scenario"] == scen]
                    if len(scen_row) > 0:
                        r = scen_row.iloc[0]
                        with kpi_cols[idx]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>{scen}</h4>
                                <div class="value">{r['Mean_Inter_Model_IoU']:.3f} <span style="font-size:0.9rem; color:#94a3b8;">IoU</span></div>
                                <div style="font-size:0.8rem; margin-top:4px; color:#cbd5e1;">
                                    <b>MTCNN Det:</b> {r['MTCNN_Detection_Rate_Pct']}% ({r['MTCNN_Avg_Infer_Sec']:.3f}s)<br>
                                    <b>RetinaFace:</b> {r['RetinaFace_Detection_Rate_Pct']}% ({r['RetinaFace_Avg_Infer_Sec']:.3f}s)<br>
                                    <b>Preproc:</b> {r['Avg_Preprocess_ms']:.2f} ms
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("#### 📋 Tabel Rekapitulasi Performa 4 Skenario")
                display_cols = {
                    "Scenario": "Skenario",
                    "Avg_Preprocess_ms": "Preproc (ms)",
                    "MTCNN_Detection_Rate_Pct": "MTCNN Det (%)",
                    "MTCNN_Avg_Infer_Sec": "MTCNN Infer (s)",
                    "MTCNN_Avg_Total_Sec": "MTCNN Total (s)",
                    "RetinaFace_Detection_Rate_Pct": "RF Det (%)",
                    "RetinaFace_Avg_Infer_Sec": "RF Infer (s)",
                    "RetinaFace_Avg_Total_Sec": "RF Total (s)",
                    "Mean_Inter_Model_IoU": "Inter-Model IoU"
                }
                st.dataframe(sum_df[list(display_cols.keys())].rename(columns=display_cols), use_container_width=True)

                # Grafik Komparasi Waktu dan IoU
                st.markdown("---")
                g_col1, g_col2 = st.columns(2)

                with g_col1:
                    st.markdown("##### ⏱️ Rata-rata Waktu Inferensi per Skenario (Detik)")
                    chart_time = sum_df.set_index("Scenario")[["MTCNN_Avg_Infer_Sec", "RetinaFace_Avg_Infer_Sec"]].rename(
                        columns={"MTCNN_Avg_Infer_Sec": "MTCNN (s)", "RetinaFace_Avg_Infer_Sec": "RetinaFace (s)"}
                    )
                    st.bar_chart(chart_time)

                with g_col2:
                    st.markdown("##### 🎯 Derajat Kesepakatan Spasial (Inter-Model IoU)")
                    chart_iou = sum_df.set_index("Scenario")[["Mean_Inter_Model_IoU"]].rename(
                        columns={"Mean_Inter_Model_IoU": "Rata-rata IoU"}
                    )
                    st.line_chart(chart_iou)

                # Visualisasi Grid
                if os.path.exists(PREPROC_VIS_PATH):
                    st.markdown("---")
                    st.markdown("#### 🖼️ Grid Komparasi Visual 4 Skenario Preprocessing")
                    st.image(PREPROC_VIS_PATH, use_container_width=True, caption="Visualisasi: Original vs Brightness vs Contrast vs CLAHE beserta Deteksi MTCNN dan RetinaFace")

                # Eksplorasi Data Tabel
                st.markdown("---")
                st.markdown("#### 🔍 Eksplorasi Data Rinci (400 Baris Pengujian)")
                f_scen = st.selectbox("Filter berdasarkan Skenario:", ["Semua Skenario"] + list(detail_df["scenario"].unique()))
                filtered_detail = detail_df if f_scen == "Semua Skenario" else detail_df[detail_df["scenario"] == f_scen]
                st.dataframe(filtered_detail, use_container_width=True)

                csv_prep_bytes = filtered_detail.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Unduh Data Lengkap Preprocessing (CSV)",
                    data=csv_prep_bytes,
                    file_name="preprocessing_comparison_pcd.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Data hasil eksperimen preprocessing sedang diproses atau belum dieksekusi. Silakan jalankan `python run_preprocessing_experiment.py`.")

        with sub_tab2:
            st.markdown("### 🔬 Uji Coba Preprocessing Interaktif Secara Langsung")
            st.write("Uji pengaruh teknik preprocessing terhadap bounding box dan landmark secara real-time pada citra pilihan Anda.")

            interactive_samples = [
                "Colin_Powell_0001.jpg",
                "George_W_Bush_0002.jpg",
                "Hans_Blix_0001.jpg",
                "Jennifer_Capriati_0001.jpg",
                "Tony_Blair_0002.jpg",
                "Megawati_Sukarnoputri_0001.jpg"
            ]
            interactive_source = st.selectbox("Pilih Sampel Citra Uji:", interactive_samples)

            sample_path = os.path.join(SELECTED_DATASET_DIR, interactive_source)
            if os.path.exists(sample_path):
                img_live_bgr = cv2.imread(sample_path)
                img_live_rgb = cv2.cvtColor(img_live_bgr, cv2.COLOR_BGR2RGB)

                # Kontrol Preprocessing
                p_col1, p_col2 = st.columns([1, 2])
                with p_col1:
                    chosen_method = st.radio("Metode Preprocessing:", ["Original", "Brightness", "Contrast", "CLAHE"])

                with p_col2:
                    if chosen_method == "Original":
                        st.info("Citra asli tanpa transformasi matematis.")
                        live_proc_rgb = apply_original(img_live_rgb)
                    elif chosen_method == "Brightness":
                        beta_val = st.slider("Nilai Beta (Kecerahan):", min_value=-50, max_value=80, value=BRIGHTNESS_BETA, step=5)
                        st.caption(f"Transformasi: output = clip(1.0 * pixel + {beta_val}, 0, 255)")
                        live_proc_rgb = apply_brightness(img_live_rgb, beta=beta_val)
                    elif chosen_method == "Contrast":
                        alpha_val = st.slider("Nilai Alpha (Kontras):", min_value=0.5, max_value=2.0, value=float(CONTRAST_ALPHA), step=0.1)
                        st.caption(f"Transformasi: output = clip({alpha_val} * pixel + 0, 0, 255)")
                        live_proc_rgb = apply_contrast(img_live_rgb, alpha=alpha_val)
                    elif chosen_method == "CLAHE":
                        clip_val = st.slider("Clip Limit CLAHE:", min_value=1.0, max_value=5.0, value=float(CLAHE_CLIP_LIMIT), step=0.5)
                        st.caption(f"CLAHE pada kanal L ruang warna LAB (clipLimit={clip_val}, tileGridSize=(8,8))")
                        live_proc_rgb = apply_clahe(img_live_rgb, clip_limit=clip_val)

                # Tombol Jalankan
                if st.button("🚀 Jalankan Deteksi pada Citra Ini", type="primary"):
                    with st.spinner("Menjalankan MTCNN & RetinaFace..."):
                        t_prep_start = time.perf_counter()
                        if chosen_method == "Original":
                            proc_test = apply_original(img_live_rgb)
                        elif chosen_method == "Brightness":
                            proc_test = apply_brightness(img_live_rgb, beta=beta_val)
                        elif chosen_method == "Contrast":
                            proc_test = apply_contrast(img_live_rgb, alpha=alpha_val)
                        elif chosen_method == "CLAHE":
                            proc_test = apply_clahe(img_live_rgb, clip_limit=clip_val)
                        p_time = time.perf_counter() - t_prep_start

                        mt_ok, mt_box, mt_sc, mt_tm, mt_vis, mt_crp = run_mtcnn_inference(mtcnn_model, proc_test)
                        rf_ok, rf_box, rf_sc, rf_tm, rf_vis, rf_crp = run_retinaface_inference(proc_test)

                    # Tampilkan Komparasi 3 Kolom
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.image(proc_test, use_container_width=True, caption=f"Citra Preprocessed ({chosen_method})")
                        st.caption(f"Waktu Preprocessing: {p_time*1000:.2f} ms")
                    with r2:
                        st.image(mt_vis, use_container_width=True, caption=f"MTCNN ({mt_tm:.3f} s)")
                        if mt_crp is not None:
                            st.image(mt_crp, width=120, caption="MTCNN Crop 160×160")
                    with r3:
                        st.image(rf_vis, use_container_width=True, caption=f"RetinaFace ({rf_tm:.3f} s)")
                        if rf_crp is not None:
                            st.image(rf_crp, width=120, caption="RetinaFace Crop 160×160")

                    # Metrik Seketika
                    st.markdown("---")
                    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                    res_col1.metric("Waktu Preprocessing", f"{p_time*1000:.2f} ms")
                    res_col2.metric("Waktu MTCNN (Total)", f"{(p_time+mt_tm):.4f} s", f"Inferensi: {mt_tm:.4f}s")
                    res_col3.metric("Waktu RetinaFace (Total)", f"{(p_time+rf_tm):.4f} s", f"Inferensi: {rf_tm:.4f}s")
                    iou_live = compute_iou(mt_box, rf_box) if (mt_ok and rf_ok) else 0.0
                    res_col4.metric("Inter-Model IoU", f"{iou_live:.4f}", "Kesepakatan Tinggi" if iou_live > 0.8 else "Cukup Berbeda")
            else:
                st.error("Citra sampel tidak ditemukan.")

    # ==========================================================================
    # MENU 5: TEORI & ANALISIS METODE PCD
    # ==========================================================================
    elif menu == "📖 Teori & Analisis Metode PCD":
        st.subheader("📖 Dasar Teori & Pembahasan Pengolahan Citra Digital")

        tab1, tab2, tab3 = st.tabs(["🔬 Teori MTCNN", "🧬 Teori RetinaFace", "⚖️ Analisis Komparatif"])

        with tab1:
            st.markdown("""
            ### Multi-Task Cascaded Convolutional Networks (MTCNN)
            MTCNN merupakan metode deteksi wajah berbasis *deep learning cascaded* yang membagi proses deteksi menjadi 3 tahap bertingkat (*cascaded framework*):

            1. **P-Net (Proposal Network)**:
               * Citra input diubah ke berbagai skala (*image pyramid*).
               * Jaringan *Fully Convolutional Network* (FCN) memindai citra secara cepat untuk menemukan kandidat area wajah awal (*bounding box candidates*) beserta vektor regresinya.
               * Kandidat yang tumpang tindih dipangkas menggunakan teknik **Non-Maximum Suppression (NMS)**.
            2. **R-Net (Refine Network)**:
               * Semua kandidat yang lolos dari P-Net diperiksa kembali oleh R-Net.
               * Menolak sejumlah besar kandidat palsu (*false positives*) dan memperhalus koordinat kotak pembatas.
            3. **O-Net (Output Network)**:
               * Tahap paling kompleks yang mengekstraksi representasi fitur lebih dalam.
               * Menghasilkan prediksi akhir *bounding box* wajah dan **5 Facial Landmark Points** (mata kiri, mata kanan, hidung, sudut mulut kiri, dan sudut mulut kanan).
            """)

        with tab2:
            st.markdown("""
            ### RetinaFace (Single-Stage Dense Face Localization)
            RetinaFace adalah detektor wajah *single-stage* mutakhir yang mengintegrasikan ekstraksi fitur multi-skala:

            1. **Feature Pyramid Network (FPN)**:
               * Membangun representasi piramida fitur dari skala rendah hingga tinggi.
               * Mampu mendeteksi wajah dengan variasi ukuran ekstrem (dari wajah sangat kecil hingga wajah jarak dekat).
            2. **Context Module & Deformable Convolutions**:
               * Memperluas *receptive field* agar model dapat mengenali konteks area kepala dan orientasi wajah.
            3. **Multi-Task Loss Supervision**:
               * Pelatihan dilakukan secara bersamaan (*multi-task*) meliputi:
                 * **Klasifikasi Wajah**: Menentukan area wajah vs latar belakang.
                 * **Regresi Bounding Box**: Memperkirakan koordinat kotak pembatas secara akurat.
                 * **Facial Landmarks**: Menentukan 5 titik penting wajah.
                 * **Dense 3D Regression**: Memperkirakan bentuk mesh 3D wajah (pada model penuh).
            """)

        with tab3:
            st.markdown("""
            ### ⚖️ Analisis Komparatif & Kesimpulan Praktikum PCD

            | Aspek Perbandingan | MTCNN | RetinaFace |
            | :--- | :--- | :--- |
            | **Arsitektur** | Cascaded 3-Stage (P-Net $\\to$ R-Net $\\to$ O-Net) | Single-Stage Feature Pyramid Network (FPN) |
            | **Kecepatan Inferensi** | **Sangat Cepat (~0.06 detik/citra)** | Lebih lambat (~1.57 detik/citra) |
            | **Penggunaan Resource** | Ringan, sangat optimal untuk CPU/Edge device | Membutuhkan komputasi lebih intensif |
            | **Kerapatan Bounding Box** | Cenderung sedikit lebih longgar mencakup rambut/dahi | Cenderung lebih rapat dan presisi ke kontur wajah |
            | **Kasus Penggunaan Ideal** | Sistem real-time, streaming webcam, video surveillance | Pengenalan wajah presisi tinggi, foto ID, dataset training |

            **Kesimpulan**:
            Untuk sistem yang membutuhkan kecepatan tinggi atau berjalan pada perangkat dengan spesifikasi terbatas (CPU), **MTCNN** adalah pilihan yang sangat efisien. Namun, jika prioritas utama adalah presisi pemotongan kontur wajah (*tight bounding box*) dan akurasi pada berbagai pose ekstrem, **RetinaFace** memberikan performa visual yang lebih unggul.
            """)


if __name__ == "__main__":
    main()
