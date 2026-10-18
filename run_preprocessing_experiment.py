"""
Skrip Eksperimen Pengaruh Preprocessing Citra (Brightness, Contrast, CLAHE)
terhadap Deteksi dan Cropping Wajah: MTCNN vs RetinaFace.

Mengevaluasi 4 Skenario pada 100 Citra LFW:
1. Original: Baseline tanpa perubahan
2. Brightness: Penyesuaian kecerahan linier (alpha=1.0, beta=30)
3. Contrast: Penyesuaian kontras global (alpha=1.3, beta=0)
4. CLAHE: Contrast Limited Adaptive Histogram Equalization pada kanal L ruang warna LAB (clipLimit=2.0, tileGridSize=(8,8))

Menghasilkan:
- results/preprocessing_comparison.csv (400 baris data rinci)
- results/preprocessing_scenario_summary.csv (tabel rekapitulasi performa per skenario)
- results/preprocessing/visual_summary_scenarios.png (visualisasi perbandingan komparatif)
- Direktori artefak crops (160x160) dan visualisasi deteksi terpisah.
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN
from retinaface import RetinaFace

from preprocessing_utils import (
    PREPROCESSING_PIPELINES,
    apply_original,
    apply_brightness,
    apply_contrast,
    apply_clahe
)

# Konfigurasi Path Direktori
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SELECTED_DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset_selected")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
PREPROC_RESULTS_DIR = os.path.join(RESULTS_DIR, "preprocessing")

CSV_COMPARISON_PATH = os.path.join(RESULTS_DIR, "preprocessing_comparison.csv")
CSV_SUMMARY_PATH = os.path.join(RESULTS_DIR, "preprocessing_scenario_summary.csv")
VISUALIZATION_PATH = os.path.join(PREPROC_RESULTS_DIR, "visual_summary_scenarios.png")

CROP_SIZE = (160, 160)
SCENARIOS = ["Original", "Brightness", "Contrast", "CLAHE"]


def setup_directories():
    """Membuat struktur folder output yang terisolasi dan rapi."""
    os.makedirs(PREPROC_RESULTS_DIR, exist_ok=True)
    for scen in SCENARIOS:
        os.makedirs(os.path.join(PREPROC_RESULTS_DIR, "preprocessed_images", scen), exist_ok=True)
        os.makedirs(os.path.join(PREPROC_RESULTS_DIR, "crops", "mtcnn", scen), exist_ok=True)
        os.makedirs(os.path.join(PREPROC_RESULTS_DIR, "crops", "retinaface", scen), exist_ok=True)
        os.makedirs(os.path.join(PREPROC_RESULTS_DIR, "detection", "mtcnn", scen), exist_ok=True)
        os.makedirs(os.path.join(PREPROC_RESULTS_DIR, "detection", "retinaface", scen), exist_ok=True)
    print("[OK] Direktori hasil preprocessing telah disiapkan.")


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
    """
    Aturan seleksi wajah terbaik:
    Skor = Luas Bounding Box / (Jarak ke Pusat Citra + 10)
    Menjamin wajah subjek utama yang terfokus di tengah terpilih secara konsisten.
    """
    if not candidates:
        return None
    best_cand = None
    best_score = -1.0
    cx_img, cy_img = img_w / 2.0, img_h / 2.0
    for cand in candidates:
        x1, y1, x2, y2 = cand[0], cand[1], cand[2], cand[3]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        cx_box = (x1 + x2) / 2.0
        cy_box = (y1 + y2) / 2.0
        dist = np.sqrt((cx_box - cx_img) ** 2 + (cy_box - cy_img) ** 2)
        rank_score = area / (dist + 10.0)
        if rank_score > best_score:
            best_score = rank_score
            best_cand = cand
    return best_cand


def run_experiment():
    setup_directories()

    # Dapatkan daftar 100 citra
    if not os.path.exists(SELECTED_DATASET_DIR):
        print(f"[ERROR] Direktori {SELECTED_DATASET_DIR} tidak ditemukan!")
        sys.exit(1)

    image_files = sorted([f for f in os.listdir(SELECTED_DATASET_DIR) if f.endswith(".jpg")])
    total_imgs = len(image_files)
    print(f"[*] Ditemukan {total_imgs} citra uji pada {SELECTED_DATASET_DIR}")

    # Inisialisasi Model
    print("[*] Menginisialisasi model MTCNN dan RetinaFace...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn_model = MTCNN(keep_all=True, device=device)

    # Model warm-up
    dummy = np.zeros((250, 250, 3), dtype=np.uint8)
    _ = RetinaFace.detect_faces(dummy)
    _ = mtcnn_model.detect(dummy)
    print(f"[OK] Inisialisasi selesai. Device MTCNN: {device.upper()}")

    records = []
    start_all = time.time()

    print("\n" + "="*85)
    print(f"{'No':<4} | {'Citra':<28} | {'Skenario':<11} | {'Preproc':<8} | {'MTCNN':<12} | {'RetinaFace':<12} | {'IoU':<6}")
    print("="*85)

    for img_idx, fname in enumerate(image_files, 1):
        fpath = os.path.join(SELECTED_DATASET_DIR, fname)
        identity = "_".join(fname.split("_")[:-1])
        img_bgr_orig = cv2.imread(fpath)
        img_rgb_orig = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb_orig.shape

        for scen in SCENARIOS:
            # 1. Prapemrosesan Citra
            t_prep0 = time.perf_counter()
            prep_func = PREPROCESSING_PIPELINES[scen]["func"]
            img_rgb_proc = prep_func(img_rgb_orig)
            prep_time = time.perf_counter() - t_prep0
            img_bgr_proc = cv2.cvtColor(img_rgb_proc, cv2.COLOR_RGB2BGR)

            # Simpan citra hasil preprocessing (hanya untuk referensi visual)
            preproc_save_path = os.path.join(PREPROC_RESULTS_DIR, "preprocessed_images", scen, fname)
            cv2.imwrite(preproc_save_path, img_bgr_proc)

            # ------------------------------------------------------------------
            # 2. MTCNN INFERENCE
            # ------------------------------------------------------------------
            mt_detected = False
            mt_score = 0.0
            mt_bbox_str = "None"
            mt_box = None
            mt_crop_ok = False

            t_mt0 = time.perf_counter()
            boxes, probs, landmarks = mtcnn_model.detect(img_rgb_proc, landmarks=True)
            mt_infer_time = time.perf_counter() - t_mt0
            mt_total_time = prep_time + mt_infer_time

            mt_candidates = []
            if boxes is not None and len(boxes) > 0:
                for i, (b, p) in enumerate(zip(boxes, probs)):
                    if p is not None and p >= 0.5:
                        lms = landmarks[i] if landmarks is not None and len(landmarks) > i else []
                        mt_candidates.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(p), lms))

            mt_best = select_best_face(mt_candidates, w, h)
            if mt_best:
                mt_detected = True
                mx1, my1, mx2, my2, mt_score, mlms = mt_best
                mx1, my1 = max(0, mx1), max(0, my1)
                mx2, my2 = min(w, mx2), min(h, my2)
                mt_box = (mx1, my1, mx2, my2)
                mt_bbox_str = f"[{mx1},{my1},{mx2},{my2}]"

                # Visualisasi deteksi
                mt_vis_bgr = img_bgr_proc.copy()
                cv2.rectangle(mt_vis_bgr, (mx1, my1), (mx2, my2), (0, 140, 255), 2) # Orange BGR
                cv2.putText(mt_vis_bgr, f"MTCNN: {mt_score:.2f}", (mx1, max(15, my1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)
                if len(mlms) > 0:
                    for pt in mlms:
                        cv2.circle(mt_vis_bgr, (int(pt[0]), int(pt[1])), 2, (0, 230, 255), -1)
                cv2.imwrite(os.path.join(PREPROC_RESULTS_DIR, "detection", "mtcnn", scen, fname), mt_vis_bgr)

                # Crop & Resize
                crop_mt = img_bgr_proc[my1:my2, mx1:mx2]
                if crop_mt.shape[0] > 0 and crop_mt.shape[1] > 0:
                    crop_mt_resized = cv2.resize(crop_mt, CROP_SIZE, interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(PREPROC_RESULTS_DIR, "crops", "mtcnn", scen, fname), crop_mt_resized)
                    mt_crop_ok = True
            else:
                cv2.imwrite(os.path.join(PREPROC_RESULTS_DIR, "detection", "mtcnn", scen, fname), img_bgr_proc)

            # ------------------------------------------------------------------
            # 3. RETINAFACE INFERENCE
            # ------------------------------------------------------------------
            rf_detected = False
            rf_score = 0.0
            rf_bbox_str = "None"
            rf_box = None
            rf_crop_ok = False

            t_rf0 = time.perf_counter()
            rf_res = RetinaFace.detect_faces(img_rgb_proc)
            rf_infer_time = time.perf_counter() - t_rf0
            rf_total_time = prep_time + rf_infer_time

            rf_candidates = []
            if isinstance(rf_res, dict) and len(rf_res) > 0:
                for k, val in rf_res.items():
                    box = val.get("facial_area")
                    score = val.get("score", 1.0)
                    lms_dict = val.get("landmarks", {})
                    lms = list(lms_dict.values()) if isinstance(lms_dict, dict) else []
                    if box:
                        rf_candidates.append((int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(score), lms))

            rf_best = select_best_face(rf_candidates, w, h)
            if rf_best:
                rf_detected = True
                rx1, ry1, rx2, ry2, rf_score, rlms = rf_best
                rx1, ry1 = max(0, rx1), max(0, ry1)
                rx2, ry2 = min(w, rx2), min(h, ry2)
                rf_box = (rx1, ry1, rx2, ry2)
                rf_bbox_str = f"[{rx1},{ry1},{rx2},{ry2}]"

                # Visualisasi deteksi
                rf_vis_bgr = img_bgr_proc.copy()
                cv2.rectangle(rf_vis_bgr, (rx1, ry1), (rx2, ry2), (0, 230, 115), 2) # Green BGR
                cv2.putText(rf_vis_bgr, f"RetinaFace: {rf_score:.2f}", (rx1, max(15, ry1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 115), 1, cv2.LINE_AA)
                if len(rlms) > 0:
                    for pt in rlms:
                        cv2.circle(rf_vis_bgr, (int(pt[0]), int(pt[1])), 2, (60, 60, 255), -1)
                cv2.imwrite(os.path.join(PREPROC_RESULTS_DIR, "detection", "retinaface", scen, fname), rf_vis_bgr)

                # Crop & Resize
                crop_rf = img_bgr_proc[ry1:ry2, rx1:rx2]
                if crop_rf.shape[0] > 0 and crop_rf.shape[1] > 0:
                    crop_rf_resized = cv2.resize(crop_rf, CROP_SIZE, interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(PREPROC_RESULTS_DIR, "crops", "retinaface", scen, fname), crop_rf_resized)
                    rf_crop_ok = True
            else:
                cv2.imwrite(os.path.join(PREPROC_RESULTS_DIR, "detection", "retinaface", scen, fname), img_bgr_proc)

            # ------------------------------------------------------------------
            # 4. METRIK INTER-MODEL IoU
            # ------------------------------------------------------------------
            iou_val = compute_iou(mt_box, rf_box) if (mt_detected and rf_detected) else 0.0

            records.append({
                "image_name": fname,
                "identity": identity,
                "scenario": scen,
                "preprocess_time": round(prep_time, 5),
                "mtcnn_detected": mt_detected,
                "mtcnn_score": round(mt_score, 4),
                "mtcnn_infer_time": round(mt_infer_time, 4),
                "mtcnn_total_time": round(mt_total_time, 4),
                "mtcnn_crop": mt_crop_ok,
                "mtcnn_bbox": mt_bbox_str,
                "retinaface_detected": rf_detected,
                "retinaface_score": round(rf_score, 4),
                "retinaface_infer_time": round(rf_infer_time, 4),
                "retinaface_total_time": round(rf_total_time, 4),
                "retinaface_crop": rf_crop_ok,
                "retinaface_bbox": rf_bbox_str,
                "inter_model_iou": round(iou_val, 4)
            })

            status_mt = f"OK ({mt_infer_time:.3f}s)" if mt_detected else "FAIL"
            status_rf = f"OK ({rf_infer_time:.3f}s)" if rf_detected else "FAIL"
            iou_str = f"{iou_val:.3f}" if (mt_detected and rf_detected) else "N/A"
            print(f"{img_idx:<4} | {fname:<28} | {scen:<11} | {prep_time*1000:>5.2f}ms | {status_mt:<12} | {status_rf:<12} | {iou_str:<6}")

    elapsed_all = time.time() - start_all
    print("\n" + "="*85)
    print(f"[OK] Seluruh 400 evaluasi selesai dalam {elapsed_all/60:.2f} menit ({elapsed_all:.1f} detik).")

    # Simpan dataset CSV lengkap
    df = pd.DataFrame(records)
    df.to_csv(CSV_COMPARISON_PATH, index=False)
    print(f"[OK] File perbandingan lengkap disimpan di: {CSV_COMPARISON_PATH}")

    # Generate ringkasan statistik per skenario
    summary_df = generate_summary_table(df)
    summary_df.to_csv(CSV_SUMMARY_PATH, index=False)
    print(f"[OK] Ringkasan statistik per skenario disimpan di: {CSV_SUMMARY_PATH}")

    # Generate visualisasi perbandingan skenario
    generate_visual_summary()


def generate_summary_table(df):
    """Menghitung agregasi metrik per skenario dan per model."""
    summary_rows = []

    for scen in SCENARIOS:
        sub = df[df["scenario"] == scen]
        n = len(sub)

        # Preprocessing time
        avg_prep_ms = sub["preprocess_time"].mean() * 1000

        # MTCNN
        mt_succ = sub["mtcnn_detected"].sum()
        mt_crop = sub["mtcnn_crop"].sum()
        mt_avg_infer = sub["mtcnn_infer_time"].mean()
        mt_avg_total = sub["mtcnn_total_time"].mean()

        # RetinaFace
        rf_succ = sub["retinaface_detected"].sum()
        rf_crop = sub["retinaface_crop"].sum()
        rf_avg_infer = sub["retinaface_infer_time"].mean()
        rf_avg_total = sub["retinaface_total_time"].mean()

        # IoU
        both_detected = sub[sub["mtcnn_detected"] & sub["retinaface_detected"]]
        avg_iou = both_detected["inter_model_iou"].mean() if len(both_detected) > 0 else 0.0

        summary_rows.append({
            "Scenario": scen,
            "Total_Images": n,
            "Avg_Preprocess_ms": round(avg_prep_ms, 2),
            "MTCNN_Detected": mt_succ,
            "MTCNN_Detection_Rate_Pct": round((mt_succ / n) * 100, 1),
            "MTCNN_Crop_Rate_Pct": round((mt_crop / n) * 100, 1),
            "MTCNN_Avg_Infer_Sec": round(mt_avg_infer, 4),
            "MTCNN_Avg_Total_Sec": round(mt_avg_total, 4),
            "RetinaFace_Detected": rf_succ,
            "RetinaFace_Detection_Rate_Pct": round((rf_succ / n) * 100, 1),
            "RetinaFace_Crop_Rate_Pct": round((rf_crop / n) * 100, 1),
            "RetinaFace_Avg_Infer_Sec": round(rf_avg_infer, 4),
            "RetinaFace_Avg_Total_Sec": round(rf_avg_total, 4),
            "Mean_Inter_Model_IoU": round(avg_iou, 4)
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "="*85)
    print("                 RINGKASAN HASIL EVALUASI 4 SKENARIO PREPROCESSING")
    print("="*85)
    print(f"| {'Skenario':<11} | {'Preproc(ms)':<11} | {'MTCNN Det':<10} | {'MTCNN(s)':<9} | {'RF Det':<10} | {'RF(s)':<9} | {'IoU':<7} |")
    print("|" + "-"*13 + "|" + "-"*13 + "|" + "-"*12 + "|" + "-"*11 + "|" + "-"*12 + "|" + "-"*11 + "|" + "-"*9 + "|")
    for _, r in summary_df.iterrows():
        print(f"| {r['Scenario']:<11} | {r['Avg_Preprocess_ms']:>9.2f}ms | {r['MTCNN_Detected']:>3}/{r['Total_Images']} ({r['MTCNN_Detection_Rate_Pct']}%) | {r['MTCNN_Avg_Infer_Sec']:>7.4f}s | {r['RetinaFace_Detected']:>3}/{r['Total_Images']} ({r['RetinaFace_Detection_Rate_Pct']}%) | {r['RetinaFace_Avg_Infer_Sec']:>7.4f}s | {r['Mean_Inter_Model_IoU']:>7.4f} |")
    print("="*85 + "\n")

    return summary_df


def generate_visual_summary():
    """
    Membuat grid visualisasi 4 skenario untuk 4 contoh citra representatif:
    Baris: 4 sampel citra (Colin Powell, George W Bush, Hans Blix, Jennifer Capriati)
    Kolom:
      1) Citra Asli (Original)
      2) Preprocessing: Brightness
      3) Preprocessing: Contrast
      4) Preprocessing: CLAHE (LAB)
      5) Deteksi MTCNN (pada CLAHE)
      6) Deteksi RetinaFace (pada CLAHE)
    """
    print("[*] Membuat gambar visualisasi ringkasan skenario...")
    sample_images = [
        "Colin_Powell_0001.jpg",
        "George_W_Bush_0002.jpg",
        "Hans_Blix_0001.jpg",
        "Jennifer_Capriati_0001.jpg"
    ]

    fig, axes = plt.subplots(len(sample_images), 6, figsize=(18, 3.2 * len(sample_images)))
    headers = [
        "1. Original (Input)",
        "2. Brightness (+30)",
        "3. Contrast (x1.3)",
        "4. CLAHE (LAB)",
        "5. MTCNN Det (CLAHE)",
        "6. RetinaFace Det (CLAHE)"
    ]

    for col_idx, h in enumerate(headers):
        axes[0, col_idx].set_title(h, fontsize=11, fontweight="bold", pad=8)

    for row_idx, fname in enumerate(sample_images):
        orig_p = os.path.join(SELECTED_DATASET_DIR, fname)
        img_orig_bgr = cv2.imread(orig_p)
        img_orig_rgb = cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)

        # Preprocessed images
        img_bright = apply_brightness(img_orig_rgb)
        img_contrast = apply_contrast(img_orig_rgb)
        img_clahe = apply_clahe(img_orig_rgb)

        # Detection on CLAHE
        mt_det_p = os.path.join(PREPROC_RESULTS_DIR, "detection", "mtcnn", "CLAHE", fname)
        rf_det_p = os.path.join(PREPROC_RESULTS_DIR, "detection", "retinaface", "CLAHE", fname)

        img_mt_det = cv2.cvtColor(cv2.imread(mt_det_p), cv2.COLOR_BGR2RGB) if os.path.exists(mt_det_p) else img_clahe
        img_rf_det = cv2.cvtColor(cv2.imread(rf_det_p), cv2.COLOR_BGR2RGB) if os.path.exists(rf_det_p) else img_clahe

        row_imgs = [img_orig_rgb, img_bright, img_contrast, img_clahe, img_mt_det, img_rf_det]

        for col_idx, im in enumerate(row_imgs):
            ax = axes[row_idx, col_idx]
            ax.imshow(im)
            ax.axis("off")
            if col_idx == 0:
                short_name = fname.split("_00")[0].replace("_", " ")
                ax.text(-12, 125, short_name, rotation=90, va="center", ha="right", fontsize=10, fontweight="semibold")

    plt.tight_layout()
    plt.savefig(VISUALIZATION_PATH, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] Visualisasi ringkasan skenario disimpan di: {VISUALIZATION_PATH}")


if __name__ == "__main__":
    run_experiment()
