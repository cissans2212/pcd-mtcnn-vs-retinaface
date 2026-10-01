import os
import io
import time
import zipfile
import tarfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from facenet_pytorch import MTCNN
import torch

# Konfigurasi Path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_PATH = os.path.join(WORKSPACE_DIR, "archive2.zip")
SELECTED_DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset_selected")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")

RETINAFACE_DET_DIR = os.path.join(RESULTS_DIR, "retinaface", "detection")
RETINAFACE_CROP_DIR = os.path.join(RESULTS_DIR, "retinaface", "crops")
MTCNN_DET_DIR = os.path.join(RESULTS_DIR, "mtcnn", "detection")
MTCNN_CROP_DIR = os.path.join(RESULTS_DIR, "mtcnn", "crops")

CSV_PATH = os.path.join(RESULTS_DIR, "comparison.csv")
VISUALIZATION_PATH = os.path.join(RESULTS_DIR, "visual_comparison.png")

# Ukuran Resize Seragam (Tahap 3)
CROP_SIZE = (160, 160)

# 20 Identitas Terpilih (Masing-masing 5 gambar = 100 gambar total)
TARGET_IDENTITIES = [
    'George_W_Bush', 'Colin_Powell', 'Tony_Blair', 'Donald_Rumsfeld', 'Gerhard_Schroeder',
    'Ariel_Sharon', 'Hugo_Chavez', 'Junichiro_Koizumi', 'Serena_Williams', 'Jacques_Chirac',
    'Vladimir_Putin', 'Luiz_Inacio_Lula_da_Silva', 'Gloria_Macapagal_Arroyo', 'Jennifer_Capriati', 'Arnold_Schwarzenegger',
    'Laura_Bush', 'Hans_Blix', 'Andre_Agassi', 'Megawati_Sukarnoputri', 'Kofi_Annan'
]


def setup_directories():
    for d in [SELECTED_DATASET_DIR, RETINAFACE_DET_DIR, RETINAFACE_CROP_DIR, MTCNN_DET_DIR, MTCNN_CROP_DIR]:
        os.makedirs(d, exist_ok=True)
    print("[OK] Direktori hasil telah disiapkan.")


def extract_100_images():
    print("[*] Mengekstrak 100 gambar terpilih dari archive2.zip...")
    selected_files = []
    with zipfile.ZipFile(ARCHIVE_PATH) as z:
        with z.open("lfw-funneled.tgz") as f:
            with tarfile.open(fileobj=f, mode="r:gz") as tar:
                all_members = {m.name: m for m in tar.getmembers() if m.isfile() and m.name.endswith(".jpg")}
                for identity in TARGET_IDENTITIES:
                    prefix = f"lfw_funneled/{identity}/"
                    id_members = sorted([m for name, m in all_members.items() if name.startswith(prefix)], key=lambda x: x.name)
                    chosen = id_members[:5]
                    for m in chosen:
                        fname = os.path.basename(m.name)
                        out_path = os.path.join(SELECTED_DATASET_DIR, fname)
                        if not os.path.exists(out_path):
                            data = tar.extractfile(m).read()
                            with open(out_path, "wb") as out_f:
                                out_f.write(data)
                        selected_files.append((identity, fname, out_path))
    print(f"[OK] Total 100 gambar berhasil diekstrak ke: {SELECTED_DATASET_DIR}")
    return selected_files


def select_best_face(candidates, img_w, img_h):
    """
    Aturan seleksi jika ada > 1 wajah:
    Pilih wajah dengan skor = Luas Bounding Box / (Jarak ke Pusat Citra + 10)
    Aturan ini menjamin wajah target utama (di tengah dan paling fokus) yang terpilih.
    """
    if not candidates:
        return None
    
    best_cand = None
    best_score = -1.0
    cx_img, cy_img = img_w / 2.0, img_h / 2.0
    
    for cand in candidates:
        x1, y1, x2, y2, score = cand
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
    image_list = extract_100_images()
    
    # Inisialisasi Model
    print("[*] Menginisialisasi model MTCNN dan RetinaFace...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn_model = MTCNN(keep_all=True, device=device)
    
    # Pemanasan model (warmup) agar load weights tidak memengaruhi benchmark waktu
    dummy_img = np.zeros((250, 250, 3), dtype=np.uint8)
    _ = RetinaFace.detect_faces(dummy_img)
    _ = mtcnn_model.detect(dummy_img)
    print(f"[OK] Model siap. MTCNN berjalan pada device: {device}")
    
    records = []
    
    print("\n" + "="*70)
    print(f"{'No':<4} | {'Image Name':<30} | {'RetinaFace':<12} | {'MTCNN':<12}")
    print("="*70)
    
    for idx, (identity, fname, fpath) in enumerate(image_list, 1):
        img_bgr = cv2.imread(fpath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_bgr.shape
        
        # ----------------------------------------------------
        # 1. RETINAFACE DETECTION
        # ----------------------------------------------------
        rf_detected = False
        rf_bbox_str = "None"
        rf_crop_ok = False
        t0 = time.perf_counter()
        rf_res = RetinaFace.detect_faces(img_rgb)
        rf_time = time.perf_counter() - t0
        
        rf_candidates = []
        if isinstance(rf_res, dict) and len(rf_res) > 0:
            for k, val in rf_res.items():
                box = val.get("facial_area") # [x1, y1, x2, y2]
                score = val.get("score", 1.0)
                if box:
                    rf_candidates.append((int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(score)))
        
        rf_best = select_best_face(rf_candidates, w, h)
        if rf_best:
            rf_detected = True
            rx1, ry1, rx2, ry2, rscore = rf_best
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)
            rf_bbox_str = f"[{rx1},{ry1},{rx2},{ry2}]"
            
            # Gambar bounding box pada salinan citra asli
            rf_det_img = img_bgr.copy()
            cv2.rectangle(rf_det_img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            label = f"RetinaFace: {rscore:.2f}"
            cv2.putText(rf_det_img, label, (rx1, max(15, ry1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(RETINAFACE_DET_DIR, fname), rf_det_img)
            
            # Crop & Resize
            crop_rf = img_bgr[ry1:ry2, rx1:rx2]
            if crop_rf.shape[0] > 0 and crop_rf.shape[1] > 0:
                crop_rf_resized = cv2.resize(crop_rf, CROP_SIZE, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(RETINAFACE_CROP_DIR, fname), crop_rf_resized)
                rf_crop_ok = True
        else:
            # Jika gagal deteksi, salin gambar asli
            cv2.imwrite(os.path.join(RETINAFACE_DET_DIR, fname), img_bgr)
            
        # ----------------------------------------------------
        # 2. MTCNN DETECTION
        # ----------------------------------------------------
        mt_detected = False
        mt_bbox_str = "None"
        mt_crop_ok = False
        t1 = time.perf_counter()
        boxes, probs = mtcnn_model.detect(img_rgb)
        mt_time = time.perf_counter() - t1
        
        mt_candidates = []
        if boxes is not None and len(boxes) > 0:
            for b, p in zip(boxes, probs):
                if p is not None and p >= 0.5:
                    mt_candidates.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(p)))
                    
        mt_best = select_best_face(mt_candidates, w, h)
        if mt_best:
            mt_detected = True
            mx1, my1, mx2, my2, mscore = mt_best
            mx1, my1 = max(0, mx1), max(0, my1)
            mx2, my2 = min(w, mx2), min(h, my2)
            mt_bbox_str = f"[{mx1},{my1},{mx2},{my2}]"
            
            # Gambar bounding box pada salinan citra asli
            mt_det_img = img_bgr.copy()
            cv2.rectangle(mt_det_img, (mx1, my1), (mx2, my2), (255, 165, 0), 2) # Orange/Cyan
            label = f"MTCNN: {mscore:.2f}"
            cv2.putText(mt_det_img, label, (mx1, max(15, my1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)
            cv2.imwrite(os.path.join(MTCNN_DET_DIR, fname), mt_det_img)
            
            # Crop & Resize
            crop_mt = img_bgr[my1:my2, mx1:mx2]
            if crop_mt.shape[0] > 0 and crop_mt.shape[1] > 0:
                crop_mt_resized = cv2.resize(crop_mt, CROP_SIZE, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(MTCNN_CROP_DIR, fname), crop_mt_resized)
                mt_crop_ok = True
        else:
            cv2.imwrite(os.path.join(MTCNN_DET_DIR, fname), img_bgr)
            
        records.append({
            "image_name": fname,
            "identity": identity,
            "retinaface_detected": rf_detected,
            "mtcnn_detected": mt_detected,
            "retinaface_time": round(rf_time, 4),
            "mtcnn_time": round(mt_time, 4),
            "retinaface_crop": rf_crop_ok,
            "mtcnn_crop": mt_crop_ok,
            "retinaface_bbox": rf_bbox_str,
            "mtcnn_bbox": mt_bbox_str
        })
        
        status_rf = f"OK ({rf_time:.3f}s)" if rf_detected else "FAILED"
        status_mt = f"OK ({mt_time:.3f}s)" if mt_detected else "FAILED"
        print(f"{idx:<4} | {fname:<30} | {status_rf:<12} | {status_mt:<12}")

    # Simpan CSV perbandingan
    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)
    print("\n" + "="*70)
    print(f"[OK] File perbandingan berhasil disimpan di: {CSV_PATH}")
    
    # ----------------------------------------------------
    # TAHAP 5 — VISUALISASI PERBANDINGAN
    # ----------------------------------------------------
    generate_visual_comparison(records)
    
    # ----------------------------------------------------
    # TAHAP 6 — RINGKASAN METRIK
    # ----------------------------------------------------
    generate_summary_table(df)


def generate_visual_comparison(records):
    print("[*] Membuat visualisasi perbandingan komparatif (Tahap 5)...")
    # Pilih 6 contoh representatif dengan berbagai kondisi:
    # 1. Frontal jelas: Colin_Powell_0001
    # 2. Pose miring/profile: George_W_Bush_0002
    # 3. Kacamata/aksesoris: Hans_Blix_0001
    # 4. Ekspresi tertawa/senyum: Jennifer_Capriati_0001
    # 5. Pose berbicara/mulut terbuka: Tony_Blair_0002
    # 6. Wanita dengan rambut/pencahayaan: Megawati_Sukarnoputri_0001
    sample_names = [
        "Colin_Powell_0001.jpg",
        "George_W_Bush_0002.jpg",
        "Hans_Blix_0001.jpg",
        "Jennifer_Capriati_0001.jpg",
        "Tony_Blair_0002.jpg",
        "Megawati_Sukarnoputri_0001.jpg"
    ]
    
    fig, axes = plt.subplots(len(sample_names), 5, figsize=(15, 3 * len(sample_names)))
    col_titles = [
        "Input Asli (LFW 250x250)",
        "RetinaFace BBox",
        "RetinaFace Crop (160x160)",
        "MTCNN BBox",
        "MTCNN Crop (160x160)"
    ]
    
    for i, col_name in enumerate(col_titles):
        axes[0, i].set_title(col_name, fontsize=12, fontweight='bold', pad=10)
        
    for row_idx, sname in enumerate(sample_names):
        # Asli
        orig_p = os.path.join(SELECTED_DATASET_DIR, sname)
        img_orig = cv2.cvtColor(cv2.imread(orig_p), cv2.COLOR_BGR2RGB)
        
        # RetinaFace BBox & Crop
        rf_det_p = os.path.join(RETINAFACE_DET_DIR, sname)
        img_rf_det = cv2.cvtColor(cv2.imread(rf_det_p), cv2.COLOR_BGR2RGB) if os.path.exists(rf_det_p) else img_orig
        
        rf_crop_p = os.path.join(RETINAFACE_CROP_DIR, sname)
        img_rf_crop = cv2.cvtColor(cv2.imread(rf_crop_p), cv2.COLOR_BGR2RGB) if os.path.exists(rf_crop_p) else np.zeros((160, 160, 3), dtype=np.uint8)
        
        # MTCNN BBox & Crop
        mt_det_p = os.path.join(MTCNN_DET_DIR, sname)
        img_mt_det = cv2.cvtColor(cv2.imread(mt_det_p), cv2.COLOR_BGR2RGB) if os.path.exists(mt_det_p) else img_orig
        
        mt_crop_p = os.path.join(MTCNN_CROP_DIR, sname)
        img_mt_crop = cv2.cvtColor(cv2.imread(mt_crop_p), cv2.COLOR_BGR2RGB) if os.path.exists(mt_crop_p) else np.zeros((160, 160, 3), dtype=np.uint8)
        
        imgs = [img_orig, img_rf_det, img_rf_crop, img_mt_det, img_mt_crop]
        
        for col_idx, im in enumerate(imgs):
            ax = axes[row_idx, col_idx]
            ax.imshow(im)
            ax.axis('off')
            if col_idx == 0:
                ax.text(-10, 125, sname.split('_00')[0].replace('_', ' '), rotation=90, va='center', ha='right', fontsize=11, fontweight='semibold')
                
    plt.tight_layout()
    plt.savefig(VISUALIZATION_PATH, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[OK] Visualisasi perbandingan berhasil disimpan di: {VISUALIZATION_PATH}")


def generate_summary_table(df):
    total_imgs = len(df)
    rf_detected_cnt = df['retinaface_detected'].sum()
    mt_detected_cnt = df['mtcnn_detected'].sum()
    
    rf_fail_cnt = total_imgs - rf_detected_cnt
    mt_fail_cnt = total_imgs - mt_detected_cnt
    
    rf_crop_cnt = df['retinaface_crop'].sum()
    mt_crop_cnt = df['mtcnn_crop'].sum()
    
    rf_total_time = df['retinaface_time'].sum()
    mt_total_time = df['mtcnn_time'].sum()
    
    rf_avg_time = df['retinaface_time'].mean()
    mt_avg_time = df['mtcnn_time'].mean()
    
    print("\n" + "="*50)
    print("           RINGKASAN HASIL EVALUASI")
    print("="*50)
    print(f"| {'Parameter':<24} | {'RetinaFace':>10} | {'MTCNN':>10} |")
    print("|" + "-"*26 + "|" + "-"*12 + "|" + "-"*12 + "|")
    print(f"| {'Jumlah gambar':<24} | {total_imgs:>10} | {total_imgs:>10} |")
    print(f"| {'Berhasil deteksi':<24} | {rf_detected_cnt:>10} | {mt_detected_cnt:>10} |")
    print(f"| {'Gagal deteksi':<24} | {rf_fail_cnt:>10} | {mt_fail_cnt:>10} |")
    print(f"| {'Crop berhasil':<24} | {rf_crop_cnt:>10} | {mt_crop_cnt:>10} |")
    print(f"| {'Total waktu (detik)':<24} | {rf_total_time:>10.3f} | {mt_total_time:>10.3f} |")
    print(f"| {'Rata-rata waktu/gambar':<24} | {rf_avg_time:>9.4f}s | {mt_avg_time:>9.4f}s |")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_experiment()
