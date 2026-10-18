"""
Modul Utilitas Preprocessing Citra untuk Eksperimen PCD
Mendefinisikan fungsi-fungsi prapemrosesan:
1. Original (Baseline acuan)
2. Brightness adjustment (linear shift)
3. Contrast adjustment (linear scaling)
4. CLAHE pada kanal L ruang warna LAB (menjaga krominansi warna kulit)
"""

import cv2
import numpy as np

# Parameter Preprocessing Standar & Terdokumentasi
BRIGHTNESS_BETA = 30     # Nilai penambahan kecerahan linier (+30 intensitas piksel)
CONTRAST_ALPHA = 1.3     # Faktor pengali kontras global (alpha=1.3, beta=0)
CLAHE_CLIP_LIMIT = 2.0   # Batas pemotongan kontras lokal untuk membatasi amplifikasi noise
CLAHE_GRID_SIZE = (8, 8) # Ukuran tile kontekstual CLAHE (8x8 piksel blok)


def apply_original(img_rgb: np.ndarray) -> np.ndarray:
    """Mengembalikan salinan citra asli tanpa perubahan (Baseline)."""
    return img_rgb.copy()


def apply_brightness(img_rgb: np.ndarray, beta: int = BRIGHTNESS_BETA) -> np.ndarray:
    """
    Penyesuaian kecerahan (Brightness Adjustment).
    Transformasi: output(x,y) = clip(1.0 * pixel(x,y) + beta, 0, 255)
    Menggunakan cv2.convertScaleAbs untuk mencegah overflow/underflow uint8.
    """
    return cv2.convertScaleAbs(img_rgb, alpha=1.0, beta=beta)


def apply_contrast(img_rgb: np.ndarray, alpha: float = CONTRAST_ALPHA, beta: int = 0) -> np.ndarray:
    """
    Penyesuaian kontras (Contrast Adjustment).
    Transformasi: output(x,y) = clip(alpha * pixel(x,y) + beta, 0, 255)
    Menggunakan cv2.convertScaleAbs untuk mencegah clipping artefak yang berlebihan.
    """
    return cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)


def apply_clahe(img_rgb: np.ndarray, clip_limit: float = CLAHE_CLIP_LIMIT, tile_grid_size: tuple = CLAHE_GRID_SIZE) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Untuk citra berwarna, diterapkan khusus pada kanal Luminance (L)
    dalam ruang warna CIELAB (L*a*b*) agar persepsi warna kulit (A dan B)
    tidak terdistorsi atau memudar.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


PREPROCESSING_PIPELINES = {
    "Original": {
        "func": apply_original,
        "description": "Citra tanpa transformasi (Baseline acuan)",
        "params": "None"
    },
    "Brightness": {
        "func": apply_brightness,
        "description": f"Peningkatan kecerahan linier (alpha=1.0, beta={BRIGHTNESS_BETA})",
        "params": f"alpha=1.0, beta={BRIGHTNESS_BETA}"
    },
    "Contrast": {
        "func": apply_contrast,
        "description": f"Peningkatan kontras global (alpha={CONTRAST_ALPHA}, beta=0)",
        "params": f"alpha={CONTRAST_ALPHA}, beta=0"
    },
    "CLAHE": {
        "func": apply_clahe,
        "description": f"Peningkatan kontras adaptif kanal L ruang warna LAB (clipLimit={CLAHE_CLIP_LIMIT}, grid={CLAHE_GRID_SIZE})",
        "params": f"clipLimit={CLAHE_CLIP_LIMIT}, tileGridSize={CLAHE_GRID_SIZE}"
    }
}
