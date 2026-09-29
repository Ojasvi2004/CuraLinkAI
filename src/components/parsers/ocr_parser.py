import os

# Must be set before Tesseract launches. On a fractional CPU, multi-threading
# only causes contention and makes Tesseract slower.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import re
import shutil

import cv2
import numpy as np
import pytesseract

# ---- Tunables (override with env vars on Render) ---------------------------
TARGET_WIDTH = int(os.getenv("OCR_TARGET_WIDTH", "2400"))   # upscale small images to this width
MAX_WIDTH = int(os.getenv("OCR_MAX_WIDTH", "3200"))         # downscale only above this width
MAX_UPSCALE = 4.0
MAX_PIXELS = int(os.getenv("OCR_MAX_PIXELS", "12000000"))   # hard RAM cap (~12 MB grayscale)
GOOD_CONF = float(os.getenv("OCR_GOOD_CONF", "75"))         # skip 2nd pass if mean conf >= this
TESS_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "120"))         # seconds per Tesseract call
# ----------------------------------------------------------------------------


def _configure_tesseract() -> None:
    """Configure pytesseract with a usable binary path when PATH is incomplete."""
    configured_path = os.getenv("TESSERACT_CMD")
    if configured_path and os.path.exists(configured_path):
        pytesseract.pytesseract.tesseract_cmd = configured_path
        return

    detected_path = shutil.which("tesseract")
    if detected_path:
        pytesseract.pytesseract.tesseract_cmd = detected_path
        return

    if os.name == "nt":
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                return

    raise RuntimeError(
        "Tesseract executable not found. Set TESSERACT_CMD or add tesseract to PATH."
    )


class OCRReports:
    def __init__(self, path: str, psm: int = 6):
        """
        psm 6 = one uniform block (good for lab tables / rows).
        Try psm 4 if your reports are mostly single-column paragraphs.
        """
        self.path = path
        self.config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
        _configure_tesseract()

    # ------------------------------------------------------------------ utils
    def _print_section(self, title: str, text: str) -> None:
        print(f"\n[OCR] {title}")
        print("[OCR]" + "-" * 60)
        print(text if text else "[OCR] <empty>")
        print("[OCR]" + "-" * 60)

    # ---------------------------------------------------------------- loading
    def load_image(self):
        print(f"[OCR] Loading image: {self.path}")
        # Load straight to grayscale: 1/3 the memory of a BGR image.
        img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found: {self.path}")
        print(f"[OCR] Image loaded: {img.shape}")
        return img

    # ---------------------------------------------------------- preprocessing
    def _normalize_size(self, img):
        """
        Tesseract is most accurate when text is ~300 DPI (page ~2400px wide).
        Small images are upscaled to that; huge ones are only shrunk if very large.
        A single well-chosen scale replaces the 3-scale loop of the heavy version.
        """
        h, w = img.shape
        if w < TARGET_WIDTH:
            scale = min(TARGET_WIDTH / w, MAX_UPSCALE)
        elif w > MAX_WIDTH:
            scale = MAX_WIDTH / w
        else:
            scale = 1.0

        if h * w * scale * scale > MAX_PIXELS:           # protect RAM
            scale = (MAX_PIXELS / float(h * w)) ** 0.5

        if abs(scale - 1.0) < 0.05:
            return img
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
        print(f"[OCR] Rescaled x{scale:.2f} -> {out.shape}")
        return out

    def _flatten_background(self, gray):
        """
        Removes shadows / uneven phone-camera lighting cheaply.
        Background is estimated on a 1/8-size copy, so it costs almost nothing
        (replaces the very slow bilateralFilter on a 3-4x upscaled image).
        """
        h, w = gray.shape
        small = cv2.resize(gray, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
        small = cv2.dilate(small, np.ones((5, 5), np.uint8))   # erases dark text strokes
        small = cv2.GaussianBlur(small, (0, 0), 3)
        bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.divide(gray, np.maximum(bg, 1), scale=255)

    def preprocess(self, gray):
        """Pass 1: clean grayscale (Tesseract's own binarisation is usually best)."""
        gray = self._normalize_size(gray)
        return self._flatten_background(gray)

    def preprocess_binary(self, norm):
        """Pass 2 (only if pass 1 was weak): adaptive threshold for faint/noisy scans."""
        blurred = cv2.medianBlur(norm, 3)
        return cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
        )

    # -------------------------------------------------------------------- OCR
    def _ocr(self, img):
        """
        One Tesseract call gives text AND confidence
        (the old code ran OCR twice: once just to measure confidence).
        Returns (text, mean_confidence, good_word_count).
        """
        data = pytesseract.image_to_data(
            img,
            config=self.config,
            output_type=pytesseract.Output.DICT,
            timeout=TESS_TIMEOUT,
        )
        lines, confs = {}, []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            if not word:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                continue
            if conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append((data["left"][i], word))
            confs.append(conf)

        text = "\n".join(
            " ".join(w for _, w in sorted(words)) for _, words in sorted(lines.items())
        )
        mean_conf = float(np.mean(confs)) if confs else 0.0
        good_words = sum(1 for c in confs if c >= 70)
        return text, mean_conf, good_words

    # --------------------------------------------------------------- cleaning
    def clean_text(self, text, keep_lines=True):
        # Only strip control characters. Medical reports need symbols like
        # ° µ ± = ' " ; | & ^ so we do NOT use a character whitelist.
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        if keep_lines:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n", text)
        else:
            text = re.sub(r"\s+", " ", text)
        return text.strip()

    # -------------------------------------------------------------------- run
    def run(self, keep_lines=True):
        gray = self.load_image()
        norm = self.preprocess(gray)
        del gray

        text, conf, good = self._ocr(norm)
        print(f"[OCR] Pass 1: conf={conf:.1f}, good words={good}")

        if conf < GOOD_CONF:
            print("[OCR] Low confidence -> trying adaptive-threshold pass")
            binary = self.preprocess_binary(norm)
            text2, conf2, good2 = self._ocr(binary)
            print(f"[OCR] Pass 2: conf={conf2:.1f}, good words={good2}")
            if good2 > good:
                text, conf = text2, conf2

        clean = self.clean_text(text, keep_lines=keep_lines)
        self._print_section("Processed OCR Output", clean)
        print(f"[OCR] Extracted {len(clean)} characters (confidence {conf:.1f})")
        return clean