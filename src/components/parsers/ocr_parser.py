
import cv2
import os
import re
import shutil

import numpy as np
import pytesseract


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
        windows_candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for candidate in windows_candidates:
            if os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                return

    raise RuntimeError(
        "Tesseract executable not found. Set TESSERACT_CMD or add tesseract to PATH."
    )


class OCRReports:
    def __init__(self, path: str):
        self.path = path
        _configure_tesseract()

    def _print_section(self, title: str, text: str) -> None:
        print(f"\n[OCR] {title}")
        print("[OCR]" + "-" * 60)
        print(text if text else "[OCR] <empty>")
        print("[OCR]" + "-" * 60)

    def load_image(self):
        print(f"[OCR] Loading image: {self.path}")
        img = cv2.imread(self.path)
        if img is None:
            raise ValueError(f"Image not found: {self.path}")
        print(f"[OCR] Image loaded successfully: {img.shape}")
        return img

    def detect_text_regions(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create(5, 30, 3000)
        regions, _ = mser.detectRegions(gray)
        boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            aspect = w / float(h) if h > 0 else 0
            if 0.1 < aspect < 15 and 8 < h < 80:
                boxes.append((x, y, w, h))
        return boxes

    def merge_boxes_into_zones(self, boxes, img_shape, margin=10):
        if not boxes:
            return []
        h_img, w_img = img_shape[:2]
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for (x, y, w, h) in boxes:
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(w_img, x + w + margin), min(h_img, y + h + margin)
            mask[y1:y2, x1:x2] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (margin * 2, margin))
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [(x, y, w, h) for cnt in contours
                for x, y, w, h in [cv2.boundingRect(cnt)] if w > 20 and h > 10]

    def preprocess(self, img, scale=3):
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel)
        _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def extract_roi(self, img):
        h, w = img.shape
        return img[int(h * 0.15):int(h * 0.95), :]

    def ocr_zoomed_patches(self, img):
        boxes = self.detect_text_regions(img)
        print(f"[OCR] Detected {len(boxes)} text regions")
        zones = self.merge_boxes_into_zones(boxes, img.shape)
        if not zones:
            zones = [(0, 0, img.shape[1], img.shape[0])]
        print(f"[OCR] OCR zones selected: {len(zones)}")
        zones = sorted(zones, key=lambda z: (z[1], z[0]))
        config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        patch_texts = []
        for (x, y, w, h) in zones:
            patch = img[y:y + h, x:x + w]
            area = w * h
            zoom = 6 if area < 2000 else 4 if area < 8000 else 3
            processed_patch = self.preprocess(patch, scale=zoom)
            text = pytesseract.image_to_string(processed_patch, config=config)
            print(f"[OCR] Patch ({x}, {y}, {w}, {h}) -> {len(text.strip())} chars")
            if text.strip():
                patch_texts.append(text.strip())
        raw_text = '\n'.join(patch_texts)
        self._print_section("Raw OCR Output", raw_text)
        return raw_text

    def extract_text_multiscale(self, img):
        scales = [2, 3, 4]
        results = []
        for scale in scales:
            print(f"[OCR] Running multi-scale OCR at scale {scale}")
            processed = self.preprocess(img, scale)
            roi = self.extract_roi(processed)
            config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            text = pytesseract.image_to_string(roi, config=config)
            print(f"[OCR] Scale {scale} produced {len(text.strip())} chars")
            results.append(text)
        raw_text = max(results, key=len)
        self._print_section("Raw OCR Output", raw_text)
        return raw_text

    def get_confidence(self, img, scale=3):
        processed = self.preprocess(img, scale)
        roi = self.extract_roi(processed)
        data = pytesseract.image_to_data(
            roi, config=r'--oem 3 --psm 6',
            output_type=pytesseract.Output.DICT
        )
        confidences = [int(c) for c in data['conf']
                       if str(c).lstrip('-').isdigit() and int(c) >= 0]
        return np.mean(confidences) if confidences else 0.0

    def clean_text(self, text):
       
        text = re.sub(r'[^\w\s./:,()\[\]<>+\-*%@#]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def run(self):
        img = self.load_image()
        confidence = self.get_confidence(img)
        print(f"[OCR] Confidence score: {confidence:.1f}")

        if confidence < 70:
            print("[OCR] Low confidence -> using zoomed patch OCR")
            text = self.ocr_zoomed_patches(img)
        else:
            print("[OCR] High confidence -> using multi-scale OCR")
            text = self.extract_text_multiscale(img)

        clean = self.clean_text(text)
        self._print_section("Processed OCR Output", clean)
        print(f"[OCR] Extracted {len(clean)} characters")
        return clean  
