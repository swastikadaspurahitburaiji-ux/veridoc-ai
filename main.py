from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import io
import re
import json
import math
from pathlib import Path

import pytesseract
from pytesseract import Output

from PIL import (
    Image,
    ImageOps,
    ImageFilter,
    ImageStat,
    ImageChops,
)

# Prototype validation layer: synthetic reference database + consistency checks.
# This is intentionally separate from OCR/ML/forensics and uses no real government data.
try:
    from validation_layer import validate_identity
except Exception:
    validate_identity = None


# =========================================================
# VERIDOC AI API
# =========================================================

app = FastAPI(
    title="VERIDOC AI API",
    description="AI-assisted identity and document screening backend",
    version="1.2.0",
)


# =========================================================
# ML MODEL
# =========================================================
MODEL_PATH = Path(__file__).with_name("document_screening_model.json")
ML_MODEL = None

if MODEL_PATH.exists():
    try:
        ML_MODEL = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except Exception:
        ML_MODEL = None


def _sigmoid(value):
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def build_ml_features(ocr_result, forensic_result, identity_result):
    """Create the same feature set used during model training."""
    fields = [
        identity_result.get("name"),
        identity_result.get("date_of_birth"),
        identity_result.get("gender"),
        identity_result.get("document_number"),
    ]
    available = sum(
        1 for value in fields
        if value and str(value).strip().lower() not in {
            "not detected",
            "not confidently detected",
        }
    )
    field_completeness = available / len(fields)

    document_number_present = int(
        identity_result.get("document_number") not in {None, "", "Not detected"}
    )
    date_present = int(
        identity_result.get("date_of_birth") not in {None, "", "Not detected"}
    )
    gender_present = int(
        identity_result.get("gender") not in {None, "", "Not detected"}
    )

    authenticity_score = float(
        forensic_result.get("authenticity_score", 0)
    )

    # Prototype layout signal derived from the existing visual-signal score.
    layout_consistency = max(0.0, min(1.0, authenticity_score / 100.0))

    return [
        float(ocr_result.get("confidence", 0)),
        authenticity_score,
        float(forensic_result.get("anomaly_score", 0)),
        float(forensic_result.get("image_width", 0)),
        float(forensic_result.get("image_height", 0)),
        field_completeness,
        document_number_present,
        date_present,
        gender_present,
        layout_consistency,
    ]


def predict_with_ml(feature_values):
    """
    Run inference using the trained logistic-regression model.
    Returns None if the model file is not available.
    """
    if not ML_MODEL:
        return None

    means = ML_MODEL["feature_mean"]
    stds = ML_MODEL["feature_std"]
    weights = ML_MODEL["weights"]
    bias = ML_MODEL["bias"]

    z = float(bias)

    for value, mean, std, weight in zip(
        feature_values, means, stds, weights
    ):
        safe_std = float(std) if float(std) != 0 else 1.0
        z += ((float(value) - float(mean)) / safe_std) * float(weight)

    genuine_probability = _sigmoid(z)
    suspicious_probability = 1.0 - genuine_probability

    if genuine_probability >= 0.70:
        decision = "LIKELY GENUINE"
    elif genuine_probability <= 0.30:
        decision = "SUSPICIOUS"
    else:
        decision = "MANUAL REVIEW"

    return {
        "available": True,
        "model_type": ML_MODEL.get("model_type", "Logistic Regression"),
        "genuine_probability": round(genuine_probability * 100, 2),
        "suspicious_probability": round(suspicious_probability * 100, 2),
        "decision": decision,
        "trained_dataset_rows": ML_MODEL.get("dataset_rows"),
        "test_accuracy": round(
            float(ML_MODEL.get("test_accuracy", 0)) * 100, 2
        ),
    }


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "VERIDOC AI backend is running",
    }



# =========================================================
# IDENTITY EXTRACTION HELPERS
# =========================================================

def clean_spaces(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def mask_document_number(digits):
    digits = re.sub(r"\D", "", str(digits))
    if len(digits) == 12:
        return f"XXXX XXXX {digits[-4:]}"
    if len(digits) >= 4:
        return "X" * (len(digits) - 4) + digits[-4:]
    return "Not detected"


def is_name_candidate(candidate):
    candidate = clean_spaces(candidate)
    if not candidate or len(candidate) < 3 or len(candidate) > 60:
        return False

    # Avoid accepting numeric/address-like OCR fragments.
    letters = re.findall(r"[A-Za-z]", candidate)
    if len(letters) < 3:
        return False

    words = candidate.split()
    if not (2 <= len(words) <= 5):
        return False

    blocked = {
        "government", "india", "male", "female", "address",
        "dob", "date", "birth", "aadhaar", "uidai", "passport",
        "driving", "licence", "license", "pan", "card",
        "authority", "identification", "signature", "valid",
        "enrolment", "enrollment", "number", "west", "bengal",
    }
    if any(w.lower().strip(".,:;-") in blocked for w in words):
        return False

    # Reject candidates that look like a sentence or address.
    if any(ch.isdigit() for ch in candidate):
        return False
    if len(candidate) > 45:
        return False

    return True


def score_name_candidate(candidate, index, lines):
    candidate = clean_spaces(candidate)
    words = candidate.split()
    score = 0

    if 2 <= len(words) <= 4:
        score += 18

    if all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", w) for w in words):
        score += 10

    # Strong signal when candidate follows a Name/Nam label.
    if index > 0 and re.search(r"\b(?:name|nam)\b", lines[index - 1], re.I):
        score += 50

    # Strong signal when the candidate appears immediately before DOB.
    if index + 1 < len(lines) and re.search(r"(?:dob|d\/o\/b|date\s*of\s*birth)", lines[index + 1], re.I):
        score += 35

    # Identity-card text is generally near the beginning of the crop.
    score += max(0, 10 - index)

    return score


def find_best_name(text, priority_text=""):
    """
    Extract a name conservatively. Priority OCR is normally the small
    identity-card region where the name is printed, which is much more
    reliable than OCR over the whole photographed document.
    """
    sources = []
    if priority_text:
        sources.append(priority_text)
    sources.append(text)

    candidates = []

    for source_rank, source in enumerate(sources):
        lines = [clean_spaces(line) for line in source.splitlines()]
        lines = [line for line in lines if line]

        # 1) Explicit Name/Nam label.
        for i, line in enumerate(lines):
            m = re.search(
                r"(?:^|\b)(?:name|nam)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{2,})$",
                line,
                flags=re.I,
            )
            if m:
                candidate = clean_spaces(m.group(1))
                if is_name_candidate(candidate):
                    candidates.append((source_rank, 1000, candidate))

            # OCR sometimes separates the label and value:
            if re.fullmatch(r"(?:name|nam)\s*:?", line, flags=re.I) and i + 1 < len(lines):
                candidate = clean_spaces(lines[i + 1])
                if is_name_candidate(candidate):
                    candidates.append((source_rank, 950, candidate))

        # 2) A name immediately before DOB is a very strong layout signal.
        for i, line in enumerate(lines):
            if re.search(r"(?:dob|d\/o\/b|date\s*of\s*birth)", line, re.I):
                if i > 0:
                    candidate = clean_spaces(re.sub(r"^[^A-Za-z]+", "", lines[i - 1]))
                    if is_name_candidate(candidate):
                        candidates.append((source_rank, 900, candidate))

                # Tesseract may split a two-word name into two separate lines
                # (e.g. "Prasanta" / "Manna") immediately before DOB.
                if i >= 2:
                    first = clean_spaces(re.sub(r"^[^A-Za-z]+", "", lines[i - 2]))
                    second = clean_spaces(re.sub(r"^[^A-Za-z]+", "", lines[i - 1]))
                    if (
                        re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", first)
                        and re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", second)
                    ):
                        candidate = clean_spaces(f"{first} {second}")
                        if is_name_candidate(candidate):
                            candidates.append((source_rank, 980, candidate))

        # 3) General candidate scoring.
        for i, line in enumerate(lines):
            candidate = clean_spaces(re.sub(r"^[^A-Za-z]+", "", line))
            candidate = re.sub(r"\s{2,}", " ", candidate)
            if is_name_candidate(candidate):
                score = score_name_candidate(candidate, i, lines)
                candidates.append((source_rank, score, candidate))

    if not candidates:
        return "Not detected"

    # Prefer priority source first, then strongest evidence.
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return candidates[0][2]



# =========================================================
# OCR HELPERS
# =========================================================

def preprocess_image(image: Image.Image, mode: str = "normal", max_side: int = 1800) -> Image.Image:
    """Prepare an image for OCR while keeping cloud CPU usage reasonable."""
    image = ImageOps.exif_transpose(image).convert("RGB")

    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)

    if mode == "threshold":
        return gray.point(lambda p: 255 if p > 155 else 0)

    if mode == "sharp":
        return gray.filter(ImageFilter.SHARPEN)

    return gray


def _ocr_with_confidence(image: Image.Image, psm: int = 6):
    """Single Tesseract pass returning both text and confidence."""
    data = pytesseract.image_to_data(
        image,
        config=f"--psm {psm}",
        output_type=Output.DICT,
    )

    words = []
    confidences = []

    for i, value in enumerate(data.get("text", [])):
        value = clean_spaces(value)
        try:
            confidence = float(data.get("conf", ["-1"])[i])
        except (ValueError, TypeError, IndexError):
            confidence = -1

        if value:
            words.append(value)
        if confidence >= 0:
            confidences.append(confidence)

    text = "\n".join(words)
    confidence = round(
        sum(confidences) / len(confidences), 2
    ) if confidences else 0.0

    return text, confidence


def _crop_identity_region(image: Image.Image, document_type: str = ""):
    """
    Extract the high-value identity text block from a photographed
    Aadhaar-style card.  The user's sample has the name/DOB/gender in the
    lower mini-card; a tight crop plus explicit upscaling gives Tesseract
    enough character resolution to read that block reliably.

    For non-Aadhaar documents, keep the broader fallback crop.
    """
    width, height = image.size
    kind = (document_type or "").lower()

    if "aadhaar" in kind or "aadhar" in kind:
        left = int(width * 0.30)
        right = int(width * 0.70)
        top = int(height * 0.655)
        bottom = int(height * 0.725)
        crop = image.crop((left, top, right, bottom))

        # The identity text is physically small in a phone photograph.
        # Upscale before OCR instead of merely limiting the max side.
        crop = crop.resize(
            (crop.width * 4, crop.height * 4),
            Image.Resampling.LANCZOS,
        )
        return crop

    left = int(width * 0.18)
    right = int(width * 0.75)
    top = int(height * 0.60)
    bottom = int(height * 0.80)
    return image.crop((left, top, right, bottom))


def extract_text_from_image(image_bytes: bytes, document_type: str = ""):
    """
    Fast but stronger OCR strategy:
    1. OCR the main document area.
    2. OCR the compact lower identity region separately.
    3. Use the identity-region OCR as the preferred source for name/DOB/gender.

    This avoids trying to infer identity fields from a noisy full-frame photo.
    """
    try:
        original = Image.open(io.BytesIO(image_bytes))
        original = ImageOps.exif_transpose(original).convert("RGB")

        width, height = original.size

        # Main document crop: remove most background while retaining the card.
        main_left = int(width * 0.10)
        main_right = int(width * 0.90)
        main_top = int(height * 0.12)
        main_bottom = int(height * 0.82)

        main_crop = original.crop(
            (main_left, main_top, main_right, main_bottom)
        )

        main_prepared = preprocess_image(
            main_crop,
            "normal",
            max_side=1800,
        )

        main_text, main_confidence = _ocr_with_confidence(
            main_prepared,
            psm=6,
        )

        # Identity crop is smaller, so upscale it for much better character
        # separation. This is the key improvement for the user's photographed card.
        identity_crop = _crop_identity_region(original, document_type=document_type)
        identity_prepared = preprocess_image(
            identity_crop,
            "normal",
            max_side=1200,
        )

        identity_text, identity_confidence = _ocr_with_confidence(
            identity_prepared,
            psm=6,
        )

        combined_text = "\n".join(
            part for part in (main_text, identity_text) if part
        )

        confidences = [
            c for c in (main_confidence, identity_confidence)
            if c > 0
        ]
        combined_confidence = round(
            sum(confidences) / len(confidences),
            2,
        ) if confidences else 0.0

        return {
            "text": combined_text,
            "confidence": combined_confidence,
            "identity_text": identity_text,
            "identity_confidence": identity_confidence,
        }

    except Exception as error:
        raise Exception(f"OCR processing failed: {error}")



def extract_identity_fields(text: str, priority_text: str = ""):
    """Conservative OCR-based identity extraction."""
    lines = [
        clean_spaces(line)
        for line in text.splitlines()
        if clean_spaces(line)
    ]
    lower_text = text.lower()

    # Date of birth
    dob = "Not detected"
    dob_patterns = [
        r"\b(0?[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-](19|20)\d{2}\b",
        r"\b(19|20)\d{2}[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])\b",
    ]
    for pattern in dob_patterns:
        match = re.search(pattern, priority_text or text)
        if match:
            dob = match.group(0)
            break

    # Gender
    gender = "Not detected"
    gender_source = (priority_text or "") + "\n" + lower_text
    if re.search(r"\bfemale\b|\bwoman\b|\bfema1e\b", gender_source, re.I):
        gender = "Female"
    elif re.search(r"\bmale\b|\bman\b|\bma1e\b", gender_source, re.I):
        gender = "Male"

    # Document number
    document_number = "Not detected"
    aadhaar_matches = re.findall(r"(?<!\d)(?:\d[\s-]?){12}(?!\d)", text)
    for candidate in aadhaar_matches:
        digits = re.sub(r"\D", "", candidate)
        if len(digits) == 12:
            document_number = mask_document_number(digits)
            break

    # Name: priority identity crop first.
    name = find_best_name(text, priority_text)

    return {
        "name": name,
        "date_of_birth": dob,
        "gender": gender,
        "document_number": document_number,
    }


# =========================================================
# IMAGE FORENSICS
# =========================================================

def analyze_image_forensics(image_bytes: bytes):
    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        width, height = image.size
        megapixels = (width * height) / 1_000_000

        gray = ImageOps.grayscale(image)

        edges = gray.filter(
            ImageFilter.FIND_EDGES
        )

        edge_mean = ImageStat.Stat(
            edges
        ).mean[0]

        blurred = image.filter(
            ImageFilter.GaussianBlur(radius=2)
        )

        difference = ImageChops.difference(
            image,
            blurred,
        )

        diff_mean = sum(
            ImageStat.Stat(difference).mean
        ) / 3

        anomaly_score = 0

        if megapixels < 0.5:
            anomaly_score += 20
        elif megapixels < 1:
            anomaly_score += 10

        if edge_mean > 45:
            anomaly_score += 10

        if diff_mean > 18:
            anomaly_score += 15

        anomaly_score = min(
            anomaly_score,
            100,
        )

        authenticity_score = max(
            0,
            100 - anomaly_score,
        )

        if anomaly_score < 25:
            tampering_status = "PASS"
        elif anomaly_score < 50:
            tampering_status = "REVIEW"
        else:
            tampering_status = "SUSPICIOUS"

        return {
            "authenticity_score": round(
                authenticity_score,
                2,
            ),
            "tampering_status": tampering_status,
            "anomaly_score": round(
                anomaly_score,
                2,
            ),
            "image_width": width,
            "image_height": height,
        }

    except Exception as error:
        raise Exception(
            f"Image forensics analysis failed: {str(error)}"
        )


# =========================================================
# RISK SCORE
# =========================================================

def calculate_risk_score(
    ocr_confidence,
    authenticity_score,
    tampering_status,
):
    ocr_risk = max(
        0,
        100 - ocr_confidence,
    )

    authenticity_risk = max(
        0,
        100 - authenticity_score,
    )

    if tampering_status == "PASS":
        tampering_risk = 5
    elif tampering_status == "REVIEW":
        tampering_risk = 45
    else:
        tampering_risk = 85

    risk = (
        (ocr_risk * 0.25)
        + (authenticity_risk * 0.45)
        + (tampering_risk * 0.30)
    )

    return round(
        min(max(risk, 0), 100),
        2,
    )


def get_risk_level(risk_score):
    if risk_score < 25:
        return "LOW"
    elif risk_score < 50:
        return "MEDIUM"
    elif risk_score < 75:
        return "HIGH"
    return "CRITICAL"



# =========================================================
# RISK EXPLANATION
# =========================================================

def build_risk_explanation(
    ocr_confidence,
    authenticity_score,
    tampering_status,
    anomaly_score,
    identity_result,
):
    """
    Generate human-readable reasons behind the prototype risk score.
    These are decision-support explanations, not proof of authenticity.
    """
    reasons = []
    recommendations = []

    if tampering_status == "PASS":
        reasons.append("No significant visual anomaly signal was detected.")
    elif tampering_status == "REVIEW":
        reasons.append("Some visual signals require additional manual review.")
        recommendations.append("Review the original document image manually.")
    else:
        reasons.append("The prototype detected stronger visual anomaly signals.")
        recommendations.append("Do not rely on the automated result alone; perform manual verification.")

    if authenticity_score >= 80:
        reasons.append("The prototype visual-signal assessment is relatively favorable.")
    elif authenticity_score >= 50:
        reasons.append("The visual-signal assessment is moderate and should be reviewed.")
        recommendations.append("Compare the document with an authorized reference or verification source.")
    else:
        reasons.append("The visual-signal assessment is low.")
        recommendations.append("Perform manual verification before accepting the document.")

    if ocr_confidence >= 75:
        reasons.append("OCR confidence is high.")
    elif ocr_confidence >= 50:
        reasons.append("OCR confidence is moderate.")
        recommendations.append("Review extracted identity fields against the source document.")
    else:
        reasons.append("OCR confidence is low, so extracted text may contain recognition errors.")
        recommendations.append("Check the extracted name, date of birth, gender and document number manually.")

    if anomaly_score == 0:
        reasons.append("The prototype image anomaly score is 0.00.")
    elif anomaly_score < 25:
        reasons.append("The image anomaly signal is relatively low.")
    else:
        reasons.append(f"The image anomaly signal is {anomaly_score:.2f}.")

    missing_fields = []
    for field_name, field_value in identity_result.items():
        if not field_value or str(field_value).lower() in {
            "not detected",
            "not confidently detected",
        }:
            missing_fields.append(field_name.replace("_", " ").title())

    if missing_fields:
        reasons.append(
            "Some identity fields could not be confidently extracted: "
            + ", ".join(missing_fields)
            + "."
        )
        recommendations.append("Confirm missing identity fields from the original document.")

    if not recommendations:
        recommendations.append(
            "Use this prototype as a screening aid and follow the organization's verification process."
        )

    return {
        "summary": f"{get_risk_level(calculate_risk_score(ocr_confidence, authenticity_score, tampering_status))} risk based on available prototype signals.",
        "reasons": reasons,
        "recommendations": recommendations,
    }


# =========================================================
# PROTOTYPE DOCUMENT VALIDATION
# =========================================================

def detect_document_type(text: str):
    """Detect document type from OCR text using transparent prototype rules."""
    u = str(text or "").upper()
    if ("AADHAAR" in u or "UIDAI" in u or "UNIQUE IDENTIFICATION" in u
            or re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", u)):
        return "Aadhaar Card"
    if ("PERMANENT ACCOUNT NUMBER" in u or "INCOME TAX DEPARTMENT" in u
            or re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", u)):
        return "PAN Card"
    if "PASSPORT" in u:
        return "Passport"
    if ("DRIVING LICENCE" in u or "DRIVING LICENSE" in u
            or re.search(r"\bDL\s*(NO|NUMBER)\b", u)):
        return "Driving Licence"
    return "Unknown"


def local_basic_validation(identity_result, selected_type, ocr_text):
    """Fallback validation used if validation_layer.py is unavailable."""
    detected = detect_document_type(ocr_text)
    issues = []
    if detected != "Unknown" and selected_type and detected.casefold() != selected_type.casefold():
        issues.append({"field": "document_type", "selected": selected_type, "detected": detected})
    numbers = re.findall(r"(?<!\d)(?:\d[\s-]?){12}(?!\d)", str(ocr_text or ""))
    unique_numbers = {re.sub(r"\D", "", x) for x in numbers}
    if len(unique_numbers) > 1:
        issues.append({"field": "multiple_identity", "count": len(unique_numbers), "reason": "More than one Aadhaar-like identity number was detected."})
    return {
        "status": "PASS" if not issues else ("MANUAL REVIEW" if any(i["field"] == "multiple_identity" for i in issues) else "FAIL"),
        "database_available": False,
        "detected_document_type": detected,
        "matched_record": None,
        "multiple_identities": any(i["field"] == "multiple_identity" for i in issues),
        "identity_candidates": [],
        "mismatches": issues,
        "message": "Basic prototype consistency checks completed."
    }


# =========================================================
# UPLOAD + SCREENING
# =========================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form("Identity Document"),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    content_type = file.content_type or ""

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    if content_type == "application/pdf":
        return {
            "status": "received",
            "filename": file.filename,
            "content_type": content_type,
            "message": (
                "PDF received. Current prototype OCR "
                "supports JPG and PNG images."
            ),
            "ocr": {
                "text": "",
                "confidence": 0,
            },
            "identity": {
                "name": "Not detected",
                "date_of_birth": "Not detected",
                "gender": "Not detected",
                "document_number": "Not detected",
            },
            "forensics": {
                "authenticity_score": 0,
                "tampering_status": "REVIEW",
                "anomaly_score": 0,
                "image_width": 0,
                "image_height": 0,
            },
            "risk": {
                "score": 50,
                "level": "MEDIUM",
            },
        }

    # -----------------------------------------------------
    # Image validation
    # -----------------------------------------------------

    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload JPG or PNG.",
        )

    # -----------------------------------------------------
    # OCR
    # -----------------------------------------------------

    try:
        ocr_result = extract_text_from_image(
            file_bytes,
            document_type=document_type,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    # -----------------------------------------------------
    # Identity extraction
    # -----------------------------------------------------

    identity_result = extract_identity_fields(
        ocr_result["text"],
        priority_text=ocr_result.get("identity_text", ""),
    )

    # -----------------------------------------------------
    # Forensics
    # -----------------------------------------------------

    try:
        forensic_result = analyze_image_forensics(
            file_bytes
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    # -----------------------------------------------------
    # Risk
    # -----------------------------------------------------

    risk_score = calculate_risk_score(
        ocr_confidence=ocr_result["confidence"],
        authenticity_score=forensic_result[
            "authenticity_score"
        ],
        tampering_status=forensic_result[
            "tampering_status"
        ],
    )

    risk_level = get_risk_level(
        risk_score
    )

    risk_explanation = build_risk_explanation(
        ocr_confidence=ocr_result["confidence"],
        authenticity_score=forensic_result["authenticity_score"],
        tampering_status=forensic_result["tampering_status"],
        anomaly_score=forensic_result["anomaly_score"],
        identity_result=identity_result,
    )

    # -----------------------------------------------------
    # ML inference
    # -----------------------------------------------------
    ml_features = build_ml_features(
        ocr_result=ocr_result,
        forensic_result=forensic_result,
        identity_result=identity_result,
    )
    ml_result = predict_with_ml(ml_features)

    # -----------------------------------------------------
    # Document + database consistency validation
    # -----------------------------------------------------
    if validate_identity is not None:
        try:
            validation_result = validate_identity(
                identity=identity_result,
                selected_type=document_type,
                ocr_text=ocr_result["text"],
            )
        except Exception as error:
            validation_result = local_basic_validation(
                identity_result, document_type, ocr_result["text"]
            )
            validation_result["message"] = (
                "Validation fallback used because the prototype database layer could not be loaded."
            )
    else:
        validation_result = local_basic_validation(
            identity_result, document_type, ocr_result["text"]
        )

    # The final decision is server-generated. The frontend does not send or edit
    # extracted identity fields, which prevents client-side result manipulation.
    if validation_result.get("status") in {"FAIL", "MANUAL REVIEW"}:
        if ml_result and ml_result.get("decision") == "LIKELY GENUINE":
            ml_result = dict(ml_result)
            ml_result["decision"] = "MANUAL REVIEW"

    # -----------------------------------------------------
    # Final response
    # -----------------------------------------------------

    return {
        "status": "success",
        "filename": file.filename,
        "document_type": document_type,
        "content_type": content_type,
        "message": "Document screening completed successfully",
        "ocr": {
            "text": ocr_result["text"],
            "confidence": ocr_result["confidence"],
        },
        "identity": identity_result,
        "forensics": forensic_result,
        "risk": {
            "score": risk_score,
            "level": risk_level,
        },
        "risk_explanation": risk_explanation,
        "ml": ml_result,
        "verification_status": build_verification_status(
            document_type,
            validation_result.get("detected_document_type", "Unable to determine"),
            validation_result,
            forensic_result,
            ml_result,
        ),
        "validation": validation_result,
        "detected_document_type": validation_result.get("detected_document_type", "Unknown"),
    }



def build_verification_status(selected_type, detected_type, db_validation, forensics, ml_result):
    """Conservative prototype verdict logic.

    A missing reference-database record is NOT treated as proof of invalidity.
    """
    reasons = []

    if detected_type and detected_type != "Unable to determine":
        if selected_type and selected_type.lower() != detected_type.lower():
            reasons.append("Document type mismatch")

    if isinstance(db_validation, dict):
        db_status = str(db_validation.get("status", "")).upper()
        if db_status in {"MISMATCH", "DUPLICATE"}:
            reasons.append("Reference data mismatch or duplicate record")
        elif db_status in {"NOT_FOUND", "UNAVAILABLE"}:
            reasons.append("Reference record unavailable; manual review required")

    if isinstance(forensics, dict):
        tamper = str(forensics.get("tampering_status", "")).upper()
        if tamper in {"FAIL", "SUSPICIOUS", "TAMPERING_DETECTED"}:
            reasons.append("Possible document tampering")

    if isinstance(ml_result, dict):
        decision = str(ml_result.get("decision", "")).upper()
        if decision == "SUSPICIOUS":
            reasons.append("Screening model indicates possible concern")

    # Hard mismatch/forensic concern first.
    hard = any(x in reasons for x in [
        "Document type mismatch",
        "Reference data mismatch or duplicate record",
        "Possible document tampering",
    ])
    if hard:
        return {"status": "INVALID", "level": "HIGH CONCERN", "reasons": reasons}

    # No reference record or uncertain automated result => manual review.
    if any("manual review" in x.lower() for x in reasons):
        return {"status": "MANUAL REVIEW", "level": "REVIEW", "reasons": reasons}

    if any("possible concern" in x.lower() for x in reasons):
        return {"status": "MANUAL REVIEW", "level": "REVIEW", "reasons": reasons}

    return {"status": "SCREENING PASS", "level": "LOW CONCERN", "reasons": reasons}
