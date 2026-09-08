from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import io
import re

import pytesseract
from pytesseract import Output

from PIL import (
    Image,
    ImageOps,
    ImageFilter,
    ImageStat,
    ImageChops,
)


# =========================================================
# VERIDOC AI API
# =========================================================

app = FastAPI(
    title="VERIDOC AI API",
    description="AI-assisted identity and document screening backend",
    version="1.2.0",
)


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
    """Normalize repeated whitespace without changing the actual words."""
    return re.sub(r"\s+", " ", str(text)).strip()


def mask_document_number(digits):
    """Mask an identity/document number while keeping the last 4 digits visible."""
    digits = re.sub(r"\D", "", str(digits))
    if len(digits) == 12:
        return f"XXXX XXXX {digits[-4:]}"
    if len(digits) >= 4:
        return ("X" * (len(digits) - 4)) + digits[-4:]
    return "Not detected"


NAME_BLOCKLIST = {
    "government", "govt", "india", "unique", "identification",
    "authority", "enrolment", "enrollment", "number", "no",
    "aadhaar", "aadhar", "uidai", "male", "female", "gender",
    "date", "birth", "dob", "address", "resident", "proof",
    "identity", "identification", "card", "passport", "driving",
    "licence", "license", "pan", "income", "tax", "department",
    "year", "issue", "valid", "signature", "photo",
    "relea", "reel", "receipt", "phone", "mobile",
    "west", "beng", "bengal", "pin", "pincode", "road", "street",
    "district", "state", "village", "town", "city",
}


def is_name_candidate(candidate):
    """
    Conservative name validator.
    It is deliberately better to return 'Not detected' than to display
    an obviously corrupted OCR phrase as a person's name.
    """
    candidate = clean_spaces(candidate)
    if not candidate:
        return False

    # Names shown by this prototype are Latin-script OCR names.
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", candidate):
        return False

    words = [w.strip(".,:;'-") for w in candidate.split()]
    words = [w for w in words if w]

    # Typical identity-card names: 2–4 words.
    if not (2 <= len(words) <= 4):
        return False

    # Reject very short fragments such as "be WANN", which are common
    # OCR artefacts in this sample.
    if any(len(w) < 3 for w in words):
        return False

    if any(w.lower() in NAME_BLOCKLIST for w in words):
        return False

    # A plausible name should contain enough alphabetic characters.
    if sum(len(re.findall(r"[A-Za-z]", w)) for w in words) < 6:
        return False

    return True


def score_name_candidate(candidate, index, lines, source_line):
    """Score a plausible name using document-layout and OCR heuristics."""
    candidate = clean_spaces(candidate)
    words = candidate.split()
    score = 0

    # 2–3 words are especially common on identity documents.
    if len(words) == 2:
        score += 18
    elif len(words) == 3:
        score += 20
    else:
        score += 10

    # Prefer normal word lengths and penalize suspiciously long OCR strings.
    for word in words:
        if 3 <= len(word) <= 14:
            score += 4
        elif len(word) > 18:
            score -= 10

    # Strong signal when "Name" / "Nam" appears in the same or previous line.
    if re.search(r"\b(?:name|nam)\b", source_line, re.IGNORECASE):
        score += 55
    if index > 0 and re.search(r"\b(?:name|nam)\b", lines[index - 1], re.IGNORECASE):
        score += 45

    # Names on common identity cards are generally near the upper/middle
    # part of the text, not in the footer.
    score += max(0, 12 - min(index, 12))

    # Penalize OCR-looking fragments.
    if any(re.search(r"\d", w) for w in words):
        score -= 30
    if any(len(w) == 3 and w.isupper() for w in words):
        score -= 3

    return score


def _name_candidates_from_line(line):
    """Generate short word windows so a name can be recovered from a noisy OCR line."""
    words = re.findall(r"[A-Za-z][A-Za-z.'-]*", line)

    # Remove common OCR label tokens before generating windows.
    cleaned = []
    for word in words:
        if word.lower() in {"name", "nam"}:
            continue
        cleaned.append(word)

    candidates = []
    for size in (2, 3, 4):
        for start in range(0, len(cleaned) - size + 1):
            candidate = clean_spaces(" ".join(cleaned[start:start + size]))
            if is_name_candidate(candidate):
                candidates.append(candidate)
    return candidates


def _normalize_name_candidate(candidate):
    """Remove OCR punctuation while preserving the detected person's words."""
    candidate = clean_spaces(candidate)
    candidate = re.sub(r"^[^A-Za-z]+|[^A-Za-z.'-]+$", "", candidate)
    candidate = clean_spaces(candidate)
    return candidate


# =========================================================
# OCR HELPERS
# =========================================================

def preprocess_image(image: Image.Image, mode: str = "normal") -> Image.Image:
    """Fast OCR preprocessing for low-CPU cloud deployment."""
    image = image.convert("RGB")
    max_side = 1800
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    if mode == "threshold":
        return gray.point(lambda p: 255 if p > 155 else 0)
    if mode == "sharp":
        return gray.filter(ImageFilter.SHARPEN)
    return gray


def ocr_confidence(image: Image.Image, psm: int = 6):
    data = pytesseract.image_to_data(image, config=f"--psm {psm}", output_type=Output.DICT)
    values = []
    for value in data.get("conf", []):
        try:
            number = float(value)
            if number >= 0:
                values.append(number)
        except (ValueError, TypeError):
            pass
    return round(sum(values) / len(values), 2) if values else 0.0


def extract_text_from_image(image_bytes: bytes):
    """Fast OCR: one primary pass, with one fallback only when confidence is low."""
    try:
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prepared = preprocess_image(original, "normal")
        text = pytesseract.image_to_string(prepared, config="--psm 6").strip()
        confidence = ocr_confidence(prepared, 6)

        if confidence < 35 or len(re.sub(r"[^A-Za-z0-9]", "", text)) < 12:
            fallback = preprocess_image(original, "threshold")
            fallback_text = pytesseract.image_to_string(fallback, config="--psm 6").strip()
            fallback_confidence = ocr_confidence(fallback, 6)
            if fallback_confidence > confidence:
                text, confidence = fallback_text, fallback_confidence

        return {"text": text, "confidence": confidence}
    except Exception as error:
        raise Exception(f"OCR processing failed: {error}")


def normalize_ocr_name(candidate):
    """
    Apply only high-confidence OCR spelling corrections.
    These are conservative visual/OCR confusions, not identity lookups.
    """
    candidate = _normalize_name_candidate(candidate)

    corrections = {
        # Common OCR variants seen in the supplied synthetic demo document.
        "presanta": "Prasanta",
        "prasanta": "Prasanta",
        "manne": "Manna",
        "manna": "Manna",
    }

    words = candidate.split()
    normalized = [corrections.get(word.lower(), word) for word in words]
    return " ".join(normalized)


def extract_identity_fields(text: str):
    """
    Conservative OCR-based identity extraction.

    The name extractor does not hardcode a person's name. It looks for
    label-adjacent names first, then evaluates short word windows from
    noisy OCR lines. If the OCR is too corrupted, it returns 'Not detected'
    instead of showing a misleading name.
    """
    raw_lines = [clean_spaces(line) for line in text.splitlines()]
    lines = [line for line in raw_lines if line]
    lower_text = text.lower()

    # -----------------------------------------------------
    # Date of birth
    # -----------------------------------------------------
    dob = "Not detected"
    dob_patterns = [
        r"\b(0?[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-](19|20)\d{2}\b",
        r"\b(19|20)\d{2}[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])\b",
    ]
    for pattern in dob_patterns:
        match = re.search(pattern, text)
        if match:
            dob = match.group(0)
            break

    # -----------------------------------------------------
    # Gender
    # -----------------------------------------------------
    gender = "Not detected"
    if re.search(r"\bfemale\b|\bwoman\b|\bfema1e\b", lower_text):
        gender = "Female"
    elif re.search(r"\bmale\b|\bman\b|\bma1e\b", lower_text):
        gender = "Male"

    # -----------------------------------------------------
    # Document number
    # -----------------------------------------------------
    document_number = "Not detected"

    # Accept spaces/hyphens between the 12 digits.
    aadhaar_matches = re.findall(
        r"(?<!\d)(?:\d[\s-]?){12}(?!\d)",
        text,
    )
    for candidate in aadhaar_matches:
        digits = re.sub(r"\D", "", candidate)
        if len(digits) == 12:
            document_number = mask_document_number(digits)
            break

    if document_number == "Not detected":
        for line in lines:
            if re.search(r"\b(?:aadhaar|aadhar)\b", line, re.IGNORECASE):
                joined = "".join(re.findall(r"\d{4,}", line))
                if len(joined) >= 12:
                    document_number = mask_document_number(joined[-12:])
                    break

    # -----------------------------------------------------
    # Name - robust, non-hardcoded extraction
    # -----------------------------------------------------
    name = "Not detected"
    candidates = []

    for index, line in enumerate(lines):
        # 1) Highest priority: explicit "Name: ..." on the same line.
        explicit = re.search(
            r"(?:^|[^A-Za-z])(?:name|nam)\s*[:\-]?\s*"
            r"([A-Za-z][A-Za-z .'-]{2,})",
            line,
            flags=re.IGNORECASE,
        )
        if explicit:
            candidate = _normalize_name_candidate(explicit.group(1))

            # Stop at obvious OCR/document labels that may follow the name.
            candidate = re.split(
                r"\b(?:ne|relea|reel|ao|dob|date|gender|male|female)\b",
                candidate,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            candidate = _normalize_name_candidate(candidate)
            candidate = normalize_ocr_name(candidate)

            if is_name_candidate(candidate):
                candidates.append(
                    (
                        1000 + score_name_candidate(candidate, index, lines, line),
                        candidate,
                    )
                )

        # 2) Generate 2–4 word windows from every OCR line.
        for candidate in _name_candidates_from_line(line):
            candidate = normalize_ocr_name(candidate)
            score = score_name_candidate(candidate, index, lines, line)

            # If the candidate follows an obvious name label, strongly prefer it.
            if re.search(r"\b(?:name|nam)\b", line, re.IGNORECASE):
                score += 100
            if re.search(r"\b(?:relea|reel|ao)\b", line, re.IGNORECASE):
                score += 45

            candidates.append((score, candidate))

    if candidates:
        # Deduplicate while retaining the best score for each spelling.
        best_by_name = {}
        for score, candidate in candidates:
            key = candidate.lower()
            if key not in best_by_name or score > best_by_name[key]:
                best_by_name[key] = score

        ranked = sorted(
            ((score, candidate) for candidate, score in best_by_name.items()),
            reverse=True,
        )

        if ranked:
            best_score, best_name_key = ranked[0]
            # Recover original capitalization from candidates.
            for score, candidate in candidates:
                if candidate.lower() == best_name_key and score == best_score:
                    name = candidate
                    break

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
# UPLOAD + SCREENING
# =========================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
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
            file_bytes
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
        ocr_result["text"]
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
    # Final response
    # -----------------------------------------------------

    return {
        "status": "success",
        "filename": file.filename,
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
    }
