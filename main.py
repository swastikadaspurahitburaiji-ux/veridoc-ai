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


def extract_identity_fields(text: str):
    """Prototype rule-based identity extraction on top of OCR."""
    lines = [
        clean_spaces(line)
        for line in text.splitlines()
        if clean_spaces(line)
    ]
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
    aadhaar_matches = re.findall(r"(?<!\d)(?:\d[\s-]?){12}(?!\d)", text)
    for candidate in aadhaar_matches:
        digits = re.sub(r"\D", "", candidate)
        if len(digits) == 12:
            document_number = mask_document_number(digits)
            break

    if document_number == "Not detected":
        for line in lines:
            if "aadhaar" in line.lower():
                joined = "".join(re.findall(r"\d{4,}", line))
                if len(joined) >= 12:
                    document_number = mask_document_number(joined[-12:])
                    break

    # -----------------------------------------------------
    # Name - improved scoring
    # -----------------------------------------------------
    name = "Not detected"
    candidates = []

    # Explicit Name label always wins.
    for line in lines:
        match = re.search(
            r"(?:^|\b)(?:name|nam)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{2,})$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            candidate = clean_spaces(match.group(1))
            if is_name_candidate(candidate):
                candidates.append((1000, candidate))

    # Score all likely human-name lines.
    for index, line in enumerate(lines):
        candidate = clean_spaces(re.sub(r"^[^A-Za-z]+", "", line))

        # OCR sometimes appends fragments such as "Ne Relea Ao" after the name.
        candidate = re.split(
            r"\b(?:ne|relea|reel|ao)\b\s*:?",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        candidate = clean_spaces(candidate)

        if is_name_candidate(candidate):
            score = score_name_candidate(candidate, index, lines)
            candidates.append((score, candidate))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_name = candidates[0]
        if best_score >= 12:
            name = best_name

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
