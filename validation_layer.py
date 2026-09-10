from pathlib import Path
import json, re

DB_PATH = Path(__file__).with_name("demo_identity_database.json")

def norm(v):
    return re.sub(r"\s+", " ", str(v or "")).strip().casefold()

def doc_key(v):
    return re.sub(r"[^A-Z0-9]", "", str(v or "").upper())

def load_db():
    if not DB_PATH.exists():
        return []
    try:
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

def detect_document_type(text):
    u = str(text or "").upper()
    if ("AADHAAR" in u or "UIDAI" in u or "UNIQUE IDENTIFICATION" in u
        or re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", u)):
        return "Aadhaar Card"
    if ("PERMANENT ACCOUNT NUMBER" in u or "INCOME TAX DEPARTMENT" in u
        or re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", u)):
        return "PAN Card"
    if "PASSPORT" in u:
        return "Passport"
    if "DRIVING LICENCE" in u or "DRIVING LICENSE" in u or re.search(r"\bDL\s*(NO|NUMBER)\b", u):
        return "Driving Licence"
    return "Unknown"

def identity_candidates(text):
    t=str(text or "")
    out=[]
    for m in re.finditer(r"(?<!\d)(\d{4}\s?\d{4}\s?\d{4})(?!\d)", t):
        d=m.group(1).replace(" ","")
        out.append(("Aadhaar Card", "XXXX XXXX "+d[-4:]))
    for m in re.finditer(r"\b[A-Z]{5}\d{4}[A-Z]\b", t.upper()):
        out.append(("PAN Card", m.group(0)))
    for m in re.finditer(r"\b(?:P\d{7}|DL\d{6})\b", t.upper()):
        out.append(("Passport" if m.group(0).startswith("P") else "Driving Licence", m.group(0)))
    seen=set(); unique=[]
    for x in out:
        if x not in seen:
            seen.add(x); unique.append(x)
    return [{"type":a,"number":b} for a,b in unique]

def validate_identity(identity, selected_type, ocr_text=""):
    db=load_db()
    detected=detect_document_type(ocr_text)
    candidates=identity_candidates(ocr_text)
    issues=[]
    if detected!="Unknown" and selected_type and norm(detected)!=norm(selected_type):
        issues.append({"field":"document_type","selected":selected_type,"detected":detected})
    if len(candidates)>1:
        issues.append({"field":"multiple_identity","count":len(candidates),"candidates":candidates})
    submitted=identity.get("document_number") if isinstance(identity,dict) else None
    key=doc_key(submitted)
    record=next((r for r in db if key and doc_key(r["document_number"])==key),None)
    if record:
        for field in ("name","date_of_birth","gender"):
            value=identity.get(field)
            if value and norm(value) not in ("not detected","not confidently detected") and norm(value)!=norm(record.get(field)):
                issues.append({"field":field,"submitted":value,"database":record.get(field)})
    elif key or candidates:
        # A record missing from the prototype database does NOT prove the
        # document is fake. Treat it as manual review.
        issues.append({"field":"database_record","reason":"No matching prototype record"})
    status="PASS" if not issues else "FAIL"
    if any(i["field"]=="multiple_identity" for i in issues):
        status="MANUAL REVIEW"
    elif any(i["field"]=="database_record" for i in issues):
        status="MANUAL REVIEW"
    return {
        "status":status,
        "database_available":bool(db),
        "detected_document_type":detected,
        "matched_record":record,
        "multiple_identities":len(candidates)>1,
        "identity_candidates":candidates,
        "mismatches":issues
    }
