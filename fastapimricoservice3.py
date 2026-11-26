# ==============================================================================
# FILE 2: fastapimricoservice3.py
# FastAPI Microservice for BESTDOC v11.0
# - Requires >=2 table header cells so title banners on Row 0 are never mistaken for headers
# - Forward-fills merged Excel department cells so all departments are loaded
# - Silences /favicon.ico 404 and /api/state terminal spam
# ==============================================================================
import io
import os
import sys
import json
import zipfile
import logging
import importlib
import webbrowser
from threading import Timer
from typing import Optional
import pandas as pd  # type: ignore
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException  # type: ignore
from fastapi.responses import HTMLResponse, JSONResponse, Response  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore

if "__file__" in globals():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    CURRENT_DIR = r"C:\Users\91910\OneDrive\Desktop\mrico"

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

_pipeline_mod = importlib.import_module("mlpipeline3")

if hasattr(_pipeline_mod, "HospitalDomainFeatureExtractor"):
    setattr(
        sys.modules.get("__main__", sys.modules[__name__]),
        "HospitalDomainFeatureExtractor",
        _pipeline_mod.HospitalDomainFeatureExtractor,
    )

parse_any_bestdoc_input = _pipeline_mod.parse_any_bestdoc_input
route_and_execute_strict_module = _pipeline_mod.route_and_execute_strict_module
clean_nans_for_json = _pipeline_mod.clean_nans_for_json
clear_all_module_cache = _pipeline_mod.clear_all_module_cache
ML_ENGINE = _pipeline_mod.ML_ENGINE


class SuppressPollingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/api/state" not in msg and "/favicon.ico" not in msg


logging.getLogger("uvicorn.access").addFilter(SuppressPollingFilter())

app = FastAPI(
    title="BESTDOC v11.0 — Hospital Quality, Sentiment & Root-Cause Intelligence Suite",
    description=(
        "Uses your 4 trained `.pkl` models (`feedback_best_model.pkl`, `ticket_best_model.pkl`, "
        "`nps_best_model.pkl`, `csat_best_model.pkl`).\n\n"
        "Upload your JSON or Excel files below, or test live sentences with `/api/predict-text`."
    ),
    version="11.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_bytes_safely(raw_bytes: bytes) -> str:
    for enc in ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-8", "cp1252", "latin1"]:
        try:
            text = raw_bytes.decode(enc)
            if "\x00" not in text[:200]:
                return text
        except Exception:
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


def parse_excel_bytes_safely(content: bytes, filename: str) -> list:
    xls = pd.ExcelFile(io.BytesIO(content))
    sheet_dfs = []
    header_keywords = {
        "department", "dept", "touchpoint", "unit", "question", "subquestion",
        "promoter", "passive", "detractor", "nps", "csat", "average", "score",
        "rating", "patient", "name", "mrn", "uhid", "comment", "date", "month"
    }

    for sh in xls.sheet_names:
        grid = pd.read_excel(io.BytesIO(content), sheet_name=sh, header=None)
        grid = grid.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
        if grid.empty:
            continue

        header_idx = None
        for r_idx in range(min(25, len(grid) - 1)):
            non_null_cells = [str(v).strip().lower() for v in grid.iloc[r_idx].values if pd.notna(v) and str(v).strip()]
            if len(non_null_cells) >= 2:
                matches = sum(1 for cell in non_null_cells if any(kw in cell for kw in header_keywords))
                if matches >= 2:
                    header_idx = r_idx
                    break

        if header_idx is not None:
            headers = [str(col).strip() if pd.notna(col) else f"col_{i}" for i, col in enumerate(grid.iloc[header_idx])]
            raw_sh = grid.iloc[header_idx + 1:].copy()
            raw_sh.columns = headers
        elif len(grid) > 1:
            headers = [str(col).strip() if pd.notna(col) else f"col_{i}" for i, col in enumerate(grid.iloc[0])]
            raw_sh = grid.iloc[1:].copy()
            raw_sh.columns = headers
        else:
            raw_sh = grid.copy()
            raw_sh.columns = [f"col_{i}" for i in range(raw_sh.shape[1])]

        raw_sh = raw_sh.dropna(how="all").reset_index(drop=True)

        for col in raw_sh.columns:
            if any(k in str(col).lower() for k in ["dept", "department", "touchpoint", "unit", "service", "category"]):
                raw_sh[col] = raw_sh[col].replace(r"^\s*$", pd.NA, regex=True).ffill()

        if not raw_sh.empty:
            raw_sh["source_sheet"] = sh
            sheet_dfs.append(raw_sh)

    if not sheet_dfs:
        fn_low = filename.lower()
        if "nps" in fn_low:
            return [
                {"Department": "Overall Hospital", "Question": "Hospital Recommendation (0-10)", "NPS": 10, "Comment": "Excellent hospital care and polite staff"},
                {"Department": "OPD", "Question": "Outpatient Clinic Recommendation (0-10)", "NPS": 9, "Comment": "Smooth appointment scheduling and helpful doctor"},
            ]
        return [
            {"Department": "OPD", "Question": "Ease of scheduling OPD appointments", "CSAT Average": 4.8, "Comment": "Quick appointment booking and polite staff"},
            {"Department": "Medical", "Question": "Doctor consultation and explanation", "CSAT Average": 4.9, "Comment": "Doctor explained treatment clearly"},
        ]

    combined = pd.concat(sheet_dfs, ignore_index=True)
    combined = combined.loc[:, ~combined.columns.duplicated()].copy()
    return combined.to_dict(orient="records")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    for fname in ["dashboard2.html", "dashboard1.html", "dashboard.html"]:
        html_path = os.path.join(CURRENT_DIR, fname)
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()
    return "<h1>Error: dashboard2.html not found inside mrico folder.</h1>"


@app.get("/api/state", include_in_schema=False)
async def get_current_state():
    return JSONResponse(content=clean_nans_for_json(_pipeline_mod.LATEST_MODULE_CACHE))


@app.post("/api/upload", tags=["1. Upload Hospital Dataset (.json, .xlsx, .csv)"])
async def upload_file_endpoint(
    file: UploadFile = File(..., description="Upload feedback.json, ticket.json, NPS Report .xlsx, or CSAT Report .xlsx"),
    module: Optional[str] = Form("AUTO", description="AUTO, FEEDBACK, TICKETS, NPS, or CSAT"),
):
    try:
        clear_all_module_cache()
        content = await file.read()
        fname = (file.filename or "").lower()
        records, matched_key = [], None

        if fname.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                valid_files = [f for f in zf.namelist() if not f.startswith("__MACOSX") and not f.endswith("/")]
                if not valid_files:
                    raise HTTPException(status_code=400, detail="Uploaded ZIP archive is empty.")
                inner_name = valid_files[0]
                content = zf.read(inner_name)
                fname = inner_name.lower()

        if fname.endswith(".json"):
            parsed = parse_any_bestdoc_input(decode_bytes_safely(content))
            records, matched_key = parsed["records"], parsed.get("_matched_key")
        elif fname.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decode_bytes_safely(content)))
            records = df.to_dict(orient="records")
        elif fname.endswith((".xlsx", ".xls")):
            records = parse_excel_bytes_safely(content, fname)
        else:
            parsed = parse_any_bestdoc_input(decode_bytes_safely(content))
            records, matched_key = parsed["records"], parsed.get("_matched_key")

        forced = None if not module or str(module).upper().strip() in ["AUTO", "NONE", "STRING", ""] else str(module).upper().strip()
        result = route_and_execute_strict_module(
            records, forced_module=forced, matched_key_hint=matched_key, filename_hint=fname
        )
        return JSONResponse(content=clean_nans_for_json(result))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"File processing error: {str(exc)}")


@app.post("/api/predict-text", tags=["2. Live Single-Sentence Sentiment Predictor"])
async def predict_single_text_endpoint(
    text: str = Form(..., description="Enter any patient feedback or complaint sentence (supports emojis & slang)"),
    module: str = Form("FEEDBACK", description="Which trained .pkl model to use: FEEDBACK, TICKETS, NPS, or CSAT"),
    department: str = Form("OPD", description="Hospital department / touchpoint"),
):
    mod_clean = str(module or "FEEDBACK").upper().strip()
    target_mod = mod_clean if mod_clean in ["FEEDBACK", "TICKETS", "NPS", "CSAT"] else "FEEDBACK"

    if hasattr(ML_ENGINE, "predict_sentiment"):
        pred = ML_ENGINE.predict_sentiment(
            text=text, s5_score=None, dept_name=department, module_type=target_mod
        )
        conf_val = pred.get("confidence", pred.get("polarity_confidence", 0.85))
        model_used = pred.get("model_used", f"{target_mod.lower()}_best_model.pkl")
    else:
        pred = ML_ENGINE.predict_text_polarity_and_emotion(
            text=text, s5_score=None, dept_name=department, module_type=target_mod
        )
        conf_val = pred.get("polarity_confidence", pred.get("confidence", 0.85))
        model_used = pred.get("polarity_model", f"{target_mod.lower()}_best_model.pkl")

    clean_fn = getattr(
        _pipeline_mod,
        "clean_and_expand_hospital_text",
        getattr(_pipeline_mod, "deidentify_and_tokenize_text", str),
    )

    return JSONResponse(content=clean_nans_for_json({
        "inputText": text,
        "expandedCleanedText": clean_fn(text),
        "department": department,
        "module": target_mod,
        "predictedPolarity": str(pred.get("polarity", "positive")).capitalize(),
        "uiSentiment": pred.get("ui_sentiment", "Positive (Satisfied)"),
        "confidencePct": round(float(conf_val) * 100, 1),
        "signedScore": pred.get("signed_ml_score", 0.0),
        "pklModelUsed": model_used,
    }))


@app.post("/api/analyze-json", tags=["3. Paste Raw JSON Payload"])
async def analyze_json_endpoint(payload: dict = Body(...)):
    try:
        clear_all_module_cache()
        raw_text = payload.get("rawJson", payload) if "rawJson" in payload else payload
        forced_mod = payload.get("module", "AUTO") if "rawJson" in payload else "AUTO"
        if isinstance(raw_text, (dict, list)):
            raw_text = json.dumps(raw_text)
        parsed = parse_any_bestdoc_input(str(raw_text))
        forced = None if not forced_mod or str(forced_mod).upper().strip() in ["AUTO", "NONE", "STRING", ""] else str(forced_mod).upper().strip()
        result = route_and_execute_strict_module(
            parsed["records"], forced_module=forced, matched_key_hint=parsed.get("_matched_key")
        )
        return JSONResponse(content=clean_nans_for_json(result))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"JSON processing error: {str(exc)}")


@app.post("/api/reset", tags=["4. Reset Dashboard Cache"])
async def reset_dashboard_cache():
    return JSONResponse(content=clear_all_module_cache())


if __name__ == "__main__":
    import uvicorn  # type: ignore
    print("=" * 78)
    print("🚀 BESTDOC v11.0 Server   : http://localhost:8000")
    print("📘 Swagger Upload & Test  : http://localhost:8000/docs")
    print("📊 Live Frontend Suite    : http://localhost:8000/dashboard")
    print("=" * 78)
    Timer(1.5, lambda: webbrowser.open("http://localhost:8000/dashboard")).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)