# ==============================================================================
# FILE 1: mlpipeline3.py
# BESTDOC v11.0 — Multi-Model Classical ML Sentiment & Root-Cause Engine
# - Left Green Box (Tickets): Ranked by Most Solved Tickets & Highest SLA %
# - Right Red Box (Tickets): Ranked by Highest Unresolved Open Ticket Backlog
# - Pareto Problem Severity Scaling + Real-Date Period Filtering Across All 4 Modules
# ==============================================================================
import os
import re
import sys
import ast
import json
import math
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, date
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import joblib  # type: ignore
import nltk  # type: ignore

try:
    from fastapi import HTTPException  # type: ignore
except ImportError:
    class HTTPException(Exception):
        def __init__(self, status_code: int = 400, detail: str = "Error"):
            self.status_code = status_code
            self.detail = detail
            super().__init__(f"HTTP {status_code}: {detail}")

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.pipeline import FeatureUnion, Pipeline  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.naive_bayes import ComplementNB  # type: ignore
from sklearn.svm import LinearSVC  # type: ignore
from sklearn.calibration import CalibratedClassifierCV  # type: ignore
from sklearn.ensemble import VotingClassifier  # type: ignore

for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

if "__file__" in globals():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    CURRENT_DIR = r"C:\Users\91910\OneDrive\Desktop\mrico"


# ==============================================================================
# CUSTOM DOMAIN EXTRACTOR (REQUIRED FOR UNPICKLING JUPYTER NOTEBOOK .PKL FILES)
# ==============================================================================
class HospitalDomainFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        pos_w = {
            "promoter", "excellent", "good", "great", "polite", "caring", "clean", "spotless",
            "resolved", "solved", "helpful", "smooth", "quick", "painless", "recommend",
            "confident", "supportive", "gentle", "well", "timely", "satisfied", "attentive",
            "courteous", "emoji_happy_satisfied", "emoji_love_excellent", "emoji_thumbs_up_good",
            "emoji_grateful_thank_you", "emoji_promoter_love", "emoji_promoter_happy",
            "emoji_promoter_good", "emoji_very_satisfied", "emoji_satisfied_good", "emoji_thumbs_up"
        }
        neg_w = {
            "detractor", "waited", "slow", "delay", "delayed", "rude", "dirty", "worst",
            "terrible", "poor", "unacceptable", "ignored", "misplaced", "late", "crowded",
            "stains", "leaking", "disappointed", "dissatisfied", "argued", "rushed",
            "confusing", "escalated", "unresolved", "complaint", "emoji_angry_furious_worst",
            "emoji_crying_sad_pain", "emoji_thumbs_down_poor", "emoji_detractor_angry",
            "emoji_detractor_sad", "emoji_detractor_poor", "emoji_dissatisfied_angry",
            "emoji_sad_poor", "emoji_thumbs_down"
        }
        neu_w = {
            "passive", "okay", "average", "routine", "standard", "normal", "moderate",
            "acceptable", "fair", "decent", "satisfactory", "brief", "neither",
            "emoji_neutral_average", "emoji_passive_neutral", "emoji_neutral_okay"
        }
        contrast_w = {"but", "however", "though", "although", "except"}

        for text in X:
            tokens = str(text).lower().split()
            p_c = sum(1 for t in tokens if t in pos_w)
            n_c = sum(1 for t in tokens if t in neg_w)
            u_c = sum(1 for t in tokens if t in neu_w)
            has_c = 1 if any(t in contrast_w for t in tokens) else 0
            tot = max(1, p_c + n_c + u_c)
            feats.append([p_c / tot, n_c / tot, u_c / tot, p_c, n_c, u_c, has_c])
        return np.array(feats, dtype=np.float64)


setattr(sys.modules.get("__main__", sys.modules[__name__]), "HospitalDomainFeatureExtractor", HospitalDomainFeatureExtractor)

EMOJI_MAP = {
    "😍": " emoji_love_excellent ", "😊": " emoji_happy_satisfied ", "🙂": " emoji_pleasant_good ",
    "👍": " emoji_thumbs_up_good ", "🙏": " emoji_grateful_thank_you ", "😐": " emoji_neutral_average ",
    "😕": " emoji_confused_unsure ", "😢": " emoji_crying_sad_pain ", "😭": " emoji_sobbing_very_sad ",
    "😡": " emoji_angry_furious_worst ", "🤬": " emoji_rude_unacceptable ", "👎": " emoji_thumbs_down_poor ",
    ":)": " emoji_happy_good ", ":(": " emoji_sad_dissatisfied "
}

SLANG_ABBR_MAP = {
    r"\bgr8\b": "great", r"\bgud\b": "good", r"\bvry\b": "very", r"\bpls\b": "please",
    r"\bthx\b": "thanks", r"\bhrs\b": "hours", r"\bhr\b": "hour", r"\bmins\b": "minutes",
    r"\bopd\b": "outpatient department", r"\bappt\b": "appointment", r"\bdoc\b": "doctor",
    r"\bdr\b": "doctor", r"\brx\b": "prescription", r"\blab\b": "laboratory",
    r"\bdissatisifed\b": "dissatisfied", r"\brecption\b": "reception"
}

SENTIMENT_CONFIG = {
    "NEGATIVE_MAX_S5": 2.6,
    "NEUTRAL_MAX_S5": 3.7,
    "NPS_PROMOTER_MIN": 9,
    "NPS_PASSIVE_MIN": 7,
    "MIXED_NEGATIVE_PROB_THRESHOLD": 0.55,
    "NEGATIVE_KEYWORDS": [
        "dissatisfied", "very dissatisfied", "poor", "very poor", "bad",
        "worst", "rude", "delay", "slow", "detractor", "unhappy", "terrible"
    ],
    "NEUTRAL_KEYWORDS": [
        "neutral", "average", "passive", "fair", "ok", "okay", "satisfactory"
    ],
    "POSITIVE_KEYWORDS": [
        "excellent", "very good", "good", "satisfied", "very satisfied",
        "promoter", "happy", "thank", "great"
    ],
}

LATEST_MODULE_CACHE: Dict[str, Any] = {
    "activeModule": "NONE",
    "NPS": None,
    "CSAT": None,
    "FEEDBACK": None,
    "TICKETS": None,
}

PHONE_RE = re.compile(r"(?:\+91[\-\s]?)?[6-9]\d{9}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
DUMMY_RE = re.compile(r"(far far away|word mountains|vokalia|consonantia|lorem|ipsum|blind texts|test comment|dummy|asdf)", re.I)
CONTRAST_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s+\b(?=(?:but|however|although|though|except|yet|whereas)\b)", flags=re.I)
MONTH_COL_RE = re.compile(r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\'\-\_]*\d{2,4}$", re.I)

QUESTION_CATEGORY_MAP = {
    "scheduling": "Appointment Scheduling & Clinic Access",
    "appointment": "Appointment Scheduling & Clinic Access",
    "opd": "Appointment Scheduling & Clinic Access",
    "medical": "Physician Consultation & Clinical Care",
    "doctor": "Physician Consultation & Clinical Care",
    "nursing": "Nursing Care & Ward Responsiveness",
    "nurse": "Nursing Care & Ward Responsiveness",
    "laboratory": "Diagnostic Lab & Report Turnaround",
    "lab": "Diagnostic Lab & Report Turnaround",
    "pharmacy": "Pharmacy Dispensing & Medication Counseling",
    "housekeeping": "Hospital Cleanliness & Facility Hygiene",
    "clean": "Hospital Cleanliness & Facility Hygiene",
    "billing": "Billing Transparency & Discharge Speed",
    "overall": "Overall Hospital Service Experience",
}


def clean_and_expand_hospital_text(text: str) -> str:
    if not text:
        return ""
    s = EMAIL_RE.sub(" ", str(text))
    s = PHONE_RE.sub(" ", s)
    for emo, rep in EMOJI_MAP.items():
        s = s.replace(emo, rep)
    s = s.lower()
    for pat, rep in SLANG_ABBR_MAP.items():
        s = re.sub(pat, rep, s)
    s = re.sub(r"[^a-z0-9_\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_mixed_clauses_nltk(text: str) -> List[str]:
    if not text:
        return []
    raw_s = str(text).strip()
    try:
        sents = nltk.tokenize.sent_tokenize(raw_s)
    except Exception:
        sents = [raw_s]
    clauses: List[str] = []
    for sent in sents:
        parts = [clean_and_expand_hospital_text(p) for p in CONTRAST_SPLIT_RE.split(sent) if p and len(p.strip()) >= 3]
        clauses.extend([p for p in parts if p])
    return clauses if clauses else [clean_and_expand_hospital_text(raw_s)]


def infer_clinical_category(dept_name: str, question_text: str) -> str:
    combined = f"{dept_name} {question_text}".lower()
    for kw, label in QUESTION_CATEGORY_MAP.items():
        if kw in combined:
            return label
    return f"{dept_name} Clinical Service Quality"


def clear_all_module_cache() -> dict:
    global LATEST_MODULE_CACHE
    LATEST_MODULE_CACHE.clear()
    LATEST_MODULE_CACHE.update({
        "activeModule": "NONE",
        "NPS": None,
        "CSAT": None,
        "FEEDBACK": None,
        "TICKETS": None,
    })
    return LATEST_MODULE_CACHE


def clean_nans_for_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): clean_nans_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_nans_for_json(x) for x in obj]
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return 0.0 if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def clear_and_set_module_cache(mod: str, payload: dict) -> dict:
    clear_all_module_cache()
    payload["dashboardLink"] = "http://localhost:8000/dashboard"
    payload["activeMLModel"] = ML_ENGINE.get_winner_name(mod)
    clean_payload = clean_nans_for_json(payload)
    LATEST_MODULE_CACHE["activeModule"] = mod
    LATEST_MODULE_CACHE[mod] = clean_payload
    return clean_payload


def is_valid_patient_comment(text: Any) -> bool:
    if text is None:
        return False
    try:
        if pd.isna(text):
            return False
    except Exception:
        pass
    s = str(text).strip()
    if len(s) < 4:
        return False
    if s.lower() in {
        "nan", "none", "null", "nothing", "nil", "na", "n/a", "no", "ok", "yes",
        "-", ".", "--", "true", "false", "excellent", "good", "average", "poor",
        "dissatisfied", "satisfied", "neutral", "very good", "very poor"
    }:
        return False
    if re.fullmatch(r"[\d\.\-\%\s\/]+", s) or DUMMY_RE.search(s):
        return False
    return True


def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        s = str(val).replace("%", "").strip()
        v = float(pd.to_numeric(s, errors="coerce"))
        return default if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    except Exception:
        return default


def safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    if isinstance(val, dict):
        return str(
            val.get("name") or val.get("department") or val.get("assignee")
            or val.get("doctorName") or val.get("text") or val.get("label") or default
        ).strip()
    if isinstance(val, list):
        pts = [safe_str(x, "") for x in val if safe_str(x, "")]
        return ", ".join(pts) if pts else default
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    s = str(val).strip()
    return default if s.lower() in ["nan", "none", "null", ""] else s


def parse_any_hospital_timestamp(val: Any) -> Optional[pd.Timestamp]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass

    if isinstance(val, (pd.Timestamp, datetime, date)):
        ts = pd.Timestamp(val)
        return ts if ts.year >= 2020 else None

    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null", "0", "-", ""}:
        return None

    if re.fullmatch(r"\d+(\.\d+)?", s):
        num = float(s)
        if num > 1e11:
            ts = pd.to_datetime(num, unit="ms", errors="coerce")
            return ts if (ts is not None and not pd.isna(ts) and ts.year >= 2020) else None
        elif num > 1e8:
            ts = pd.to_datetime(num, unit="s", errors="coerce")
            return ts if (ts is not None and not pd.isna(ts) and ts.year >= 2020) else None
        return None

    ts = pd.to_datetime(s, errors="coerce")
    if ts is not None and not pd.isna(ts) and ts.year >= 2020:
        return ts
    return None


def extract_record_timestamp(rec: dict) -> Optional[pd.Timestamp]:
    if not isinstance(rec, dict):
        return None
    candidate_keys = [
        "respondedAt", "visitDate", "createDateTime", "createDate",
        "createdAt", "updatedDate", "updatedAt", "date", "Date",
        "Visit Date", "Survey Date", "Created Date", "Month", "Period"
    ]
    for k in candidate_keys:
        if k in rec and rec[k] is not None:
            ts = parse_any_hospital_timestamp(rec[k])
            if ts is not None:
                return ts
    resp = rec.get("respondent")
    if isinstance(resp, dict):
        for k in candidate_keys:
            if k in resp and resp[k] is not None:
                ts = parse_any_hospital_timestamp(resp[k])
                if ts is not None:
                    return ts
    return None


def parse_any_bestdoc_input(raw_text: str) -> dict:
    clean = str(raw_text or "").strip().lstrip("\ufeff")
    if not clean:
        raise HTTPException(status_code=400, detail="Empty JSON input.")
    if clean.startswith('"'):
        clean = "{" + clean + "}"
    for fn in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.sub(r",\s*([\]}])", r"\1", t)),
        lambda t: ast.literal_eval(t),
    ]:
        try:
            d = fn(clean)
            if isinstance(d, list):
                return {"records": d, "_matched_key": None}
            if isinstance(d, dict):
                for w in ["data", "result", "response", "payload"]:
                    if isinstance(d.get(w), list):
                        return {"records": d[w], "_matched_key": w}
                    if isinstance(d.get(w), dict):
                        d = d[w]
                        break
                for k in [
                    "respondentWithTicketDetails", "tickets", "ticketList", "idData",
                    "records", "surveyList", "feedbackList", "npsList", "csatList"
                ]:
                    if isinstance(d.get(k), list):
                        return {"records": d[k], "_matched_key": k}
                for k, v in d.items():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return {"records": v, "_matched_key": k}
                return {"records": [d], "_matched_key": None}
        except Exception:
            continue
    raise HTTPException(status_code=400, detail="Invalid JSON format. Please check your JSON syntax.")


# ==============================================================================
# MULTI-MODEL LOADER (LOADS ALL 4 SEPARATE .PKL FILES ONCE)
# ==============================================================================
class MultiModelHospitalMLEngine:
    def __init__(self) -> None:
        self.models: Dict[str, Dict[str, Any]] = {}
        self.fallback_pipe: Any = None  # Disabled: no seven-example fallback.
        self._is_loaded: bool = False
        self.load_all_trained_models()

    def load_all_trained_models(self, force_reload: bool = False) -> None:
        if self._is_loaded and not force_reload:
            return

        pkl_map = {
            "FEEDBACK": ["feedback_best_model.pkl"],
            "TICKETS": ["ticket_best_model.pkl"],
            "NPS": ["nps_best_model.pkl"],
            "CSAT": ["csat_best_model.pkl"],
        }

        loaded_models: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []

        for mod, candidates in pkl_map.items():
            model_loaded = False

            for fname in candidates:
                fpath = os.path.join(CURRENT_DIR, fname)
                if not os.path.isfile(fpath):
                    continue

                try:
                    bundle = joblib.load(fpath)

                    if isinstance(bundle, dict):
                        pipe = bundle.get("polarity_pipeline")
                        encoder = bundle.get("polarity_encoder")
                        winner = str(bundle.get("polarity_winner", "Saved Classical ML Model"))
                        feature_cols = list(bundle.get("feature_cols", []))
                    else:
                        pipe = bundle
                        encoder = None
                        winner = "Saved Model"
                        feature_cols = []

                    if pipe is None or not callable(getattr(pipe, "predict_proba", None)):
                        raise ValueError(
                            "Model must be a pipeline/estimator with predict_proba()."
                        )

                    loaded_models[mod] = {
                        "pipeline": pipe,
                        "encoder": encoder,
                        "winner": f"{winner} ({fname})",
                        "feature_cols": feature_cols,
                    }
                    print(f"✅ [{mod}] Loaded trained model: {fname} -> Winner: {winner}")
                    model_loaded = True
                    break

                except Exception as exc:
                    errors.append(f"{mod}: could not load {fname}: {exc}")
                    print(f"⚠️ Could not load {fname} for {mod}: {exc}")

            if not model_loaded:
                errors.append(
                    f"{mod}: no usable model found; expected {', '.join(candidates)}"
                )

        if errors:
            details = "\n".join(f"  - {item}" for item in errors)
            raise RuntimeError(
                "Required trained models could not be loaded. "
                "Check that all four model files exist beside mlpipeline3.py "
                "and that they load successfully:\n"
                f"{details}"
            )

        self.models = loaded_models
        self._is_loaded = True

    def load_all_saved_pkl_models(self, force_reload: bool = False) -> None:
        self.load_all_trained_models(force_reload=force_reload)

    def get_winner_name(self, mod: str = "FEEDBACK") -> str:
        info = self.models.get(mod.upper())
        if info is None:
            raise RuntimeError(f"No trained model configured for module {mod!r}.")
        return info["winner"]

    def _predict_clauses(
        self,
        clauses: List[str],
        mod: str,
        dept_name: str,
        question_type: str,
    ) -> Tuple[np.ndarray, List[str]]:
        info = self.models.get(mod.upper())
        if info is None:
            raise RuntimeError(f"No trained model configured for module {mod!r}.")

        pipe = info["pipeline"]
        encoder = info["encoder"]
        feature_cols = info["feature_cols"]

        if feature_cols:
            df_in = pd.DataFrame({
                "Text": clauses,
                "department": [dept_name] * len(clauses),
                "question_type": [question_type] * len(clauses),
                "module_type": [mod] * len(clauses),
                "no_of_tickets": [1 if mod == "TICKETS" else 0] * len(clauses),
                "dept_seq_no": [1] * len(clauses),
                "word_count": [len(c.split()) for c in clauses],
                "char_length": [len(c) for c in clauses],
            })

            missing_cols = [col for col in feature_cols if col not in df_in.columns]
            if missing_cols:
                raise RuntimeError(
                    f"{mod} model requires unavailable input columns: {missing_cols}"
                )

            model_input = df_in[feature_cols]
        else:
            model_input = clauses

        try:
            probas = np.asarray(pipe.predict_proba(model_input), dtype=float)
        except Exception as exc:
            raise RuntimeError(
                f"Prediction failed for {mod} using {info['winner']}: {exc}"
            ) from exc

        if probas.ndim != 2 or probas.shape[0] != len(clauses):
            raise RuntimeError(
                f"{mod} model returned an invalid probability array: {probas.shape}"
            )

        if encoder is not None and hasattr(encoder, "classes_"):
            raw_classes = [str(c).lower() for c in encoder.classes_]
        elif hasattr(pipe, "classes_"):
            raw_classes = [str(c).lower() for c in pipe.classes_]
        else:
            raise RuntimeError(
                f"Cannot determine class labels for {mod} model {info['winner']}."
            )

        if len(raw_classes) != probas.shape[1]:
            raise RuntimeError(
                f"{mod} model has {len(raw_classes)} labels but returned "
                f"{probas.shape[1]} probability columns."
            )

        norm_classes = [
            "negative" if "neg" in label
            else "neutral" if "neu" in label
            else "positive"
            for label in raw_classes
        ]

        if not {"negative", "neutral", "positive"}.issubset(set(norm_classes)):
            raise RuntimeError(
                f"{mod} model must provide Negative, Neutral, and Positive classes; "
                f"found {raw_classes}."
            )

        return probas, norm_classes

    def predict_sentiment(
        self,
        text: str,
        s5_score: Any = None,
        answer_label: str = "",
        dept_name: str = "General",
        question_text: str = "",
        question_type: str = "CSAT",
        module_type: str = "FEEDBACK",
    ) -> Dict[str, Any]:
        has_free_text = is_valid_patient_comment(text)
        ans_clean = str(answer_label or "").strip().lower()
        neg_max = SENTIMENT_CONFIG["NEGATIVE_MAX_S5"]
        neu_max = SENTIMENT_CONFIG["NEUTRAL_MAX_S5"]

        if has_free_text:
            clauses = split_mixed_clauses_nltk(text)
        else:
            if s5_score is not None and not pd.isna(s5_score):
                score = float(s5_score)
                if score <= neg_max or any(
                    keyword in ans_clean
                    for keyword in SENTIMENT_CONFIG["NEGATIVE_KEYWORDS"]
                ):
                    synth = (
                        f"{dept_name} {question_text} {answer_label} "
                        "dissatisfied poor service delay slow"
                    )
                elif score <= neu_max or any(
                    keyword in ans_clean
                    for keyword in SENTIMENT_CONFIG["NEUTRAL_KEYWORDS"]
                ):
                    synth = (
                        f"{dept_name} {question_text} {answer_label} "
                        "neutral average routine service okay"
                    )
                else:
                    synth = (
                        f"{dept_name} {question_text} {answer_label} "
                        "excellent good polite satisfied care"
                    )
            else:
                synth = f"{dept_name} {question_text} {answer_label}"

            clauses = [clean_and_expand_hospital_text(synth)]

        if not clauses:
            clauses = [clean_and_expand_hospital_text(text)]

        target_module = str(module_type or "FEEDBACK").upper().strip()
        if target_module not in self.models:
            raise RuntimeError(f"No trained model configured for module {target_module!r}.")

        probas, classes = self._predict_clauses(
            clauses=clauses,
            mod=target_module,
            dept_name=dept_name,
            question_type=question_type,
        )

        neg_idx = classes.index("negative")
        pos_idx = classes.index("positive")

        worst_clause_idx = int(np.argmax(probas[:, neg_idx]))
        if (
            len(clauses) > 1
            and probas[worst_clause_idx, neg_idx]
            >= SENTIMENT_CONFIG["MIXED_NEGATIVE_PROB_THRESHOLD"]
        ):
            agg_p = probas[worst_clause_idx]
        else:
            agg_p = np.mean(probas, axis=0)

        if not has_free_text and s5_score is not None and not pd.isna(s5_score):
            score = float(s5_score)
            polarity = (
                "negative" if score <= neg_max
                else "neutral" if score <= neu_max
                else "positive"
            )
            confidence = round(float(np.max(agg_p)), 3)
        else:
            best_idx = int(np.argmax(agg_p))
            polarity = classes[best_idx]
            confidence = round(float(agg_p[best_idx]), 3)

            cleaned_text = clean_and_expand_hospital_text(text)
            if (
                any(word in cleaned_text.split()
                    for word in SENTIMENT_CONFIG["NEGATIVE_KEYWORDS"])
                and "not" not in cleaned_text.split()
            ):
                polarity = "negative"

        ui_sentiment = (
            "Negative (Dissatisfied)"
            if polarity == "negative"
            else "Positive (Satisfied)"
            if polarity == "positive"
            else "Neutral (Passive)"
        )

        return {
            "polarity": polarity,
            "ui_sentiment": ui_sentiment,
            "confidence": confidence,
            "signed_ml_score": round(
                float(agg_p[pos_idx] - agg_p[neg_idx]), 3
            ),
            "model_used": self.get_winner_name(target_module),
        }


MultiModuleHospitalMLEngine = MultiModelHospitalMLEngine
ML_ENGINE = MultiModelHospitalMLEngine()

# ==============================================================================
# SCORE NORMALIZATION & ROOT-CAUSE HELPERS
# ==============================================================================
def normalize_bestdoc_score_to_5(raw_score: Any, question_type: str = "", question_text: str = "", answer_label: str = "", is_negative_flag: bool = False) -> float:
    qt_up = str(question_type or "").upper().strip()
    q_low = str(question_text or "").lower()
    ans_low = str(answer_label or "").lower().strip()
    is_nps_scale = (qt_up == "NPS") or ("0 to 10" in q_low) or ("0-10" in q_low)

    val = pd.to_numeric(str(raw_score).replace("%", "").strip() if raw_score is not None else np.nan, errors="coerce")
    if pd.isna(val):
        if is_negative_flag or ans_low in SENTIMENT_CONFIG["NEGATIVE_KEYWORDS"]:
            return 1.8
        if ans_low in SENTIMENT_CONFIG["NEUTRAL_KEYWORDS"]:
            return 3.3
        if ans_low in SENTIMENT_CONFIG["POSITIVE_KEYWORDS"]:
            return 4.8
        return 3.5

    v = float(val)
    if v > 10.0:
        s5 = v / 20.0
    elif is_nps_scale or v > 5.0:
        if v < SENTIMENT_CONFIG["NPS_PASSIVE_MIN"]:
            s5 = max(1.0, (v / 6.0) * 2.4)
        elif v < SENTIMENT_CONFIG["NPS_PROMOTER_MIN"]:
            s5 = 3.1 + ((v - 7.0) * 0.4)
        else:
            s5 = 4.5 + ((v - 9.0) * 0.5)
    else:
        s5 = v

    if is_negative_flag and ans_low in SENTIMENT_CONFIG["NEGATIVE_KEYWORDS"]:
        s5 = min(s5, 2.0)
    elif ans_low in SENTIMENT_CONFIG["NEUTRAL_KEYWORDS"] and s5 > SENTIMENT_CONFIG["NEUTRAL_MAX_S5"]:
        s5 = 3.3
    elif ans_low in ["excellent", "very satisfied"] and s5 < 4.5:
        s5 = 4.9

    return round(max(1.0, min(5.0, s5)), 2)


def extract_hospital_themes(texts: list, top_n: int = 3) -> str:
    valid_docs = [
        str(t).strip() for t in texts[:2500]
        if is_valid_patient_comment(t) and not str(t).startswith("Ticket ") and not str(t).startswith("Patient ")
    ]
    if len(valid_docs) < 2:
        return ""
    try:
        vec = TfidfVectorizer(ngram_range=(2, 3), stop_words="english", max_features=25)
        mat = vec.fit_transform(valid_docs)
        scores = np.asarray(mat.sum(axis=0)).flatten()
        vocab = np.array(vec.get_feature_names_out())
        top_idx = scores.argsort()[::-1][:top_n]
        phrases = [vocab[i] for i in top_idx if scores[i] > 0]
        return ", ".join(phrases)
    except Exception:
        return ""


def extract_bestdoc_department_and_doctor(record: dict) -> tuple:
    if not isinstance(record, dict):
        return "General Hospital", ""
    dept_raw = (
        record.get("dept") or record.get("Department") or record.get("department")
        or record.get("Touchpoint") or record.get("touchpoint") or record.get("deptName")
    )
    dept_name = safe_str(dept_raw, "")
    doctor_name, consultant_dept = "", ""

    resp = record.get("respondent")
    if isinstance(resp, dict):
        doctor_name = safe_str(resp.get("attendingDoctor"), "")
        add_param = resp.get("additionalParam")
        if isinstance(add_param, dict):
            consultants = add_param.get("consultant")
            if isinstance(consultants, list) and consultants and isinstance(consultants[0], dict):
                consultant_dept = safe_str(consultants[0].get("department"), "")
                if not doctor_name:
                    doctor_name = safe_str(consultants[0].get("doctorName"), "")

    if (not dept_name or dept_name.lower() in ["overall rating", "general", "overall", "hospital"]) and consultant_dept:
        return consultant_dept.title(), doctor_name
    return (dept_name if dept_name else "General Hospital"), doctor_name


def infer_why_low_rating(dept_name: str, question_text: str = "", comment_text: str = "", extra_keywords: str = "", neg_count: int = -1, mean_rating: float = 0.0) -> str:
    if neg_count == 0 and mean_rating >= 4.0:
        return f"Zero negative responses recorded in {dept_name} (Mean {mean_rating}/5.0) — No service bottlenecks detected."
    q_clean = str(question_text or "").strip()
    c_clean = str(comment_text or "").strip()
    cat_label = infer_clinical_category(dept_name, q_clean)
    if is_valid_patient_comment(c_clean) and not c_clean.startswith("Ticket ") and not c_clean.startswith("Patient "):
        kw_tag = f" | Key Themes: [{extra_keywords}]" if extra_keywords else ""
        return f'Patient Verbatim: "{c_clean[:100]}" ({cat_label}){kw_tag}'
    if extra_keywords:
        return f"Recurring complaint themes in {dept_name}: [{extra_keywords}] ({cat_label})"
    if q_clean and "how would you rate the overall services" not in q_clean.lower():
        return f"Dissatisfaction recorded on '{q_clean[:85]}' ({cat_label})"
    return f"Low overall service rating in {dept_name} ({cat_label}) — requires Guest Relations follow-up."


def infer_why_high_rating(dept_name: str, question_text: str = "", comment_text: str = "") -> str:
    q_clean = str(question_text or "").strip()
    c_clean = str(comment_text or "").strip()
    cat_label = infer_clinical_category(dept_name, q_clean)
    if is_valid_patient_comment(c_clean) and not c_clean.startswith("Ticket ") and not c_clean.startswith("Patient "):
        return f'Patient Praise: "{c_clean[:100]}" ({cat_label})'
    if q_clean and "how would you rate the overall services" not in q_clean.lower():
        return f"High patient satisfaction on '{q_clean[:85]}' ({cat_label})"
    return f"High patient satisfaction on 'Overall Hospital Services (0-10)' ({cat_label})"


# ==============================================================================
# REAL-DATE & OVERALL HOSPITAL MULTI-DEPARTMENT TREND BUILDER
# ==============================================================================
def build_real_period_trend_series(
    df: pd.DataFrame,
    mode: str,
    overall_pos_pct: float,
    overall_neu_pct: float,
    overall_neg_pct: float,
    all_depts_list: List[str],
) -> List[dict]:
    depts_clean = [d for d in all_depts_list if d and str(d).lower() not in {"nan", "none"}]
    depts_summary = ", ".join(depts_clean[:6]) + (f" (+{len(depts_clean)-6} more)" if len(depts_clean) > 6 else "")
    if not depts_summary:
        depts_summary = "All Hospital Touchpoints"

    total_dataset_rows = max(1, len(df))

    # Case A: Wide Month Columns in Excel (e.g. Apr' 26, May' 26)
    month_cols = [c for c in df.columns if MONTH_COL_RE.match(str(c).strip())]
    if month_cols:
        wide_trend = []
        for mc in month_cols:
            vals = pd.to_numeric(df[mc].astype(str).str.replace("%", "").str.strip(), errors="coerce").dropna()
            if len(vals) == 0:
                continue
            mean_v = float(vals.mean())
            if mode == "NPS":
                pos_p = int(round(min(100.0, max(0.0, (vals >= 9).mean() * 100.0 if vals.max() <= 10 else mean_v))))
                neg_p = int(round(min(100.0 - pos_p, max(0.0, (vals < 7).mean() * 100.0 if vals.max() <= 10 else max(0.0, 100.0 - mean_v)))))
                neu_p = int(max(0, 100 - pos_p - neg_p))
                net_val = int(pos_p - neg_p)
            else:
                s100 = mean_v * 20.0 if mean_v <= 5.0 else (mean_v * 10.0 if mean_v <= 10.0 else mean_v)
                pos_p = int(round(min(100.0, max(0.0, s100))))
                rem = max(0, 100 - pos_p)
                neu_p = int(round(rem * 0.35))
                neg_p = int(max(0, rem - neu_p))
                net_val = int(pos_p)

            pl = str(mc).strip()
            wide_trend.append({
                "period": pl,
                "positivePct": pos_p,
                "neutralPct": neu_p,
                "negativePct": neg_p,
                "netScore": net_val,
                "circleLabel": f"{net_val}{'%' if mode != 'NPS' else ''}",
                "totalRecords": int(len(vals)),
                "departmentsIncluded": depts_summary,
                "hoverReason": f"{pl} ({depts_summary}) • Pos: {pos_p}% | Neu: {neu_p}% | Neg: {neg_p}%",
            })
        if wide_trend:
            return wide_trend

    # Case B: Real Timestamps (_dt) in JSON / Excel
    if "_dt" in df.columns:
        df_valid_dt = df[df["_dt"].notna()].copy()
        if len(df_valid_dt) > 0:
            df_valid_dt["_dt"] = pd.to_datetime(df_valid_dt["_dt"], errors="coerce")
            df_valid_dt = df_valid_dt[df_valid_dt["_dt"].notna()].sort_values("_dt")

            if len(df_valid_dt) > 0:
                df_valid_dt["_ym"] = df_valid_dt["_dt"].dt.to_period("M")
                if df_valid_dt["_ym"].nunique() >= 2:
                    df_valid_dt["_period_key"] = df_valid_dt["_dt"].dt.strftime("%Y-%m")
                    df_valid_dt["_period_label"] = df_valid_dt["_dt"].dt.strftime("%b' %y")
                elif df_valid_dt["_dt"].dt.date.nunique() >= 2:
                    df_valid_dt["_period_key"] = df_valid_dt["_dt"].dt.strftime("%Y-%m-%d")
                    df_valid_dt["_period_label"] = df_valid_dt["_dt"].dt.strftime("%d %b' %y")
                else:
                    df_valid_dt["_period_key"] = df_valid_dt["_dt"].dt.strftime("%Y-%m")
                    df_valid_dt["_period_label"] = df_valid_dt["_dt"].dt.strftime("%b' %y")

                real_trend = []
                total_dated_rows = max(1, len(df_valid_dt))

                for _, grp in df_valid_dt.groupby("_period_key", sort=True):
                    if len(grp) == 0:
                        continue
                    pl = str(grp["_period_label"].iloc[0])
                    tot = len(grp)

                    grp_depts = (
                        [str(d) for d in grp["_dept"].dropna().unique() if str(d).strip()]
                        if "_dept" in grp.columns else depts_clean
                    )
                    grp_dept_str = ", ".join(grp_depts[:5]) + (f" (+{len(grp_depts)-5} more)" if len(grp_depts) > 5 else "")

                    # SPECIAL LOGIC FOR TICKETS: More Tickets = Higher Problem Share %
                    if mode == "TICKETS":
                        problem_share_pct = int(round((tot / total_dated_rows) * 100.0))
                        solved_c = int(grp["is_solved"].sum()) if "is_solved" in grp.columns else 0
                        open_c = tot - solved_c
                        solved_pct = int(round((solved_c / max(1, tot)) * 100.0))
                        open_pct = int(round((open_c / max(1, tot)) * 100.0))

                        top_m_dept = grp["_dept"].value_counts().index[0] if "_dept" in grp.columns and len(grp) > 0 else "Hospital"
                        top_m_cnt = int(grp["_dept"].value_counts().iloc[0]) if "_dept" in grp.columns and len(grp) > 0 else tot

                        real_trend.append({
                            "period": pl,
                            "positivePct": solved_pct,          # Green Track: SLA Solved %
                            "neutralPct": open_pct,             # Amber Track: Open Unresolved Backlog %
                            "negativePct": problem_share_pct,   # Red Track: Share of Total Hospital Ticket Surge %
                            "netScore": tot,                    # Raw Ticket Volume
                            "circleLabel": f"{tot} Tkt ({problem_share_pct}%)",
                            "totalRecords": int(tot),
                            "departmentsIncluded": grp_dept_str or depts_summary,
                            "hoverReason": (
                                f"{pl} • {tot} Escalated Tickets ({problem_share_pct}% of Total Hospital Problem Load) • "
                                f"Top Bottleneck: {top_m_dept} ({top_m_cnt} Tkt) • Open: {open_c} ({open_pct}%) | Solved: {solved_c} ({solved_pct}%)"
                            ),
                        })
                    else:
                        if "_sent" in grp.columns:
                            pos_c = int(grp["_sent"].astype(str).str.contains("Positive").sum())
                            neu_c = int(grp["_sent"].astype(str).str.contains("Neutral").sum())
                            neg_c = int(grp["_sent"].astype(str).str.contains("Negative").sum())
                        elif "_sc" in grp.columns:
                            pos_c = int((grp["_sc"] >= SENTIMENT_CONFIG["NPS_PROMOTER_MIN"]).sum())
                            neu_c = int(((grp["_sc"] >= SENTIMENT_CONFIG["NPS_PASSIVE_MIN"]) & (grp["_sc"] < SENTIMENT_CONFIG["NPS_PROMOTER_MIN"])).sum())
                            neg_c = int((grp["_sc"] < SENTIMENT_CONFIG["NPS_PASSIVE_MIN"]).sum())
                        else:
                            pos_c, neu_c, neg_c = tot, 0, 0

                        pos_p = int(round((pos_c / max(1, tot)) * 100.0))
                        neg_p = int(round((neg_c / max(1, tot)) * 100.0))
                        neu_p = int(max(0, 100 - pos_p - neg_p))
                        net_val = int(pos_p - neg_p) if mode == "NPS" else int(pos_p)

                        real_trend.append({
                            "period": pl,
                            "positivePct": pos_p,
                            "neutralPct": neu_p,
                            "negativePct": neg_p,
                            "netScore": net_val,
                            "circleLabel": f"{net_val}{'%' if mode != 'NPS' else ''}",
                            "totalRecords": int(tot),
                            "departmentsIncluded": grp_dept_str or depts_summary,
                            "hoverReason": (
                                f"{pl} • Overall Hospital ({tot} Records across {len(grp_depts)} Depts: {grp_dept_str}) • "
                                f"Pos: {pos_p}% ({pos_c}) | Neu: {neu_p}% ({neu_c}) | Neg: {neg_p}% ({neg_c})"
                            ),
                        })
                if real_trend:
                    return real_trend

    # Case C: Single Batch Fallback
    if mode == "TICKETS":
        return [{
            "period": "Uploaded Report",
            "positivePct": 0,
            "neutralPct": 100,
            "negativePct": 100,
            "netScore": total_dataset_rows,
            "circleLabel": f"{total_dataset_rows} Tkt (100%)",
            "totalRecords": total_dataset_rows,
            "departmentsIncluded": depts_summary,
            "hoverReason": f"Uploaded Report • {total_dataset_rows} Total Escalated Tickets across {depts_summary}",
        }]

    pos_p = int(round(safe_float(overall_pos_pct, 85.0)))
    neg_p = int(round(safe_float(overall_neg_pct, 10.0)))
    neu_p = int(max(0, 100 - pos_p - neg_p))
    net_val = int(pos_p - neg_p) if mode == "NPS" else int(pos_p)

    return [{
        "period": "Uploaded Report",
        "positivePct": pos_p,
        "neutralPct": neu_p,
        "negativePct": neg_p,
        "netScore": net_val,
        "circleLabel": f"{net_val}{'%' if mode != 'NPS' else ''}",
        "totalRecords": total_dataset_rows,
        "departmentsIncluded": depts_summary,
        "hoverReason": f"Uploaded Report ({total_dataset_rows} Records across {depts_summary}) • Pos: {pos_p}% | Neu: {neu_p}% | Neg: {neg_p}%",
    }]


def extract_top_5_comments(df: pd.DataFrame, module_type: str = "FEEDBACK") -> Tuple[List[dict], List[dict]]:
    cols = list(df.columns)
    dept_c = next((c for c in cols if str(c).lower().strip() in ["_dept", "department", "dept", "touchpoint", "unit", "service", "category"]), None)
    q_c = next((c for c in cols if str(c).lower().strip() in ["question", "subquestion", "item", "parameter", "survey question", "_prob"]), None)
    score_c = next((c for c in cols if re.search(r"_s5|nps|csat|score|rating|recommend|average|answervalue", str(c), re.I)), None)
    explicit_comm_cols = [
        c for c in cols
        if any(k in str(c).lower().strip() for k in ["comment", "comments", "feedback", "remark", "remarks", "review", "verbatim", "suggestion", "statement of problem", "description", "_comm"])
    ]

    pos_candidates, neg_candidates = [], []
    if explicit_comm_cols:
        for _, r in df.head(5000).iterrows():
            row_dept = safe_str(r.get(dept_c), "Hospital General") if dept_c else "Hospital General"
            row_q = safe_str(r.get(q_c), "") if q_c else ""
            raw_sc = pd.to_numeric(r.get(score_c), errors="coerce") if score_c else np.nan
            s5 = np.nan
            if not pd.isna(raw_sc):
                val = float(raw_sc)
                s5 = max(1.0, min(5.0, val / 20.0 if val > 10.0 else (val / 2.0 if val > 5.0 else val)))

            for cc in explicit_comm_cols:
                if cc == dept_c or cc == q_c:
                    continue
                txt = safe_str(r.get(cc), "")
                if not is_valid_patient_comment(txt):
                    continue
                c_dept = str(cc).split(":", 1)[0].strip() if ":" in str(cc) else row_dept
                ml_pred = ML_ENGINE.predict_sentiment(text=txt, s5_score=s5, dept_name=c_dept, question_text=row_q, module_type=module_type)
                pol, signed_sc = ml_pred["polarity"], ml_pred["signed_ml_score"]
                has_sc = not pd.isna(s5)

                if pol == "negative" or (has_sc and s5 <= SENTIMENT_CONFIG["NEGATIVE_MAX_S5"]):
                    disp_rating = round(s5 if has_sc else max(1.2, 2.5 + signed_sc), 1)
                    neg_candidates.append({
                        "_rank": signed_sc - (disp_rating / 5.0),
                        "department": c_dept,
                        "rating": f"🚨 {min(2.6, max(1.0, disp_rating))}/5.0",
                        "comment": txt[:175],
                        "reason": infer_why_low_rating(c_dept, row_q, txt),
                    })
                elif pol == "positive" or (has_sc and s5 > SENTIMENT_CONFIG["NEUTRAL_MAX_S5"]):
                    disp_rating = round(s5 if has_sc else min(4.9, 4.0 + signed_sc), 1)
                    pos_candidates.append({
                        "_rank": signed_sc + (disp_rating / 5.0),
                        "department": c_dept,
                        "rating": f"⭐ {max(3.8, min(5.0, disp_rating))}/5.0",
                        "comment": txt[:175],
                        "reason": infer_why_high_rating(c_dept, row_q, txt),
                    })

    pos_candidates.sort(key=lambda x: x["_rank"], reverse=True)
    neg_candidates.sort(key=lambda x: x["_rank"])

    unique_pos, seen_pos = [], set()
    for p in pos_candidates:
        key = p["comment"].lower().strip()
        if key not in seen_pos:
            seen_pos.add(key)
            unique_pos.append({k: v for k, v in p.items() if k != "_rank"})

    unique_neg, seen_neg = [], set()
    for n in neg_candidates:
        key = n["comment"].lower().strip()
        if key not in seen_neg:
            seen_neg.add(key)
            unique_neg.append({k: v for k, v in n.items() if k != "_rank"})

    if (len(unique_pos) < 5 or len(unique_neg) < 5) and "_structured_summary" in df.columns:
        for _, r in df.head(500).iterrows():
            sent_val = str(r.get("_sent", ""))
            is_solved_row = int(r.get("is_solved", 0)) == 1 if "is_solved" in df.columns else False
            d_val = safe_str(r.get("_dept") or r.get("Department"), "Hospital")
            q_val = safe_str(r.get("Question") or r.get("_prob"), "Overall Hospital Services (0-10)")
            badge_val = safe_str(r.get("_badge"), "5.0/5.0 (Excellent)")
            summary_txt = safe_str(r.get("_structured_summary"), "")
            if not summary_txt:
                continue
            key = summary_txt.lower().strip()

            if ("Positive" in sent_val or (module_type == "TICKETS" and is_solved_row)) and len(unique_pos) < 5 and key not in seen_pos:
                seen_pos.add(key)
                unique_pos.append({
                    "department": d_val,
                    "rating": f"✅ Solved • {badge_val}" if (module_type == "TICKETS" and is_solved_row) else f"⭐ {badge_val}",
                    "comment": summary_txt[:180],
                    "reason": f"Ticket resolved within SLA in '{d_val}' by Guest Relations." if (module_type == "TICKETS" and is_solved_row) else infer_why_high_rating(d_val, q_val),
                })
            elif ("Negative" in sent_val or "Neutral" in sent_val) and not (module_type == "TICKETS" and is_solved_row) and len(unique_neg) < 5 and key not in seen_neg:
                seen_neg.add(key)
                unique_neg.append({
                    "department": d_val,
                    "rating": f"🚨 {badge_val}",
                    "comment": summary_txt[:180],
                    "reason": infer_why_low_rating(d_val, q_val),
                })

    if not unique_pos:
        unique_pos.append({
            "department": "No Positive / Resolved Records Yet",
            "rating": "⏳ 0 Solved",
            "comment": "All records in this batch are open complaints or negative/neutral escalations awaiting resolution.",
            "reason": "Complete Guest Relations callback to generate positive recovery ratings.",
        })
    if not unique_neg:
        unique_neg.append({
            "department": "All Touchpoints (Zero Negative Feedback)",
            "rating": "🎉 100% Positive",
            "comment": "Zero negative or dissatisfied responses were found in this dataset — every rated parameter achieved high satisfaction.",
            "reason": "No service bottlenecks detected; maintain current clinical and operational standards.",
        })

    return unique_pos[:5], unique_neg[:5]


# ==============================================================================
# MODULE 1: NPS PIPELINE (USES nps_best_model.pkl + REAL DATES + ALL DEPTS)
# ==============================================================================
def execute_nps_pipeline(raw_records: list) -> dict:
    if not raw_records:
        raise HTTPException(status_code=400, detail="No NPS records found.")
    df = pd.DataFrame(raw_records)
    cols = list(df.columns)
    dept_c = next((c for c in cols if any(k in str(c).lower().strip() for k in ["department", "dept", "touchpoint", "unit", "service", "branch", "speciality", "specialty"])), None)
    if not dept_c and "source_sheet" in cols:
        dept_c = "source_sheet"
    comm_c = next((c for c in cols if any(k in str(c).lower().strip() for k in ["comment", "comments", "feedback", "remarks"])), None)
    score_col = next((c for c in cols if re.search(r"nps|score|rating|recommend|answervalue", str(c), re.I)), cols[0])

    df["_sc"] = pd.to_numeric(df[score_col].astype(str).str.replace("%", ""), errors="coerce").fillna(8.5)
    df["_sc"] = np.where(df["_sc"] > 10.0, np.clip((df["_sc"] + 100.0) / 20.0, 0.0, 10.0), df["_sc"])
    df["_dept"] = [extract_bestdoc_department_and_doctor(r)[0] if isinstance(r, dict) and ("dept" in r or "respondent" in r) else safe_str(r.get(dept_c) if dept_c and isinstance(r, dict) else "Overall Hospital", "Overall Hospital") for r in raw_records]
    df["_dt"] = [extract_record_timestamp(r) for r in raw_records]

    prom_min = SENTIMENT_CONFIG["NPS_PROMOTER_MIN"]
    pass_min = SENTIMENT_CONFIG["NPS_PASSIVE_MIN"]
    df["_sent"] = np.where(df["_sc"] >= prom_min, "Positive (Satisfied)", np.where(df["_sc"] >= pass_min, "Neutral (Passive)", "Negative (Dissatisfied)"))

    nps_rows, subq_rows = [], []
    for dn, grp in df.groupby("_dept", sort=False):
        tot = max(1, len(grp))
        pr_c = int((grp["_sc"] >= prom_min).sum())
        pa_c = int(((grp["_sc"] >= pass_min) & (grp["_sc"] < prom_min)).sum())
        de_c = int((grp["_sc"] < pass_min).sum())
        pr = round(float(pr_c / tot * 100.0), 1)
        pa = round(float(pa_c / tot * 100.0), 1)
        de = round(float(de_c / tot * 100.0), 1)
        sc = round(pr - de, 1)
        mean_10 = safe_float(grp["_sc"].mean(), 8.5)

        comms = [v for v in (grp[comm_c].dropna().astype(str).tolist() if comm_c else []) if is_valid_patient_comment(v)]
        kw_str = extract_hospital_themes(comms)
        sample_c = comms[0] if comms else ""
        w_low = infer_why_low_rating(str(dn), "Overall Hospital Services (0-10)", sample_c, kw_str, neg_count=de_c, mean_rating=mean_10 / 2.0)
        w_high = infer_why_high_rating(str(dn), "Overall Hospital Services (0-10)", sample_c)

        nps_rows.append({
            "touchpoint": str(dn), "department": str(dn), "question": "Overall Hospital Services (0-10)",
            "npsAvgRating": mean_10, "meanRating": round(mean_10 / 2.0, 2),
            "positiveCount": pr_c, "neutralCount": pa_c, "negativeCount": de_c,
            "promoterPct": pr, "passivePct": pa, "detractorPct": de,
            "positivePct": pr, "neutralPct": pa, "lowRatingPct": de, "npsScore": sc,
            "whyLowReason": w_low, "whyBadReason": w_low, "whyHighReason": w_high, "whyGoodReason": w_high,
            "recommendedAction": f"Conduct detractor callback recovery in '{dn}'." if de_c > 0 else f"Maintain high satisfaction in '{dn}'.",
            "hoverReason": f"{dn} • Promoters: {pr}% ({pr_c}) | Passives: {pa}% ({pa_c}) | Detractors: {de}% ({de_c}) • NPS: {sc}",
        })
        subq_rows.append({
            "shortLabel": f"{str(dn)[:18]}: NPS (0-10)",
            "negativePct": de, "neutralPct": pa, "positivePct": pr,
            "hoverReason": w_low if de > 0 else w_high,
        })

    worst_l = sorted(nps_rows, key=lambda x: (-x["negativeCount"], safe_float(x["npsScore"]), -safe_float(x["detractorPct"])))
    best_l = sorted(nps_rows, key=lambda x: (safe_float(x["npsScore"]), x["positiveCount"], safe_float(x["promoterPct"])), reverse=True)
    w0, b0 = worst_l[0], best_l[0]

    tot_all = max(1, len(df))
    overall_prom = round(float((df["_sc"] >= prom_min).sum() / tot_all * 100.0), 1)
    overall_pass = round(float(((df["_sc"] >= pass_min) & (df["_sc"] < prom_min)).sum() / tot_all * 100.0), 1)
    overall_det = round(float((df["_sc"] < pass_min).sum() / tot_all * 100.0), 1)
    overall_nps = round(overall_prom - overall_det, 1)
    overall_rating = safe_float(df["_sc"].mean(), 8.5)

    all_depts = [x["department"] for x in nps_rows]
    period_trend = build_real_period_trend_series(df, "NPS", overall_prom, overall_pass, overall_det, all_depts)
    top_pos_c, top_neg_c = extract_top_5_comments(df, "NPS")

    return clear_and_set_module_cache("NPS", {
        "status": "SUCCESS", "module": "NPS", "activeMode": "NPS",
        "autoDetectedBadge": "Auto-Detected: NPS ONLY",
        "trendTitle": "OVERALL HOSPITAL NET PROMOTER SCORE (NPS) TREND — ALL DEPARTMENTS",
        "kpis": {
            "npsScore": overall_nps, "npsAvgRating": overall_rating,
            "promoterPct": overall_prom, "passivePct": overall_pass, "detractorPct": overall_det,
            "totalRespondents": len(df),
            "lowestArea": "None (0% Detractors)" if overall_det == 0 else f"{w0['department']} (NPS: {w0['npsScore']})",
        },
        "strengthHeadline": f"TOP NPS ADVOCACY ACROSS {len(nps_rows)} HOSPITAL UNITS (LEAD: '{b0['department'].upper()}')",
        "strengthExplanation": f"Overall Hospital Promoters: {overall_prom}% across {len(df):,} records (Model: {ML_ENGINE.get_winner_name('NPS')}).",
        "topStrengthCards": [{
            "department": x["department"],
            "metricBadge": f"🌟 {x['positiveCount']} Promoters ({x['promoterPct']}% | {x['npsAvgRating']}/10)",
            "sentimentBreakdown": f"😍 {x['positiveCount']} Promoters • 😐 {x['neutralCount']} Passives • 😡 {x['negativeCount']} Detractors",
            "whyGoodReason": x["whyHighReason"],
            "bestPractice": f"Share '{x['department']}' service protocol across other hospital touchpoints."
        } for x in best_l[:4]],
        "sentimentHeadline": "ZERO DETRACTORS DETECTED — ALL UNITS ACHIEVED POSITIVE NPS" if overall_det == 0 else f"WHY '{w0['department'].upper()}' RECEIVED LOW NPS ({w0['detractorPct']}% DETRACTORS)",
        "sentimentExplanation": "No detractors were recorded in this batch; cards below confirm 0% detractor risk." if overall_det == 0 else "Touchpoints with the highest detractor concentration requiring Guest Relations recovery.",
        "rootCauseCards": [{
            "department": x["department"],
            "metricBadge": f"✅ 0 Detractors ({x['npsAvgRating']}/10)" if x["negativeCount"] == 0 else f"🚨 {x['negativeCount']} Detractors ({x['detractorPct']}% | NPS {x['npsScore']})",
            "sentimentBreakdown": f"😡 {x['negativeCount']} Detractors • 😐 {x['neutralCount']} Passives • 😍 {x['positiveCount']} Promoters",
            "whyBadReason": x["whyLowReason"],
            "recommendedAction": x["recommendedAction"]
        } for x in worst_l[:4]],
        "touchpointRatings": nps_rows[:12],
        "subQuestionDrivers": subq_rows[:10],
        "periodTrend": period_trend,
        "topPositiveComments": top_pos_c,
        "topNegativeComments": top_neg_c,
    })


# ==============================================================================
# MODULE 2: CSAT PIPELINE (USES csat_best_model.pkl + REAL DATES + ALL DEPTS)
# ==============================================================================
def execute_csat_pipeline(raw_records: list) -> dict:
    if not raw_records:
        raise HTTPException(status_code=400, detail="No CSAT records found.")
    df = pd.DataFrame(raw_records)
    cols = list(df.columns)
    dept_c = next((c for c in cols if any(k in str(c).lower().strip() for k in ["department", "dept", "touchpoint", "service", "unit", "category"])), None)
    if not dept_c and "source_sheet" in cols:
        dept_c = "source_sheet"
    q_c = next((c for c in cols if any(k in str(c).lower().strip() for k in ["question", "subquestion", "item", "parameter"])), None)
    avg_c = next((c for c in cols if re.search(r"csat|average|avg|mean|score|rating|answervalue", str(c), re.I)), None)
    comm_c = next((c for c in cols if any(k in str(c).lower().strip() for k in ["comment", "comments", "feedback", "remarks"])), None)

    row_records = []
    for idx, r in df.iterrows():
        rec_dict = raw_records[idx] if idx < len(raw_records) and isinstance(raw_records[idx], dict) else {}
        dept = safe_str(r.get(dept_c) if dept_c else (r.get("Department") or r.get("department")), "General Hospital")
        qt = safe_str(r.get(q_c) if q_c else (r.get("Question") or r.get("question")), f"Service Quality in {dept}")
        comm_txt = safe_str(r.get(comm_c), "") if comm_c else ""
        avg2 = safe_float(r.get(avg_c) if avg_c else 4.5, 4.5)
        avg2 = round(avg2 / 20.0 if avg2 > 10.0 else (avg2 / 2.0 if avg2 > 5.0 else avg2), 2)

        ml_pred = ML_ENGINE.predict_sentiment(text=comm_txt, s5_score=avg2, dept_name=dept, question_text=qt, question_type="CSAT", module_type="CSAT")
        sent = ml_pred["ui_sentiment"]

        row_records.append({
            "_dept": dept,
            "Question": qt,
            "_s5": avg2,
            "_sent": sent,
            "Comment": comm_txt,
            "_dt": extract_record_timestamp(rec_dict),
        })

    df_proc = pd.DataFrame(row_records)
    for mc in cols:
        if MONTH_COL_RE.match(str(mc).strip()):
            df_proc[mc] = df[mc].values

    tp_rows, subq_rows = [], []
    for dn, grp in df_proc.groupby("_dept", sort=False):
        tot = max(1, len(grp))
        pos_c = int((grp["_sent"].str.contains("Positive")).sum())
        neu_c = int((grp["_sent"].str.contains("Neutral")).sum())
        neg_c = int((grp["_sent"].str.contains("Negative")).sum())
        mean_v = safe_float(grp["_s5"].mean(), 4.5)

        pos_pct = round(pos_c / tot * 100.0, 1)
        neu_pct = round(neu_c / tot * 100.0, 1)
        neg_pct = round(neg_c / tot * 100.0, 1)

        q_sample = safe_str(grp["Question"].iloc[0], f"Service Quality in {dn}")
        comm_sample = safe_str(grp["Comment"].iloc[0], "")
        w_low = infer_why_low_rating(str(dn), q_sample, comm_sample, neg_count=neg_c, mean_rating=mean_v)
        w_high = infer_why_high_rating(str(dn), q_sample, comm_sample)

        tp_rows.append({
            "touchpoint": str(dn), "department": str(dn), "question": q_sample, "meanRating": mean_v,
            "positiveCount": pos_c, "neutralCount": neu_c, "negativeCount": neg_c,
            "positivePct": pos_pct, "neutralPct": neu_pct, "lowRatingPct": neg_pct,
            "whyLowReason": w_low, "whyBadReason": w_low, "whyHighReason": w_high, "whyGoodReason": w_high,
            "recommendedAction": f"Audit '{q_sample[:45]}' workflow in '{dn}'." if neg_c > 0 else f"Maintain high satisfaction in '{dn}'.",
            "hoverReason": f"{dn} • Pos: {pos_pct}% ({pos_c}) | Neu: {neu_pct}% ({neu_c}) | Neg: {neg_pct}% ({neg_c}) • Mean: {mean_v}/5.0",
        })

    for _, r in df_proc.head(12).iterrows():
        sc2 = round((float(r["_s5"]) / 5.0) * 100.0, 1)
        rem = round(max(0.0, 100.0 - sc2), 1)
        neu2 = round(rem * 0.35, 1)
        dis2 = round(max(0.0, rem - neu2), 1)
        subq_rows.append({
            "shortLabel": f"{str(r['_dept'])[:14]}: {str(r['Question'])[:24]}",
            "negativePct": dis2, "neutralPct": neu2, "positivePct": sc2,
            "hoverReason": f"{r['_dept']} — {r['Question']} ({r['_s5']}/5.0)",
        })

    d_worst = sorted(tp_rows, key=lambda x: (x["negativeCount"], x["lowRatingPct"], -x["meanRating"]), reverse=True)
    d_best = sorted(tp_rows, key=lambda x: (x["positivePct"], x["positiveCount"], x["meanRating"]), reverse=True)
    w0, b0 = d_worst[0], d_best[0]

    tot_all = max(1, len(df_proc))
    avg_sat = round(float((df_proc["_sent"].str.contains("Positive")).sum() / tot_all * 100.0), 1)
    avg_neu = round(float((df_proc["_sent"].str.contains("Neutral")).sum() / tot_all * 100.0), 1)
    avg_dis = round(max(0.0, 100.0 - avg_sat - avg_neu), 1)
    avg_score = safe_float(df_proc["_s5"].mean(), 4.5)

    all_depts = [d["department"] for d in tp_rows]
    period_trend = build_real_period_trend_series(df_proc, "CSAT", avg_sat, avg_neu, avg_dis, all_depts)
    top_pos_c, top_neg_c = extract_top_5_comments(df_proc, "CSAT")

    return clear_and_set_module_cache("CSAT", {
        "status": "SUCCESS", "module": "CSAT", "activeMode": "CSAT",
        "autoDetectedBadge": "Auto-Detected: CSAT ONLY",
        "trendTitle": "OVERALL HOSPITAL CSAT SATISFACTION RATE TREND (%) — ALL DEPARTMENTS",
        "kpis": {
            "currentCsatAvg": avg_score, "currentSatisfiedPct": avg_sat,
            "currentNeutralPct": avg_neu, "currentDissatisfiedPct": avg_dis,
            "lowestDepartment": "None (0% Dissatisfied)" if avg_dis == 0 else f"{w0['department']} ({w0['lowRatingPct']}% Neg)",
        },
        "strengthHeadline": f"TOP CSAT SATISFACTION ACROSS {len(tp_rows)} HOSPITAL UNITS (LEAD: '{b0['department'].upper()}')",
        "strengthExplanation": f"Overall Hospital CSAT: {avg_sat}% across {len(tp_rows)} departments (Model: {ML_ENGINE.get_winner_name('CSAT')}).",
        "topStrengthCards": [{
            "department": d["department"],
            "metricBadge": f"🌟 {d['positiveCount']} Pos ({d['positivePct']}% | {d['meanRating']}/5)",
            "sentimentBreakdown": f"😊 {d['positiveCount']} Pos • 😐 {d['neutralCount']} Neu • 😡 {d['negativeCount']} Neg",
            "whyGoodReason": d["whyHighReason"],
            "bestPractice": f"Share '{d['department']}' service protocol across other hospital touchpoints."
        } for d in d_best[:4]],
        "sentimentHeadline": "ZERO BOTTLENECKS DETECTED — ALL TOUCHPOINTS ACHIEVED POSITIVE CSAT" if avg_dis == 0 else f"WHY '{w0['department'].upper()}' RECEIVED LOW CSAT ({w0['lowRatingPct']}% UNSATISFIED)",
        "sentimentExplanation": "No negative CSAT responses were recorded in this batch; cards below confirm 0% dissatisfaction." if avg_dis == 0 else "Departments with the lowest CSAT scores requiring workflow optimization.",
        "rootCauseCards": [{
            "department": d["department"],
            "metricBadge": f"✅ 0 Neg ({d['meanRating']}/5)" if d["negativeCount"] == 0 else f"🚨 {d['negativeCount']} Neg ({d['lowRatingPct']}% | {d['meanRating']}/5)",
            "sentimentBreakdown": f"😡 {d['negativeCount']} Neg • 😐 {d['neutralCount']} Neu • 😊 {d['positiveCount']} Pos",
            "whyBadReason": d["whyLowReason"],
            "recommendedAction": d["recommendedAction"]
        } for d in d_worst[:4]],
        "touchpointRatings": tp_rows[:12],
        "subQuestionDrivers": subq_rows[:10],
        "periodTrend": period_trend,
        "topPositiveComments": top_pos_c,
        "topNegativeComments": top_neg_c,
    })


# ==============================================================================
# MODULE 3: FEEDBACK PIPELINE (USES feedback_best_model.pkl + REAL DATES + ALL DEPTS)
# ==============================================================================
def execute_feedback_survey_pipeline(raw_records: list) -> dict:
    if not raw_records:
        raise HTTPException(status_code=400, detail="No feedback records found.")

    has_nested = any(isinstance(r, dict) and isinstance(r.get("respondedAnswerList"), list) for r in raw_records[:25])
    if has_nested:
        flat_rows, respondent_sentiments = [], []
        for r in raw_records:
            if not isinstance(r, dict):
                continue
            resp_name = safe_str(r.get("name"), "Patient")
            resp_mrn = safe_str(r.get("mrn") or r.get("uhid"), "")
            resp_dt = extract_record_timestamp(r)
            ans_list = r.get("respondedAnswerList") or []
            resp_scores, resp_has_neg = [], False

            for a in ans_list:
                if not isinstance(a, dict):
                    continue
                dept_raw = (
                    a.get("dept") or a.get("department") or a.get("deptName")
                    or a.get("touchpoint") or r.get("dept") or r.get("department")
                )
                dept_name = safe_str(dept_raw, "General")
                q_txt = safe_str(a.get("question") or a.get("questionName"), f"{dept_name} Service Feedback")
                q_type = safe_str(a.get("questionType"), "CSAT").upper()
                ans_label = safe_str(a.get("answer"), "")
                ans_val = a.get("answerValue") if a.get("answerValue") is not None else a.get("score")
                comm_txt = safe_str(a.get("comment") or r.get("comments"), "")
                is_neg_flag = bool(a.get("isNegative", False))
                ans_dt = extract_record_timestamp(a) or resp_dt

                s5 = normalize_bestdoc_score_to_5(ans_val, q_type, q_txt, ans_label, is_neg_flag)
                ml_pred = ML_ENGINE.predict_sentiment(
                    text=comm_txt, s5_score=s5, answer_label=ans_label,
                    dept_name=dept_name, question_text=q_txt, question_type=q_type, module_type="FEEDBACK"
                )
                sent = ml_pred["ui_sentiment"]

                resp_scores.append(s5)
                if "Negative" in sent:
                    resp_has_neg = True

                disp_badge = f"{ans_val}/10 (NPS)" if q_type == "NPS" else f"{s5}/5.0 ({ans_label or 'Excellent'})"
                mrn_tag = f" [MRN: {resp_mrn}]" if resp_mrn else ""
                cat_label = infer_clinical_category(dept_name, q_txt)
                short_q = "Overall Hospital Services (0-10)" if q_type == "NPS" else q_txt[:75]
                structured_txt = (
                    comm_txt if is_valid_patient_comment(comm_txt)
                    else f"Patient {resp_name}{mrn_tag} rated '{ans_label or ans_val}' ({disp_badge}) on {cat_label}: \"{short_q}\""
                )

                flat_rows.append({
                    "Department": dept_name, "Question": short_q, "QuestionType": q_type,
                    "Answer": ans_label, "Rating": s5, "_s5": s5, "_sent": sent, "_dept": dept_name,
                    "Comment": comm_txt, "_dt": ans_dt,
                    "_badge": disp_badge, "_structured_summary": structured_txt,
                })

            if resp_scores:
                avg_r_s5 = float(np.mean(resp_scores))
                respondent_sentiments.append(
                    "Negative (Dissatisfied)" if (resp_has_neg or avg_r_s5 <= SENTIMENT_CONFIG["NEGATIVE_MAX_S5"])
                    else ("Neutral (Passive)" if avg_r_s5 <= SENTIMENT_CONFIG["NEUTRAL_MAX_S5"] else "Positive (Satisfied)")
                )

        df = pd.DataFrame(flat_rows) if flat_rows else pd.DataFrame(raw_records)
        total_forms = len(respondent_sentiments) if respondent_sentiments else len(df)
        overall_sent = pd.Series(respondent_sentiments) if respondent_sentiments else df["_sent"]
    else:
        df = pd.DataFrame(raw_records)
        df["_dt"] = [extract_record_timestamp(r) for r in raw_records]
        total_forms = len(df)
        overall_sent = None

    cols = list(df.columns)
    dept_c = next((c for c in cols if str(c).lower().strip() in ["_dept", "department", "dept", "touchpoint", "unit", "service", "category"]), None)
    q_c = next((c for c in cols if str(c).lower().strip() in ["question", "statement of problem", "issue", "subquestion", "parameter"]), None)
    score_c = next((c for c in cols if str(c).lower().strip() in ["_s5", "rating", "score", "csat average", "csat", "department score", "overall rating", "answervalue"]), None)
    comm_c = next((c for c in cols if str(c).lower().strip() in ["comment", "comments", "feedback", "remarks", "statement of problem"]), None)

    if "_sent" not in df.columns or "_s5" not in df.columns:
        txt_s = df[comm_c].fillna("").astype(str) if comm_c else pd.Series("", index=df.index)
        sc_s = pd.to_numeric(df[score_c], errors="coerce") if score_c else pd.Series(4.5, index=df.index)
        dept_s = df[dept_c].fillna("General").astype(str) if dept_c else pd.Series("General", index=df.index)
        sents = []
        for idx, txt in enumerate(txt_s.tolist()):
            val = sc_s.iloc[idx]
            s5_v = val / 20.0 if val > 10.0 else (val / 2.0 if val > 5.0 else val)
            sents.append(ML_ENGINE.predict_sentiment(text=txt, s5_score=s5_v, dept_name=dept_s.iloc[idx], module_type="FEEDBACK")["ui_sentiment"])
        df["_sent"] = pd.Series(sents, index=df.index)
        s5 = np.where(sc_s > 10.0, sc_s / 20.0, np.where(sc_s > 5.0, sc_s / 2.0, sc_s))
        df["_s5"] = pd.Series(s5, index=df.index).fillna(4.5)

    df["_dept"] = df[dept_c].apply(lambda x: safe_str(x, "General Hospital")) if dept_c else "General Hospital"

    DEPT_PRIORITY = {"overall rating": 1, "opd": 2, "medical": 3, "nursing": 4, "laboratory": 5, "pharmacy": 6, "housekeeping": 7}

    tp_summary, subq_drivers = [], []
    for dn, grp in df.groupby("_dept", sort=False):
        if not dn or str(dn).lower() in ["nan", "none"]:
            continue
        neg_c = int((grp["_sent"].str.contains("Negative")).sum())
        neu_c = int((grp["_sent"].str.contains("Neutral")).sum())
        pos_c = int((grp["_sent"].str.contains("Positive")).sum())
        tot_c = max(1, len(grp))
        mean_v = safe_float(grp["_s5"].mean(), 5.0)
        comms_list = [v for v in (grp[comm_c].dropna().astype(str).tolist() if comm_c else []) if is_valid_patient_comment(v)]
        kw_str = extract_hospital_themes(comms_list)
        q_sample = safe_str(grp[q_c].iloc[0], f"{dn} Service Experience") if q_c and len(grp) > 0 else f"{dn} Service Experience"
        w_low = infer_why_low_rating(str(dn), q_sample, comms_list[0] if comms_list else "", kw_str, neg_count=neg_c, mean_rating=mean_v)
        w_high = infer_why_high_rating(str(dn), q_sample, comms_list[0] if comms_list else "")

        prio = DEPT_PRIORITY.get(str(dn).lower().strip(), 5)

        tp_summary.append({
            "touchpoint": str(dn), "meanRating": mean_v, "_prio": prio,
            "negativeCount": neg_c, "neutralCount": neu_c, "positiveCount": pos_c,
            "lowRatingPct": round(neg_c / tot_c * 100.0, 1),
            "neutralPct": round(neu_c / tot_c * 100.0, 1),
            "positivePct": round(pos_c / tot_c * 100.0, 1),
            "whyBadReason": w_low, "whyLowReason": w_low, "whyGoodReason": w_high, "whyHighReason": w_high,
            "recommendedAction": f"Deploy queue & service recovery in '{dn}'." if neg_c > 0 else f"Maintain high satisfaction in '{dn}'.",
            "hoverReason": f"{dn} • {pos_c} Pos ({round(pos_c/tot_c*100,1)}%), {neu_c} Neu, {neg_c} Neg • Mean: {mean_v}/5.0",
        })
        subq_drivers.append({
            "department": str(dn), "subQuestion": q_sample, "shortLabel": f"{str(dn)}: {q_sample[:24]}",
            "meanRating": mean_v, "negativeCount": neg_c, "neutralCount": neu_c, "positiveCount": pos_c,
            "negativePct": round(neg_c / tot_c * 100.0, 1), "neutralPct": round(neu_c / tot_c * 100.0, 1), "positivePct": round(pos_c / tot_c * 100.0, 1),
            "whyLowReason": w_low, "whyHighReason": w_high,
            "recommendedAction": f"Improve turnaround in '{dn}'." if neg_c > 0 else f"Maintain '{dn}' standard.",
            "hoverReason": f"{dn} ({q_sample[:45]}) • Pos: {round(pos_c/tot_c*100,1)}% | Neu: {round(neu_c/tot_c*100,1)}% | Neg: {round(neg_c/tot_c*100,1)}%",
        })

    if overall_sent is None:
        overall_sent = df["_sent"]

    tp_summary.sort(key=lambda x: (x["negativeCount"], x["lowRatingPct"], -x["meanRating"], -x["_prio"]), reverse=True)
    subq_drivers.sort(key=lambda x: (x["negativeCount"], x["negativePct"], -x["meanRating"]), reverse=True)
    tp_best = sorted(tp_summary, key=lambda x: (x["positivePct"], x["positiveCount"], x["meanRating"], -x["_prio"]), reverse=True)

    pos_tot = int((overall_sent.str.contains("Positive")).sum())
    neu_tot = int((overall_sent.str.contains("Neutral")).sum())
    neg_tot = int((overall_sent.str.contains("Negative")).sum())
    sat_rate = round((pos_tot / max(1, total_forms)) * 100.0, 1)
    neu_rate = round((neu_tot / max(1, total_forms)) * 100.0, 1)
    neg_rate = round((neg_tot / max(1, total_forms)) * 100.0, 1)

    w0, b0 = tp_summary[0], tp_best[0]
    all_depts = [t["touchpoint"] for t in tp_best]
    period_trend = build_real_period_trend_series(df, "FEEDBACK", sat_rate, neu_rate, neg_rate, all_depts)
    top_pos_c, top_neg_c = extract_top_5_comments(df, "FEEDBACK")

    return clear_and_set_module_cache("FEEDBACK", {
        "status": "SUCCESS", "module": "FEEDBACK", "activeMode": "FEEDBACK",
        "autoDetectedBadge": "Auto-Detected: FEEDBACK ONLY",
        "trendTitle": "OVERALL HOSPITAL PATIENT SATISFACTION RATE TREND (%) — ALL DEPARTMENTS",
        "kpis": {
            "totalFeedbackResponses": int(total_forms), "totalTouchpointRatings": int(len(df)),
            "positivePatients": pos_tot, "neutralPatients": neu_tot,
            "criticalRiskPatients": neg_tot, "satisfactionRatePct": sat_rate,
            "lowestTouchpoint": f"None (0 Neg across {len(tp_summary)} Depts)" if neg_tot == 0 else f"{w0['touchpoint']} ({w0['negativeCount']:,} Neg)",
        },
        "strengthHeadline": f"HIGH POSITIVE FEEDBACK ACROSS {len(tp_summary)} DEPARTMENTS ({sat_rate}% OVERALL POSITIVE)",
        "strengthExplanation": f"All {len(tp_summary)} hospital touchpoints evaluated across {total_forms:,} respondents ({len(df):,} ratings • Model: {ML_ENGINE.get_winner_name('FEEDBACK')}).",
        "topStrengthCards": [{
            "department": t["touchpoint"],
            "metricBadge": f"🌟 {t['positiveCount']:,} Pos ({t['positivePct']}% | {t['meanRating']}/5)",
            "sentimentBreakdown": f"😊 {t['positiveCount']:,} Pos • 😐 {t['neutralCount']:,} Neu • 😡 {t['negativeCount']:,} Neg",
            "whyGoodReason": t["whyGoodReason"],
            "bestPractice": f"Share '{t['touchpoint']}' service protocol across other hospital touchpoints."
        } for t in tp_best[:4]],
        "sentimentHeadline": f"ZERO BOTTLENECKS DETECTED — ALL {len(tp_summary)} TOUCHPOINTS ACHIEVED POSITIVE RATINGS" if neg_tot == 0 else f"WHY '{w0['touchpoint'].upper()}' RECEIVED NEGATIVE FEEDBACK ({w0['negativeCount']:,} NEGATIVE)",
        "sentimentExplanation": "No negative feedback was recorded in this batch; cards below confirm 0% dissatisfaction across monitored units." if neg_tot == 0 else "Touchpoints with the highest negative sentiment requiring Guest Relations intervention.",
        "rootCauseCards": [{
            "department": t["touchpoint"],
            "metricBadge": f"✅ 0 Neg ({t['meanRating']}/5)" if t["negativeCount"] == 0 else f"🚨 {t['negativeCount']:,} Neg ({t['lowRatingPct']}% | {t['meanRating']}/5)",
            "sentimentBreakdown": f"😡 {t['negativeCount']:,} Neg • 😐 {t['neutralCount']} Neu • 😊 {t['positiveCount']} Pos",
            "whyBadReason": t["whyBadReason"],
            "recommendedAction": t["recommendedAction"]
        } for t in tp_summary[:4]],
        "touchpointRatings": tp_best[:12],
        "subQuestionDrivers": subq_drivers[:10],
        "periodTrend": period_trend,
        "topPositiveComments": top_pos_c,
        "topNegativeComments": top_neg_c,
    })


# ==============================================================================
# MODULE 4: UPGRADED TICKET MANAGEMENT PIPELINE (SEPARATE SOLVED VS OPEN RANKING)
# ==============================================================================
def execute_ticket_triage_pipeline(raw_tickets: list) -> dict:
    if not raw_tickets:
        raise HTTPException(status_code=400, detail="No ticket records found.")

    norm_rows = []
    for r in raw_tickets:
        if not isinstance(r, dict):
            continue
        dept_name, doc_name = extract_bestdoc_department_and_doctor(r)
        status_val = safe_str(r.get("status") or r.get("assignmentStatus") or r.get("Status"), "OPEN").upper()
        tkt_code = safe_str(r.get("ticketCode") or r.get("Ticket ID") or r.get("id"), "TKT")
        tkt_dt = extract_record_timestamp(r)

        resp = r.get("respondent") if isinstance(r.get("respondent"), dict) else {}
        pat_name = safe_str(resp.get("name"), "Patient")
        pat_uhid = safe_str(resp.get("uhid") or resp.get("mrn"), "")
        assignees = safe_str(r.get("ticketAssignee"), "Guest Relations")
        q_txt = safe_str(r.get("question") or r.get("Statement of Problem"), "Overall Hospital Services (0-10)")
        ans_label = safe_str(r.get("answer"), "Dissatisfied")
        raw_val = r.get("answerValue") if r.get("answerValue") is not None else (r.get("npsScore") or r.get("Score") or 5)
        comm_txt = safe_str(r.get("comment") or r.get("comments") or r.get("correctionAction"), "")

        q_type = "NPS" if (r.get("npsScore") is not None or "0 to 10" in q_txt.lower()) else "CSAT"
        s5 = normalize_bestdoc_score_to_5(raw_val, q_type, q_txt, ans_label, True)
        ml_pred = ML_ENGINE.predict_sentiment(
            text=comm_txt, s5_score=s5, answer_label=ans_label,
            dept_name=dept_name, question_text=q_txt, question_type=q_type, module_type="TICKETS"
        )
        sent = ml_pred["ui_sentiment"]

        doc_tag = f" ({doc_name})" if doc_name else ""
        uhid_tag = f" [UHID: {pat_uhid}]" if pat_uhid else ""
        score_tag = f"{raw_val}/10 (NPS)" if q_type == "NPS" else f"{s5}/5.0 ({ans_label})"
        structured_summary = (
            comm_txt if is_valid_patient_comment(comm_txt)
            else f"Ticket {tkt_code} ({status_val}) • Patient {pat_name}{uhid_tag} rated '{ans_label}' ({score_tag}) in {dept_name}{doc_tag} — Assigned to: {assignees}"
        )

        norm_rows.append({
            "_dept": dept_name, "_doc": doc_name, "_status": status_val,
            "_prob": f"Escalated {q_type} Rating: '{ans_label}' ({score_tag}) in {dept_name}{doc_tag}",
            "_comm": comm_txt, "_s5": s5, "_sent": sent, "_dt": tkt_dt,
            "_badge": f"{tkt_code} • {ans_label} ({raw_val}/10)", "_structured_summary": structured_summary,
        })

    df_proc = pd.DataFrame(norm_rows)
    df_proc["is_solved"] = df_proc["_status"].isin(["RESOLVED", "CLOSED", "SOLVED", "COMPLETED", "DONE"]).astype(int)
    df_proc["is_open"] = (df_proc["is_solved"] == 0).astype(int)

    tot_tkts = max(1, int(len(df_proc)))
    dept_counts = df_proc["_dept"].value_counts()
    max_dept_tickets = max(1, int(dept_counts.max())) if len(dept_counts) > 0 else 1

    dept_sum, subq_rows = [], []
    for dn, grp in df_proc.groupby("_dept", sort=False):
        tot, op, sl = int(len(grp)), int(grp["is_open"].sum()), int(grp["is_solved"].sum())
        neg = int((grp["_sent"].str.contains("Negative")).sum())
        neu = int((grp["_sent"].str.contains("Neutral")).sum())
        mean_v = safe_float(grp["_s5"].mean(), 2.0)

        valid_probs = [p for p in grp["_prob"].dropna().astype(str).tolist()[:300] if is_valid_patient_comment(p)]
        valid_comms = [c for c in grp["_comm"].dropna().astype(str).tolist()[:300] if is_valid_patient_comment(c)]
        kw_str = extract_hospital_themes(valid_comms)
        rc = infer_why_low_rating(str(dn), valid_probs[0] if valid_probs else "", valid_comms[0] if valid_comms else "", kw_str, neg_count=neg + op, mean_rating=mean_v)

        hospital_problem_share_pct = round((tot / tot_tkts) * 100.0, 1)
        pareto_problem_index_pct = round((op / max_dept_tickets) * 100.0, 1)
        solved_index_pct = round((sl / max_dept_tickets) * 100.0, 1)
        sla_resolution_pct = round((sl / max(1, tot)) * 100.0, 1)

        dept_sum.append({
            "touchpoint": str(dn), "department": str(dn), "totalTickets": tot,
            "openTickets": op, "solvedTickets": sl, "negativeCount": op, "neutralCount": neu, "positiveCount": sl,
            "lowRatingPct": hospital_problem_share_pct,
            "paretoProblemPct": pareto_problem_index_pct,
            "neutralPct": round(neu / max(1, tot) * 100.0, 1),
            "positivePct": sla_resolution_pct,
            "meanRating": mean_v,
            "topRootCause": rc, "whyLowReason": rc, "whyBadReason": rc,
            "whyGoodReason": f"{sl} of {tot} tickets resolved ({sla_resolution_pct}% SLA resolution) in '{dn}' — Assigned to Guest Relations.",
            "recommendedAction": f"Immediate SLA callback: resolve {op} open tickets in '{dn}' ({hospital_problem_share_pct}% of hospital problem load).",
            "hoverReason": f"{dn} • {tot} Tickets ({hospital_problem_share_pct}% of Hospital Load) • {op} Open, {sl} Solved ({sla_resolution_pct}% SLA)",
        })
        subq_rows.append({
            "shortLabel": f"{str(dn)[:18]} ({tot} Tkt)",
            "negativePct": pareto_problem_index_pct,
            "neutralPct": 0.0,
            "positivePct": solved_index_pct,
            "totalTickets": tot,
            "openTickets": op,
            "hospitalSharePct": hospital_problem_share_pct,
            "hoverReason": f"{dn}: {tot} Escalated Tickets ({op} Open, {sl} Solved | {hospital_problem_share_pct}% of total hospital complaints)",
        })

    # 1. Sort Right Red Box (Bottlenecks) by MOST OPEN UNRESOLVED tickets first (e.g., Nursing 55 Open -> OPD 28 Open)
    dept_worst_open = sorted(
        dept_sum,
        key=lambda x: (x["openTickets"], x["paretoProblemPct"], x["totalTickets"]),
        reverse=True,
    )

    # 2. Sort Left Green Box (SLA Recovery) by MOST SOLVED tickets & SLA % first (e.g., Medical 74 Solved -> OPD 58 Solved)
    dept_best_solved = sorted(
        dept_sum,
        key=lambda x: (x["solvedTickets"], x["positivePct"], -x["openTickets"]),
        reverse=True,
    )

    # Sort charts by total ticket volume
    dept_sum.sort(key=lambda x: (x["totalTickets"], x["openTickets"]), reverse=True)
    subq_rows.sort(key=lambda x: (x["totalTickets"], x["openTickets"]), reverse=True)

    top_open_d = dept_worst_open[0]
    top_solved_d = dept_best_solved[0]

    tot_open, tot_solved = int(df_proc["is_open"].sum()), int(df_proc["is_solved"].sum())
    tot_neg = int((df_proc["_sent"].str.contains("Negative")).sum())
    tot_neu = int((df_proc["_sent"].str.contains("Neutral")).sum())
    tot_pos = int((df_proc["_sent"].str.contains("Positive")).sum())

    pos_pct_all = round(tot_pos / tot_tkts * 100.0, 1)
    neu_pct_all = round(tot_neu / tot_tkts * 100.0, 1)
    neg_pct_all = round(tot_neg / tot_tkts * 100.0, 1)

    all_depts = [d["department"] for d in dept_sum]
    period_trend = build_real_period_trend_series(df_proc, "TICKETS", pos_pct_all, neu_pct_all, neg_pct_all, all_depts)
    top_pos_c, top_neg_c = extract_top_5_comments(df_proc, "TICKETS")

    return clear_and_set_module_cache("TICKETS", {
        "status": "SUCCESS", "module": "TICKETS", "activeMode": "TICKETS",
        "autoDetectedBadge": "Auto-Detected: TICKETS ONLY",
        "trendTitle": "OVERALL HOSPITAL ESCALATED TICKET SURGE & PROBLEM SHARE TREND — ALL DEPARTMENTS",
        "kpis": {
            "totalTickets": tot_tkts, "openTickets": tot_open, "solvedTickets": tot_solved,
            "topDepartment": top_open_d["department"], "topDepartmentCount": top_open_d["openTickets"], "negativeTickets": tot_neg,
        },
        "strengthHeadline": f"TOP SLA RECOVERY UNITS ({tot_solved} SOLVED | LEAD: '{top_solved_d['department'].upper()}' — {top_solved_d['positivePct']}% SOLVED)",
        "strengthExplanation": f"Departments ranked by highest resolved ticket volume and SLA completion rate across {tot_tkts:,} tickets (Model: {ML_ENGINE.get_winner_name('TICKETS')}).",
        "topStrengthCards": [{
            "department": d["department"],
            "metricBadge": f"✅ {d['solvedTickets']} Solved ({d['positivePct']}% SLA | {d['totalTickets']} Total)",
            "sentimentBreakdown": f"🎉 {d['solvedTickets']} Solved • ⏳ {d['openTickets']} Open Backlog • 📊 Share: {d['lowRatingPct']}%",
            "whyGoodReason": d["whyGoodReason"],
            "bestPractice": f"Share '{d['department']}' rapid SLA resolution workflow across other units."
        } for d in dept_best_solved[:4]],
        "sentimentHeadline": f"WHY '{top_open_d['department'].upper()}' IS THE #1 OPEN BACKLOG BOTTLENECK ({top_open_d['openTickets']} OPEN | {top_open_d['totalTickets']} TOTAL)",
        "sentimentExplanation": f"Departments ranked by highest unresolved open ticket backlog requiring immediate Guest Relations callback.",
        "rootCauseCards": [{
            "department": d["department"],
            "metricBadge": f"🚨 {d['openTickets']} Open ({d['paretoProblemPct']}% Severity | {d['totalTickets']} Total)",
            "sentimentBreakdown": f"🔥 Open Backlog: {d['openTickets']} • ✅ Solved: {d['solvedTickets']} • 📊 Hospital Share: {d['lowRatingPct']}%",
            "whyBadReason": d["topRootCause"],
            "recommendedAction": d["recommendedAction"]
        } for d in dept_worst_open[:4]],
        "touchpointRatings": dept_sum[:12],
        "subQuestionDrivers": subq_rows[:10],
        "periodTrend": period_trend,
        "topPositiveComments": top_pos_c,
        "topNegativeComments": top_neg_c,
    })


def detect_module_from_records(raw_records: list, matched_key_hint: Any = None, filename_hint: str = "") -> str:
    mk = str(matched_key_hint or "").lower().strip()
    fn_low = str(filename_hint or "").lower()
    if "nps" in fn_low:
        return "NPS"
    if "csat" in fn_low or "customer satisfaction" in fn_low:
        return "CSAT"
    if "ticket" in fn_low:
        return "TICKETS"
    if mk == "respondentwithticketdetails":
        return "FEEDBACK"

    skeys = set()
    has_responded_answer_list, has_ticket_fields = False, False
    for rec in raw_records[:30]:
        if isinstance(rec, dict):
            if isinstance(rec.get("respondedAnswerList"), list):
                has_responded_answer_list = True
            for k in rec.keys():
                kl = str(k).lower().strip()
                skeys.add(kl)
                if kl in {"ticketcode", "ticketassignee", "assignmentstatus", "ticket id", "ticketid", "isescalated", "ticketlist"}:
                    has_ticket_fields = True
    if has_responded_answer_list:
        return "FEEDBACK"
    if mk in ["tickets", "ticketlist"] or has_ticket_fields:
        return "TICKETS"
    if any("promoter" in k or "detractor" in k or "nps" in k for k in skeys):
        return "NPS"
    if any("csat" in k for k in skeys):
        return "CSAT"
    return "FEEDBACK"


def route_and_execute_strict_module(raw_records: list, forced_module: Any = None, matched_key_hint: Any = None, filename_hint: str = "") -> dict:
    if not raw_records:
        raise HTTPException(status_code=400, detail="No records received.")
    fm = str(forced_module or "").upper().strip()
    if fm not in ["NPS", "CSAT", "FEEDBACK", "TICKETS"]:
        fm = detect_module_from_records(raw_records, matched_key_hint=matched_key_hint, filename_hint=filename_hint)
    if fm == "NPS":
        return execute_nps_pipeline(raw_records)
    if fm == "CSAT":
        return execute_csat_pipeline(raw_records)
    if fm == "FEEDBACK":
        return execute_feedback_survey_pipeline(raw_records)
    return execute_ticket_triage_pipeline(raw_tickets=raw_records)