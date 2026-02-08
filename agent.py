from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import logging
import os
import re
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from dotenv import load_dotenv, find_dotenv

from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceNotFoundError

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
)

# Optional: AutoSubscribe + ConversationItemAddedEvent exist in many LiveKit agents versions.
try:
    from livekit.agents import AutoSubscribe, ConversationItemAddedEvent  # type: ignore
except Exception:  # pragma: no cover
    AutoSubscribe = None  # type: ignore
    ConversationItemAddedEvent = None  # type: ignore

# function_tool location differs across versions
try:
    from livekit.agents import function_tool
except Exception:  # pragma: no cover
    from livekit.agents.llm import function_tool  # type: ignore

from livekit.plugins import azure as azure_speech
from livekit.plugins import openai as openai_plugin

# Turn detector plugin is optional depending on installed extras.
try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
except Exception:  # pragma: no cover
    MultilingualModel = None  # type: ignore

# ACS Email is optional; if not installed, we just skip sending.
try:
    from azure.communication.email import EmailClient
except Exception:  # pragma: no cover
    EmailClient = None  # type: ignore


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("phone-agent")


# -----------------------------------------------------------------------------
# Dotenv (LOCAL ONLY) — cloud should inject real env vars
# -----------------------------------------------------------------------------

def _load_dotenv_safely() -> None:
    """
    Robust dotenv loading:
    - Cloud hosting should set env vars directly (and this won't override them).
    - Local dev can use DOTENV_PATH or typical .env / .env.local locations.
    """
    # Never override cloud-injected env vars.
    override = False

    # 1) explicit path
    explicit = (os.getenv("DOTENV_PATH") or os.getenv("DOTENV_FILE") or "").strip()
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))

    # 2) next to this script (common in repos)
    candidates.extend(
        [
            Path(__file__).with_name(".env.local"),
            Path(__file__).with_name(".env"),
        ]
    )

    # 3) current working directory (common when running "python agent.py" from root)
    candidates.extend(
        [
            Path.cwd() / ".env.local",
            Path.cwd() / ".env",
        ]
    )

    # 4) python-dotenv search (walk upwards from CWD)
    auto = find_dotenv(usecwd=True)
    if auto:
        candidates.append(Path(auto))

    used: Optional[Path] = None
    for p in candidates:
        try:
            if p and p.exists() and p.is_file():
                load_dotenv(dotenv_path=p, override=override)
                used = p
                break
        except Exception:
            # Don't break startup just because dotenv is weird
            logger.debug("dotenv load failed for %s", p, exc_info=True)

    if used:
        logger.info("Loaded dotenv from %s", used)
    else:
        # It's perfectly fine in cloud setups.
        logger.info("No .env file loaded (this is expected in cloud hosting).")


_load_dotenv_safely()


# -----------------------------------------------------------------------------
# Environment (Blob paths can be overridden by config.storage.*)
# -----------------------------------------------------------------------------

BLOB_CONTAINER = (os.getenv("BLOB_CONTAINER") or "assistant").strip()
CONFIG_BLOB = (os.getenv("CONFIG_BLOB") or "config/latest.json").strip()
CUSTOMERS_BLOB = (os.getenv("CUSTOMERS_BLOB") or "data/customers.csv").strip()
CALLS_PREFIX = (os.getenv("CALLS_PREFIX") or "calls/").strip()

DEFAULT_OPENAI_API_VERSION = (os.getenv("OPENAI_API_VERSION") or "2024-10-01-preview").strip()


# -----------------------------------------------------------------------------
# Default config (must align with the dashboard UI)
# -----------------------------------------------------------------------------

def _default_config() -> Dict[str, Any]:
    return {
        "meta": {"saved_at": None, "saved_by": None},
        "agent": {
            "system_prompt": (
                "Du bist ein deutscher Telefon-Serviceassistent.\n"
                "Ablauf: Begrüßen -> Kundennummer + Name des Anrufers + (Firmenname oder Standort) abfragen -> "
                "per Tool verifizieren -> Serviceauftrag aufnehmen -> am Ende kurz zusammenfassen.\n"
                "Regeln: Keine Emojis. Kurze Sätze. Immer eine Frage auf einmal."
            ),
            # Fallback-first: Hinweis, dass Identifikation auch ohne Kundennummer möglich ist
            "welcome_message": "Willkommen beim Service. Bitte nennen Sie Ihre Kundennummer ODER Firma und Ort.",
        },
        "speech": {
            "stt_languages": ["de-AT", "de-DE"],
            "tts_language": "de-DE",
            "tts_voice": "de-DE-KatjaNeural",
        },
        "turn": {
            "preset": "very_patient",
            "enabled": True,
            "min_endpointing_delay": 1.5,
            "max_endpointing_delay": 20.0,
        },
        "customer_verification": {
            "ask_caller_name": True,
            "max_attempts": 3,
            "match_policy": "firm_or_site",  # firm_or_site, firm_and_site
            # --- Fuzzy (fallback-first; Dashboard-Felder nicht erforderlich) ---
            "fuzzy": {
                "enabled": True,
                "max_candidates_internal": 5,
                "thresholds": {"allow": 0.86, "ask": 0.72, "block": 0.50},
                "weights": {
                    "name": 0.40,
                    "ort": 0.25,
                    "plz": 0.20,
                    "addr": 0.05,
                    "phone": 0.10,
                    "email_domain": 0.10,
                },
                "normalization": {
                    "umlaute": "ae_oe_ue",
                    "eszett": "ss",
                    "punct": "drop",
                    "spaces": "collapse"
                },
                "phonetic": "koelner",
                "stt": {"use_nbest": True, "phrase_hints": "from_csv"},
                "disambiguation": {"max_turns": 2, "order": ["plz", "ort", "email_domain", "strassen_prefix"]},
                "privacy": {"no_candidate_enumeration": True, "masked_confirmations_only": True},
            },
        },
        "email": {
            "enabled": False,
            "sender": "",
            "recipients": [],
            "subject_template": "Service Call {{callId}}",
            # "attach_config": True,  # optional: if you add it later in UI
        },
        "storage": {
            "customers_csv_blob": CUSTOMERS_BLOB,
            "calls_prefix": CALLS_PREFIX,
        },
        "llm": {
            "temperature": 0.2,
        },
    }


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class VerifiedCustomer:
    customer_id: str
    firmenname: str
    standort: str
    adresse: str
    country: str
    caller_name: str


@dataclass
class ServiceOrder:
    order_id: str
    problem: str
    priority: str
    contact_phone: str
    preferred_time: str
    timestamp_utc: str


@dataclass
class SessionArtifacts:
    config: Dict[str, Any] = field(default_factory=dict)
    customers: List[Dict[str, str]] = field(default_factory=list)

    verified_customer: Optional[VerifiedCustomer] = None
    service_order: Optional[ServiceOrder] = None

    transcript: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None

    caller_number: Optional[str] = None
    call_id: Optional[str] = None

    verification_attempts: int = 0

    # --- Fuzzy diagnostics (internal; maskierte, auditierbare Angaben) ---
    fz_decision: Optional[str] = None                 # "allow" | "ask" | "block"
    fz_score: Optional[float] = None
    fz_coverage_ok: Optional[bool] = None
    fz_features: Dict[str, float] = field(default_factory=dict)   # name/ort/plz/addr/phone/email_domain
    fz_asked: List[str] = field(default_factory=list)             # welche Merkmale wurden bereits erfragt


# -----------------------------------------------------------------------------
# Azure Blob helpers
# -----------------------------------------------------------------------------

_blob_service: Optional[BlobServiceClient] = None


def _require_env(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _blob_svc() -> BlobServiceClient:
    global _blob_service
    if _blob_service is None:
        conn = _require_env("AZURE_STORAGE_CONNECTION_STRING")
        _blob_service = BlobServiceClient.from_connection_string(conn)
    return _blob_service


def _blob_client(path: str):
    return _blob_svc().get_blob_client(container=BLOB_CONTAINER, blob=path)


def load_json_blob(path: str) -> Dict[str, Any]:
    data = _blob_client(path).download_blob().readall()
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    return json.loads(data)


def load_csv_blob(path: str) -> List[Dict[str, str]]:
    raw = _blob_client(path).download_blob().readall().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(raw)))
    # normalize whitespace
    out: List[Dict[str, str]] = []
    for r in rows:
        out.append({k: (v or "").strip() for k, v in r.items()})
    return out


def write_json_blob(path: str, payload: Dict[str, Any]) -> None:
    b = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _blob_client(path).upload_blob(
        b,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json; charset=utf-8"),
    )


# -----------------------------------------------------------------------------
# LiveKit credentials bootstrap (optional)
# -----------------------------------------------------------------------------

def bootstrap_livekit_env_from_blob() -> None:
    """
    If LIVEKIT_* are not set by the hoster, try bootstrapping them from the blob config.
    This is optional and should not break cloud startup.
    """
    if os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_API_KEY") and os.getenv("LIVEKIT_API_SECRET"):
        return

    conn = (os.getenv("AZURE_STORAGE_CONNECTION_STRING") or "").strip()
    if not conn:
        return

    try:
        bsc = BlobServiceClient.from_connection_string(conn)
        blob = bsc.get_blob_client(container=BLOB_CONTAINER, blob=CONFIG_BLOB)
        data = blob.download_blob().readall()
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")

        cfg = json.loads(data) if data else {}
        lk = cfg.get("livekit", {}) if isinstance(cfg, dict) else {}

        url = lk.get("url") or lk.get("ws_url") or lk.get("LIVEKIT_URL")
        api_key = lk.get("api_key") or lk.get("key") or lk.get("LIVEKIT_API_KEY")
        api_secret = lk.get("api_secret") or lk.get("secret") or lk.get("LIVEKIT_API_SECRET")

        if url and api_key and api_secret:
            os.environ.setdefault("LIVEKIT_URL", str(url))
            os.environ.setdefault("LIVEKIT_API_KEY", str(api_key))
            os.environ.setdefault("LIVEKIT_API_SECRET", str(api_secret))
            logger.info("Bootstrapped LIVEKIT_* from blob config.")
    except Exception:
        logger.warning("LiveKit bootstrap from blob failed (non-fatal).", exc_info=True)


bootstrap_livekit_env_from_blob()


# -----------------------------------------------------------------------------
# Email helper (ACS Email)
# -----------------------------------------------------------------------------

def _acs_connection_string() -> str:
    return (
        (os.getenv("COMMUNICATION_CONNECTION_STRING_EMAIL") or "").strip()
        or (os.getenv("AZURE_COMMUNICATION_CONNECTION_STRING") or "").strip()
    )


def _render_subject(template: str, vars: Dict[str, str]) -> str:
    """
    Very small templating: replaces {{key}} with value.
    Unknown keys stay as-is.
    """
    tmpl = template or "Service Call {{callId}}"

    def repl(m: re.Match[str]) -> str:
        key = (m.group(1) or "").strip()
        return vars.get(key, m.group(0))

    return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", repl, tmpl)


def _send_email_with_attachments(
    subject: str,
    body_text: str,
    sender: str,
    recipients: List[str],
    attachments: List[Tuple[str, str, bytes]],  # (filename, content_type, data)
) -> None:
    if not recipients:
        logger.info("Email skipped: no recipients configured.")
        return
    if not sender:
        logger.warning("Email skipped: sender is empty (config.email.sender).")
        return
    conn = _acs_connection_string()
    if not conn:
        logger.warning("Email skipped: ACS connection string missing (COMMUNICATION_CONNECTION_STRING_EMAIL).")
        return
    if EmailClient is None:
        logger.warning("Email skipped: azure.communication.email not installed.")
        return

    try:
        email_client = EmailClient.from_connection_string(conn)

        atts = []
        for (fname, ctype, data) in attachments:
            atts.append(
                {
                    "name": fname,
                    "contentType": ctype,
                    "contentInBase64": base64.b64encode(data).decode("utf-8"),
                }
            )

        message = {
            "senderAddress": sender,
            "recipients": {"to": [{"address": r} for r in recipients]},
            "content": {"subject": subject, "plainText": body_text},
            "attachments": atts,
        }

        poller = email_client.begin_send(message)
        result = poller.result()
        status = (result or {}).get("status")
        msg_id = (result or {}).get("id")
        logger.info("Email sent via ACS: status=%s id=%s", status, msg_id)
    except Exception:
        logger.exception("Failed to send email via ACS")


# -----------------------------------------------------------------------------
# Helpers: config access
# -----------------------------------------------------------------------------

def _get_cfg(artifacts: SessionArtifacts, *path: str, default: Any = None) -> Any:
    cur: Any = artifacts.config
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return default if cur is None else cur


# -----------------------------------------------------------------------------
# >>>>>>  Fuzzy utilities (local, fallback-first, no extra deps)  <<<<<<
# -----------------------------------------------------------------------------

# Lightweight normalization & similarity helpers to support robust fuzzy search
_LEGAL_SUFFIX_STOPWORDS: Set[str] = {
    "gmbh", "mbh", "ag", "kg", "ug", "ohg", "kgaa", "co", "co.", "holding", "gruppe", "group"
}

def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _strip_punct(s: str) -> str:
    return re.sub(r"[^\w\s\-]", " ", s or "")

def _german_umlaut_norm(s: str, cfg: Dict[str, Any]) -> str:
    if not s:
        return s
    # default behaviour: ae/oe/ue, ss
    if (cfg or {}).get("umlaute", "ae_oe_ue") == "ae_oe_ue":
        s = (
            s.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
             .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        )
    if (cfg or {}).get("eszett", "ss") == "ss":
        s = s.replace("ß", "ss")
    return s

def normalize_text(s: str, norm_cfg: Dict[str, Any]) -> str:
    """
    - casefold
    - german umlauts + ß -> ae/oe/ue/ss (configurable)
    - drop punctuation (configurable)
    - collapse spaces
    - drop common legal suffixes
    """
    s = s or ""
    s = _german_umlaut_norm(s, norm_cfg)
    if (norm_cfg or {}).get("punct", "drop") == "drop":
        s = _strip_punct(s)
    s = _collapse_spaces(s)
    s = s.casefold()
    # drop common legal suffixes (token-level)
    tokens = [t for t in s.split(" ") if t and (t not in _LEGAL_SUFFIX_STOPWORDS)]
    return " ".join(tokens)

def eq_norm(a: str, b: str) -> bool:
    return (a or "").strip().casefold() == (b or "").strip().casefold()

def _ngrams(s: str, n: int = 3) -> Set[str]:
    s = _collapse_spaces(s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s)-n+1)}

def sim_trigram(a: str, b: str) -> float:
    A, B = _ngrams(a, 3), _ngrams(b, 3)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def sim_levenshtein(a: str, b: str) -> float:
    # normalized Levenshtein similarity (0..1)
    a, b = a or "", b or ""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur = min(
                dp[j] + 1,        # deletion
                dp[j - 1] + 1,    # insertion
                prev + cost       # substitution
            )
            prev, dp[j] = dp[j], cur
    dist = dp[lb]
    max_len = max(la, lb)
    return 1.0 - (dist / max_len)

def sim_token_set(a: str, b: str) -> float:
    ta = set([t for t in (_collapse_spaces(a).split(" ")) if t])
    tb = set([t for t in (_collapse_spaces(b).split(" ")) if t])
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0

# very compact Kölner Phonetik (sufficient for rough calls)
def phonetic_koelner(s: str) -> str:
    s = (s or "").lower()
    s = _german_umlaut_norm(s, {"umlaute": "ae_oe_ue", "eszett": "ss"})
    s = re.sub(r"[^a-z]", "", s)
    if not s:
        return ""

    # mapping as per simplified Kölner Phonetik
    def code_char(ch: str, prev: str, nxt: str) -> str:
        if ch in "aeiouyj":
            return "0"
        if ch == "h":
            return ""
        if ch == "b":
            return "1"
        if ch in "p":
            return "1" if nxt != "h" else "3"
        if ch in "d,t":
            return "2"
        if ch in "f,v,w":
            return "3"
        if ch == "p" and nxt == "h":
            return "3"
        if ch in "g,k,q":
            return "4"
        if ch == "x":
            return "48"
        if ch in "c":
            # simple variant
            return "4"
        if ch in "s,z,ß":
            return "8"
        if ch in "l":
            return "5"
        if ch in "m,n":
            return "6"
        if ch in "r":
            return "7"
        return ""

    out: List[str] = []
    for i, ch in enumerate(s):
        prev = s[i - 1] if i > 0 else ""
        nxt = s[i + 1] if i + 1 < len(s) else ""
        out.append(code_char(ch, prev, nxt))
    # remove consecutive duplicates
    dedup: List[str] = []
    for c in out:
        if not dedup or c != dedup[-1]:
            dedup.append(c)
    return "".join(dedup)

def extract_email_domain(email: str) -> str:
    m = re.search(r"@([A-Za-z0-9\.\-]+)$", (email or "").strip())
    return (m.group(1) or "").casefold() if m else ""


# -----------------------------------------------------------------------------
# Customer verification + service order tools
# -----------------------------------------------------------------------------

@function_tool
async def verify_customer(
    context: RunContext,
    customer_id: str,
    caller_name: str = "",
    firmenname: str = "",
    standort: str = "",
) -> Dict[str, Any]:
    """
    Verifiziert Kunden gegen customers.csv aus Blob.
    Verwendet config.customer_verification:
      - ask_caller_name (bool)
      - max_attempts (int)
      - match_policy: firm_or_site | firm_and_site
    """
    artifacts: SessionArtifacts = context.userdata
    artifacts.verification_attempts += 1

    max_attempts = int(_get_cfg(artifacts, "customer_verification", "max_attempts", default=3) or 3)
    if artifacts.verification_attempts > max_attempts:
        return {
            "ok": False,
            "locked": True,
            "reason": f"Maximale Verifikationsversuche erreicht ({max_attempts}).",
        }

    customer_id = (customer_id or "").strip()
    if not customer_id:
        return {"ok": False, "reason": "Kundennummer fehlt."}

    row = next((r for r in artifacts.customers if (r.get("customer_id") or "").strip() == customer_id), None)
    if not row:
        return {"ok": False, "reason": "Kundennummer nicht gefunden."}

    ask_name = bool(_get_cfg(artifacts, "customer_verification", "ask_caller_name", default=True))
    caller_name = (caller_name or "").strip()
    if ask_name and not caller_name:
        return {"ok": False, "reason": "Name des Anrufers fehlt."}

    policy = str(_get_cfg(artifacts, "customer_verification", "match_policy", default="firm_or_site") or "firm_or_site").strip()

    firmenname_in = (firmenname or "").strip()
    standort_in = (standort or "").strip()

    firm_ok = False
    site_ok = False

    if firmenname_in:
        firm_ok = (row.get("firmenname", "") or "").strip().lower() == firmenname_in.lower()
    if standort_in:
        site_ok = (row.get("standort", "") or "").strip().lower() == standort_in.lower()

    if policy == "firm_and_site":
        if not (firmenname_in and standort_in):
            return {"ok": False, "reason": "Für die Verifikation brauche ich Firmenname UND Standort."}
        if not (firm_ok and site_ok):
            return {"ok": False, "reason": "Firmenname oder Standort stimmt nicht überein."}
    else:
        # default: firm_or_site
        if not (firmenname_in or standort_in):
            return {"ok": False, "reason": "Bitte nennen Sie Firmenname oder Standort."}
        if not (firm_ok or site_ok):
            return {"ok": False, "reason": "Firmenname oder Standort stimmt nicht überein."}

    vc = VerifiedCustomer(
        customer_id=row.get("customer_id", "") or "",
        firmenname=row.get("firmenname", "") or "",
        standort=row.get("standort", "") or "",
        adresse=row.get("adresse", "") or "",
        country=row.get("country", "") or "",
        caller_name=caller_name or "",
    )
    artifacts.verified_customer = vc

    return {
        "ok": True,
        "customer_id": vc.customer_id,
        "firmenname": vc.firmenname,
        "standort": vc.standort,
        "caller_name": vc.caller_name,
    }


@function_tool
async def submit_service_order(
    context: RunContext,
    problem: str,
    priority: str = "normal",
    contact_phone: str = "",
    preferred_time: str = "",
) -> Dict[str, Any]:
    """
    Speichert einen Serviceauftrag im Session-Context.
    """
    artifacts: SessionArtifacts = context.userdata
    if not artifacts.verified_customer:
        return {"ok": False, "reason": "Kunde ist nicht verifiziert."}

    order = ServiceOrder(
        order_id=os.urandom(8).hex(),
        problem=(problem or "").strip(),
        priority=(priority or "normal").strip(),
        contact_phone=(contact_phone or "").strip(),
        preferred_time=(preferred_time or "").strip(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    artifacts.service_order = order
    return {"ok": True, "order_id": order.order_id}


# -----------------------------------------------------------------------------
# NEW TOOL: search_customers  (fuzzy, privacy-first, no candidate enumeration)
# -----------------------------------------------------------------------------

def _norm_digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def _candidate_field(row: Dict[str, str], keys: List[str]) -> str:
    for k in keys:
        if k in row and row.get(k):
            return row.get(k, "")
    return ""

def _compute_feature_scores(
    name_q: str, ort_q: str, plz_q: str, street_prefix_q: str, domain_q: str, phone_q: str,
    row: Dict[str, str],
    norm_cfg: Dict[str, Any],
) -> Dict[str, float]:
    # extract row fields w/ flexible schema
    name_r_raw = _candidate_field(row, ["firmenname", "firma", "name", "organisation", "organisation2"])
    ort_r_raw  = _candidate_field(row, ["standort", "ort", "city", "stadt"])
    plz_r_raw  = _candidate_field(row, ["plz", "postalcode", "zip"])
    addr_r_raw = _candidate_field(row, ["adresse", "address", "straße", "strasse", "str"])
    email_r    = _candidate_field(row, ["email", "e-mail", "email1", "e_mail", "mail"])
    phone_r    = _candidate_field(row, ["phone", "telefon", "telefon büro", "telefon mobil", "telefon privat", "tel"])

    name_r = normalize_text(name_r_raw, norm_cfg)
    ort_r  = normalize_text(ort_r_raw, norm_cfg)
    addr_r = normalize_text(addr_r_raw, norm_cfg)
    plz_r  = _norm_digits(plz_r_raw)
    domain_r = extract_email_domain(email_r)
    phone_r_norm = re.sub(r"[^\d\+]", "", phone_r or "")

    # name similarity (mix char + token; phonetic as booster)
    name_char = max(sim_trigram(name_q, name_r), sim_levenshtein(name_q, name_r))
    name_tok  = sim_token_set(name_q, name_r)
    name_sim  = max(name_char, name_tok)
    if name_q and name_r and phonetic_koelner(name_q) == phonetic_koelner(name_r):
        name_sim = max(name_sim, 0.95)

    # ort similarity (char + phonetic)
    ort_char = max(sim_trigram(ort_q, ort_r), sim_levenshtein(ort_q, ort_r))
    ort_sim  = ort_char
    if ort_q and ort_r and phonetic_koelner(ort_q) == phonetic_koelner(ort_r):
        ort_sim = max(ort_sim, 0.95)

    # plz
    plz_match = 1.0 if plz_q and plz_r and plz_q == plz_r else 0.0

    # address (prefix-focused)
    addr_sim = 0.0
    if street_prefix_q and addr_r:
        if addr_r.startswith(street_prefix_q):
            addr_sim = 0.8
        else:
            addr_sim = 0.6 * sim_trigram(street_prefix_q, addr_r)

    # phone, email domain
    phone_match = 1.0 if phone_q and phone_r_norm and phone_q == phone_r_norm else 0.0
    email_domain_match = 1.0 if domain_q and domain_r and domain_q == domain_r else 0.0

    return {
        "name": float(name_sim),
        "ort": float(ort_sim),
        "plz": float(plz_match),
        "addr": float(addr_sim),
        "phone": float(phone_match),
        "email_domain": float(email_domain_match),
    }

def _weighted_score(features: Dict[str, float], weights: Dict[str, float]) -> float:
    score = 0.0
    total_w = 0.0
    for k, v in features.items():
        w = float(weights.get(k, 0.0))
        score += v * w
        total_w += w
    return (score / total_w) if total_w > 0 else 0.0

def _coverage_ok(features: Dict[str, float]) -> bool:
    positives = 0
    # define "independent positive" per feature
    if features.get("name", 0.0) >= 0.80: positives += 1
    if features.get("ort", 0.0)  >= 0.80: positives += 1
    if features.get("plz", 0.0)  >= 1.00: positives += 1
    if features.get("addr", 0.0) >= 0.60: positives += 1
    if features.get("phone", 0.0) >= 1.00: positives += 1
    if features.get("email_domain", 0.0) >= 1.00: positives += 1
    return positives >= 2

def _conflict_penalty(features: Dict[str, float], plz_q: str, ort_q: str) -> float:
    penalty = 0.0
    # If name is high but PLZ absent or contradictory and ort low -> penalize slightly
    if features.get("name", 0.0) >= 0.85 and features.get("plz", 0.0) < 1.0 and features.get("ort", 0.0) < 0.60:
        penalty += 0.05
    # Known mismatch: user provided PLZ but ort similarity is very low
    if plz_q and features.get("plz", 0.0) < 1.0 and ort_q and features.get("ort", 0.0) < 0.40:
        penalty += 0.10
    return penalty

def _choose_ask_next(order: List[str], provided: Dict[str, str], asked: List[str], features: Dict[str, float]) -> str:
    already = set(asked or [])
    for feat in order:
        if feat in already:
            continue
        # ask for the most informative missing/weak feature
        if feat == "plz" and not provided.get("plz"):
            return "plz"
        if feat == "ort" and not provided.get("ort"):
            return "ort"
        if feat == "email_domain" and not provided.get("email_domain"):
            return "email_domain"
        if feat == "strassen_prefix" and not provided.get("street_prefix"):
            return "strassen_prefix"
    # fallback: ask for the weakest (if any)
    weakest = min(features.items(), key=lambda kv: kv[1])[0] if features else ""
    mapping = {"email_domain": "email_domain", "addr": "strassen_prefix"}
    return mapping.get(weakest, "")

@function_tool
async def search_customers(
    context: RunContext,
    firmenname: str = "",
    ort: str = "",
    plz: str = "",
    email: str = "",
    phone_e164: str = "",
    street_prefix: str = "",
) -> Dict[str, Any]:
    """
    Interne fuzzy Kandidatensuche OHNE Kandidatennennung nach außen.
    Rückgabe steuert Dialog: (decision, ask_next) und False-Positive-Guard via score/coverage.
    """
    artifacts: SessionArtifacts = context.userdata
    cfg = artifacts.config or {}
    cv_cfg = (cfg.get("customer_verification") or {}) if isinstance(cfg, dict) else {}
    fuzzy_cfg = (cv_cfg.get("fuzzy") or {}) if isinstance(cv_cfg, dict) else {}

    # defaults (fallback-first)
    thresholds = (fuzzy_cfg.get("thresholds") or {"allow": 0.86, "ask": 0.72, "block": 0.50})
    weights = (fuzzy_cfg.get("weights") or {"name": 0.40, "ort": 0.25, "plz": 0.20, "addr": 0.05, "phone": 0.10, "email_domain": 0.10})
    norm_cfg = (fuzzy_cfg.get("normalization") or {"umlaute": "ae_oe_ue", "eszett": "ss", "punct": "drop", "spaces": "collapse"})
    disamb = (fuzzy_cfg.get("disambiguation") or {"max_turns": 2, "order": ["plz", "ort", "email_domain", "strassen_prefix"]})
    max_k = int(fuzzy_cfg.get("max_candidates_internal", 5) or 5)

    # normalize inputs
    name_q = normalize_text(firmenname, norm_cfg)
    ort_q = normalize_text(ort, norm_cfg)
    plz_q = _norm_digits(plz)
    domain_q = extract_email_domain(email)
    phone_q = re.sub(r"[^\d\+]", "", phone_e164 or "")
    street_prefix_q = normalize_text(street_prefix, norm_cfg)

    provided = {
        "name": name_q,
        "ort": ort_q,
        "plz": plz_q,
        "email_domain": domain_q,
        "phone": phone_q,
        "street_prefix": street_prefix_q,
    }

    # compute features + scores over all customers
    scored: List[Tuple[str, float, Dict[str, float]]] = []
    for row in artifacts.customers or []:
        cid = (row.get("customer_id") or "").strip()
        if not cid:
            continue
        feats = _compute_feature_scores(name_q, ort_q, plz_q, street_prefix_q, domain_q, phone_q, row, norm_cfg)
        score_base = _weighted_score(feats, weights)
        penalty = _conflict_penalty(feats, plz_q, ort_q)
        score = max(0.0, score_base - penalty)
        scored.append((cid, score, feats))

    # shortlist
    scored.sort(key=lambda t: t[1], reverse=True)
    shortlist = scored[:max_k] if max_k > 0 else scored
    best_cid, best_score, best_feats = (shortlist[0] if shortlist else ("", 0.0, {}))

    cov_ok = _coverage_ok(best_feats) if shortlist else False

    # decision
    allow_th = float(thresholds.get("allow", 0.86))
    ask_th = float(thresholds.get("ask", 0.72))
    # block_th not explicitly needed here; everything < ask_th is effectively "block"

    if best_score >= allow_th and cov_ok:
        decision = "allow"
        ask_next = ""
    elif best_score >= ask_th:
        decision = "ask"
        order = list((disamb.get("order") or ["plz", "ort", "email_domain", "strassen_prefix"]))
        ask_next = _choose_ask_next(order, provided, artifacts.fz_asked, best_feats)
    else:
        decision = "block"
        ask_next = _choose_ask_next(list((disamb.get("order") or [])), provided, artifacts.fz_asked, best_feats)

    # update artifacts (maskierte Diagnostik)
    artifacts.fz_decision = decision
    artifacts.fz_score = float(best_score)
    artifacts.fz_coverage_ok = bool(cov_ok)
    artifacts.fz_features = {
        "name": float(best_feats.get("name", 0.0)),
        "ort": float(best_feats.get("ort", 0.0)),
        "plz": float(best_feats.get("plz", 0.0)),
        "addr": float(best_feats.get("addr", 0.0)),
        "phone": float(best_feats.get("phone", 0.0)),
        "email_domain": float(best_feats.get("email_domain", 0.0)),
    }

    # track which features were provided this turn (approximates "asked features")
    newly_provided: List[str] = []
    if plz_q: newly_provided.append("plz")
    if ort_q: newly_provided.append("ort")
    if domain_q: newly_provided.append("email_domain")
    if street_prefix_q: newly_provided.append("strassen_prefix")
    if newly_provided:
        # maintain uniqueness while preserving order
        seen = set(artifacts.fz_asked)
        for x in newly_provided:
            if x not in seen:
                artifacts.fz_asked.append(x)
                seen.add(x)

    return {
        "ok": True,
        "decision": decision,
        "score": float(best_score),
        "coverage_ok": bool(cov_ok),
        "best_candidate_customer_id": best_cid,  # INTERNAL ONLY; never speak this
        "ask_next": ask_next,
        "features": artifacts.fz_features,
        "privacy": {"enumerated": False},
    }


# -----------------------------------------------------------------------------
# Transcript helpers
# -----------------------------------------------------------------------------

def _safe_calls_prefix(prefix: str) -> str:
    p = (prefix or "").strip() or "calls/"
    return p if p.endswith("/") else (p + "/")

def _safe_blob_name(name: str) -> str:
    return (name or "").replace("/", "_").replace("\\", "_").strip() or "unknown"

def _history_to_transcript(session: AgentSession) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in getattr(session.history, "items", []):
        if getattr(item, "type", None) == "message":
            role = getattr(item, "role", "") or ""
            text = getattr(item, "text_content", "") or ""
            interrupted = bool(getattr(item, "interrupted", False))
            if not (text or "").strip():
                continue
            mapped_role = "agent" if role == "assistant" else role
            entry: Dict[str, Any] = {"role": mapped_role, "text": text}
            if interrupted:
                entry["interrupted"] = True
            out.append(entry)
    return out


# -----------------------------------------------------------------------------
# LiveKit Agent Server entrypoint
# -----------------------------------------------------------------------------

server = AgentServer()


@server.rtc_session(agent_name=os.getenv("AGENT_NAME", "phone-assistant"))
async def entrypoint(ctx: JobContext):
    logger.info("Starting phone-agent in room=%s", getattr(ctx.room, "name", "?"))

    # Optional: connect early (telephony setups behave better)
    if AutoSubscribe is not None:
        try:
            await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)  # type: ignore[attr-defined]
        except Exception:
            logger.info("ctx.connect(auto_subscribe=AUDIO_ONLY) failed/unsupported (continuing).", exc_info=True)
    else:
        try:
            await ctx.connect()  # type: ignore[attr-defined]
        except Exception:
            # Some versions auto-connect when session starts.
            logger.debug("ctx.connect not available.", exc_info=True)

    # Try to fetch caller number from participant attributes (if available)
    caller_number: str = ""
    try:
        participant = await ctx.wait_for_participant()  # type: ignore[attr-defined]
        attrs = getattr(participant, "attributes", {}) or {}
        caller_number = (
            attrs.get("sip.phoneNumber")
            or attrs.get("caller.number")
            or attrs.get("sip.callerNumber")
            or ""
        )
        logger.info("Participant=%s caller_number=%s", getattr(participant, "identity", ""), caller_number)
    except Exception:
        # Not fatal; some deployments don't expose this
        logger.debug("wait_for_participant not available or failed.", exc_info=True)

    # ---------------------------------------------------------------
    # Load config + customers (per call)
    # ---------------------------------------------------------------
    try:
        cfg = await asyncio.to_thread(load_json_blob, CONFIG_BLOB)
        if not isinstance(cfg, dict):
            raise ValueError("Config is not a JSON object")
    except ResourceNotFoundError:
        logger.warning("Config blob not found: %s — using default config.", CONFIG_BLOB)
        cfg = _default_config()
    except Exception:
        logger.exception("Failed to load config — using default config.")
        cfg = _default_config()

    storage_cfg = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
    customers_blob = (storage_cfg.get("customers_csv_blob") or CUSTOMERS_BLOB) if isinstance(storage_cfg, dict) else CUSTOMERS_BLOB
    calls_prefix = (storage_cfg.get("calls_prefix") or CALLS_PREFIX) if isinstance(storage_cfg, dict) else CALLS_PREFIX

    try:
        customers = await asyncio.to_thread(load_csv_blob, customers_blob)
    except Exception:
        logger.exception("Failed to load customers CSV from blob=%s (continuing with empty list).", customers_blob)
        customers = []

    # Robust call_id
    job_obj = getattr(ctx, "job", None)
    call_id = (
        getattr(job_obj, "id", None)
        or getattr(job_obj, "dispatch_id", None)
        or getattr(job_obj, "dispatchId", None)
        or getattr(ctx.room, "name", None)
        or "unknown"
    )
    call_id = _safe_blob_name(str(call_id))

    artifacts = SessionArtifacts(
        config=cfg,
        customers=customers,
        caller_number=caller_number or getattr(ctx.room, "name", "") or None,
        call_id=call_id,
    )

    agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
    speech_cfg = cfg.get("speech", {}) if isinstance(cfg, dict) else {}
    turn_cfg = cfg.get("turn", {}) if isinstance(cfg, dict) else {}
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    email_cfg = cfg.get("email", {}) if isinstance(cfg, dict) else {}
    cv_cfg = cfg.get("customer_verification", {}) if isinstance(cfg, dict) else {}

    # ---------------------------------------------------------------
    # Turn detector (optional)
    # ---------------------------------------------------------------
    turn_detection = None
    turn_enabled = bool((turn_cfg or {}).get("enabled", True))
    preset = str((turn_cfg or {}).get("preset", "very_patient")).strip().lower()

    if turn_enabled and preset not in ("off", "disabled", "false", "0", "no"):
        if MultilingualModel is None:
            logger.warning("Turn detector requested but MultilingualModel not available (disabled).")
        else:
            try:
                turn_detection = MultilingualModel()  # type: ignore[call-arg]
            except Exception:
                logger.warning("Turn detector init failed (disabled).", exc_info=True)

    # ---------------------------------------------------------------
    # LLM (Azure OpenAI)
    # ---------------------------------------------------------------
    llm = openai_plugin.LLM.with_azure(
        azure_deployment=_require_env("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_require_env("AZURE_OPENAI_API_KEY"),
        api_version=DEFAULT_OPENAI_API_VERSION,
        temperature=float((llm_cfg or {}).get("temperature", 0.2)),
    )

    # ---------------------------------------------------------------
    # Speech (Azure Speech)
    # ---------------------------------------------------------------
    stt_langs = (speech_cfg or {}).get("stt_languages") or ["de-AT", "de-DE"]
    if not isinstance(stt_langs, list):
        stt_langs = ["de-AT", "de-DE"]

    # Build optional phrase hints from customers (firmenname + standort)
    phrase_hints: List[str] = []
    try:
        fuzzy_cfg = (cv_cfg.get("fuzzy") or {}) if isinstance(cv_cfg, dict) else {}
        use_hints = bool((fuzzy_cfg.get("stt") or {}).get("phrase_hints", "") == "from_csv")
        if use_hints and customers:
            seen: Set[str] = set()
            for r in customers:
                for k in ("firmenname", "standort", "ort", "city"):
                    v = (r.get(k) or "").strip()
                    if not v:
                        continue
                    if v in seen:
                        continue
                    seen.add(v)
                    phrase_hints.append(v)
                    if len(phrase_hints) >= 500:
                        break
                if len(phrase_hints) >= 500:
                    break
    except Exception:
        logger.debug("Failed to build phrase hints (continuing).", exc_info=True)

    # Instantiate STT with optional hints (fallback if not supported)
    try:
        if phrase_hints:
            try:
                stt = azure_speech.STT(language=stt_langs, hints=phrase_hints)  # type: ignore[call-arg]
            except TypeError:
                # older versions may not support 'hints'
                stt = azure_speech.STT(language=stt_langs)
        else:
            stt = azure_speech.STT(language=stt_langs)
    except Exception:
        logger.warning("azure_speech.STT(language=...) failed; falling back to default STT().", exc_info=True)
        stt = azure_speech.STT()

    try:
        tts = azure_speech.TTS(
            voice=(speech_cfg or {}).get("tts_voice", "de-DE-KatjaNeural"),
            language=(speech_cfg or {}).get("tts_language", "de-DE"),
        )
    except Exception:
        logger.warning("azure_speech.TTS(voice/language) failed; falling back to default TTS().", exc_info=True)
        tts = azure_speech.TTS()

    # ---------------------------------------------------------------
    # Agent session
    # ---------------------------------------------------------------
    session = AgentSession(
        userdata=artifacts,
        stt=stt,
        tts=tts,
        turn_detection=turn_detection,
        min_endpointing_delay=float((turn_cfg or {}).get("min_endpointing_delay", 1.5)),
        max_endpointing_delay=float((turn_cfg or {}).get("max_endpointing_delay", 20.0)),
        llm=llm,
    )

    # Prefer event-based transcript if supported
    try:
        @session.on("conversation_item_added")
        def _on_item(ev: Any):  # ConversationItemAddedEvent, but keep it version-agnostic
            item = getattr(ev, "item", None)
            if item is None:
                return
            role = getattr(item, "role", "") or ""
            text = getattr(item, "text_content", "") or ""
            interrupted = bool(getattr(item, "interrupted", False))
            if not (text or "").strip():
                return
            mapped_role = "agent" if role == "assistant" else role
            artifacts.transcript.append({"role": mapped_role, "text": text, "interrupted": interrupted})
    except Exception:
        logger.debug("conversation_item_added event not available.", exc_info=True)

    # ---------------------------------------------------------------
    # Instructions (dashboard uses agent.system_prompt)
    # ---------------------------------------------------------------
    base_prompt = (
        (agent_cfg or {}).get("system_prompt")
        or (agent_cfg or {}).get("instructions")
        or (agent_cfg or {}).get("prompt")
        or _default_config()["agent"]["system_prompt"]
    )

    # Hard runtime constraints from config (so the agent behaves as configured)
    ask_name = bool((cv_cfg or {}).get("ask_caller_name", True))
    max_attempts = int((cv_cfg or {}).get("max_attempts", 3) or 3)
    match_policy = str((cv_cfg or {}).get("match_policy", "firm_or_site") or "firm_or_site")

    runtime_rules = (
        "\n\n"
        "Laufende Konfiguration (muss eingehalten werden):\n"
        f"- Name des Anrufers abfragen: {'ja' if ask_name else 'nein'}\n"
        f"- Max. Verifikationsversuche: {max_attempts}\n"
        f"- Match Policy: {match_policy}\n"
        "Regeln:\n"
        "- Sprich immer deutsch.\n"
        "- Kurze Sätze. Keine Emojis.\n"
        "- Verifiziere den Kunden IMMER über das Tool verify_customer.\n"
        "- Wenn locked=true zurückkommt: freundlich abbrechen oder an menschlichen Support verweisen.\n"
        "- Nach erfolgreicher Verifikation: Serviceauftrag strukturiert aufnehmen und submit_service_order verwenden.\n"
    )

    # Fuzzy addendum (privacy-first) — enabled by default (fallback-first)
    fuzzy_cfg = (cv_cfg or {}).get("fuzzy", {}) if isinstance(cv_cfg, dict) else {}
    fuzzy_enabled = bool((fuzzy_cfg or {}).get("enabled", True))
    fuzzy_rules = ""
    if fuzzy_enabled:
        fuzzy_rules = (
            "\n\n"
            "Fuzzy-Suche & Datenschutz (verbindlich):\n"
            "- Nutze zuerst search_customers zur internen Kandidatensuche.\n"
            "- Nenne niemals Kandidaten, Adressen oder PLZ aus dem System.\n"
            "- Stelle genau eine Frage pro Turn: PLZ, Ort, E-Mail-Domain oder Straßen-Prefix.\n"
            "- Verifiziere nur, wenn decision=allow und coverage_ok=true. Sonst eine Zusatzfrage oder Eskalation.\n"
            "- Gib niemals customer_id oder interne Systemdaten an den Anrufer weiter.\n"
            "- Bestätige maskiert (z. B. 'Beginnt Ihre PLZ mit 50…?').\n"
            "- Biete Buchstabieren an, wenn unklar.\n"
        )

    instructions = (str(base_prompt).strip() + runtime_rules + fuzzy_rules).strip()

    agent = Agent(
        instructions=instructions,
        tools=[verify_customer, submit_service_order, search_customers] if fuzzy_enabled else [verify_customer, submit_service_order],
    )

    # ---------------------------------------------------------------
    # Shutdown callback → persist log to Blob + optional email
    # ---------------------------------------------------------------
    async def persist_and_notify():
        # Build transcript
        transcript = artifacts.transcript or _history_to_transcript(session)

        payload: Dict[str, Any] = {
            "call_id": call_id,
            "room": getattr(ctx.room, "name", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "caller": artifacts.caller_number,
            "config_meta": (cfg.get("meta") if isinstance(cfg, dict) else None),
            "verified_customer": asdict(artifacts.verified_customer) if artifacts.verified_customer else None,
            "service_order": asdict(artifacts.service_order) if artifacts.service_order else None,
            "transcript": transcript,
            "summary": artifacts.summary,
        }

        # Add masked fuzzy diagnostics (no PII)
        try:
            payload["fuzzy_diagnostics"] = {
                "enabled": bool(fuzzy_enabled),
                "decision": artifacts.fz_decision,
                "score": artifacts.fz_score,
                "coverage_ok": artifacts.fz_coverage_ok,
                "features": artifacts.fz_features,
                "asked_features": artifacts.fz_asked,
                "privacy": {"no_candidate_enumeration": True},
            }
        except Exception:
            logger.debug("Adding fuzzy diagnostics failed (continuing).", exc_info=True)

        # 1) write to blob
        try:
            log_path = f"{_safe_calls_prefix(calls_prefix)}{call_id}.json"
            await asyncio.to_thread(write_json_blob, log_path, payload)
            logger.info("Call log written to %s", log_path)
        except Exception:
            logger.exception("Failed to persist call log to blob")

        # 2) optional email
        try:
            enabled = bool((email_cfg or {}).get("enabled", False))
            if not enabled:
                return

            sender = str((email_cfg or {}).get("sender", "") or "").strip()
            recipients = (email_cfg or {}).get("recipients", []) or []
            if not isinstance(recipients, list):
                recipients = []
            recipients = [str(r).strip() for r in recipients if str(r).strip()]

            subject_tmpl = str((email_cfg or {}).get("subject_template", "Service Call {{callId}}") or "Service Call {{callId}}")
            subject = _render_subject(
                subject_tmpl,
                {
                    "callId": call_id,
                    "room": str(getattr(ctx.room, "name", "") or ""),
                    "customerId": str(getattr(artifacts.verified_customer, "customer_id", "") or ""),
                },
            )

            # Short email body (details are attachments)
            lines: List[str] = []
            lines.append(f"Call ID: {call_id}")
            if artifacts.caller_number:
                lines.append(f"Caller: {artifacts.caller_number}")
            if artifacts.verified_customer:
                lines.append(
                    f"Kunde: {artifacts.verified_customer.customer_id} / {artifacts.verified_customer.firmenname} / {artifacts.verified_customer.standort}"
                )
            if artifacts.service_order:
                lines.append(f"Auftrag: {artifacts.service_order.order_id} Priorität={artifacts.service_order.priority}")
                if artifacts.service_order.problem:
                    lines.append(f"Problem: {artifacts.service_order.problem[:300]}")
            body = "\n".join(lines)

            # Attach call log + config snapshot
            call_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            cfg_bytes = json.dumps(cfg, ensure_ascii=False, indent=2).encode("utf-8")

            attachments = [
                (f"call_{call_id}.json", "application/json", call_bytes),
                (f"config_{call_id}.json", "application/json", cfg_bytes),
            ]

            await asyncio.to_thread(
                _send_email_with_attachments,
                subject,
                body,
                sender,
                recipients,
                attachments,
            )
        except Exception:
            logger.exception("Email notification failed")

    ctx.add_shutdown_callback(persist_and_notify)

    # ---------------------------------------------------------------
    # Start session & welcome
    # ---------------------------------------------------------------
    await session.start(agent=agent, room=ctx.room)

    # Prefer configured welcome; otherwise fallback-first text already in defaults
    welcome = (agent_cfg or {}).get(
        "welcome_message",
        _default_config()["agent"]["welcome_message"],
    )
    await session.say(str(welcome), allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(server)
