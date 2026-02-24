"""phone-agent — German-language AI phone service agent (LiveKit + Azure)."""
from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────────────
import os
import re
import sys
import csv
import io
import json
import logging
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# ── dotenv ──────────────────────────────────────────────────────────────────
from dotenv import load_dotenv, find_dotenv

# ── azure blob ──────────────────────────────────────────────────────────────
from azure.storage.blob import BlobServiceClient

# ── azure identity / key vault ───────────────────────────────────────────────
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# ── livekit core (try/except for version resilience) ────────────────────────
try:
    from livekit.agents import (
        AutoSubscribe,
        JobContext,
        WorkerOptions,
        cli,
    )
    from livekit.agents import function_tool, RunContext, Agent, AgentSession
except ImportError:
    from livekit.agents import (
        AutoSubscribe,
        JobContext,
        WorkerOptions,
        cli,
    )
    from livekit.agents import function_tool, RunContext, Agent, AgentSession

# ── livekit plugins ─────────────────────────────────────────────────────────
from livekit.plugins import azure as livekit_azure
from livekit.plugins import openai as livekit_openai

# ── optional: ACS Email ─────────────────────────────────────────────────────
HAS_ACS_EMAIL = False
try:
    from azure.communication.email import EmailClient
    HAS_ACS_EMAIL = True
except Exception:
    pass

# ── constants ───────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
BLOB_CONTAINER = os.environ.get("BLOB_CONTAINER", "assistant")
CONFIG_BLOB = os.environ.get("CONFIG_BLOB", "config/latest.json")
CUSTOMERS_BLOB = os.environ.get("CUSTOMERS_BLOB", "data/customers.csv")
CALLS_PREFIX = os.environ.get("CALLS_PREFIX", "calls/")
AGENT_NAME = os.environ.get("AGENT_NAME", "phone-assistant")

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# §3  .env loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_dotenv() -> None:
    """Search for .env files in multiple locations, never overriding existing vars."""
    # 1. explicit path from env
    explicit = os.environ.get("DOTENV_PATH") or os.environ.get("DOTENV_FILE")
    if explicit and os.path.isfile(explicit):
        load_dotenv(explicit, override=False)
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. next to script
    for name in (".env.local", ".env"):
        p = os.path.join(script_dir, name)
        if os.path.isfile(p):
            load_dotenv(p, override=False)

    # 3. CWD
    cwd = os.getcwd()
    if os.path.normpath(cwd) != os.path.normpath(script_dir):
        for name in (".env.local", ".env"):
            p = os.path.join(cwd, name)
            if os.path.isfile(p):
                load_dotenv(p, override=False)

    # 4. walk upward
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)


_load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# §4  Key Vault — fetched once at process startup, cached in-memory
# ═══════════════════════════════════════════════════════════════════════════

_KV_SECRET_MAP: Dict[str, str] = {
    "azure-openai-api-key":                   "AZURE_OPENAI_API_KEY",
    "azure-openai-endpoint":                  "AZURE_OPENAI_ENDPOINT",
    "azure-speech-key":                       "AZURE_SPEECH_KEY",
    "azure-speech-region":                    "AZURE_SPEECH_REGION",
    "azure-storage-connection-string":        "AZURE_STORAGE_CONNECTION_STRING",
    "livekit-url":                            "LIVEKIT_URL",
    "livekit-api-key":                        "LIVEKIT_API_KEY",
    "livekit-api-secret":                     "LIVEKIT_API_SECRET",
    "communication-connection-string-email": "AZURE_COMMUNICATION_CONNECTION_STRING",
}

_kv_cache: Dict[str, str] = {}


def _load_keyvault_secrets() -> None:
    """Pull all secrets from KV via Managed Identity. Non-fatal if KV unreachable."""
    kv_url = os.environ.get("AZURE_KEYVAULT_URL", "")
    if not kv_url:
        logger.info("AZURE_KEYVAULT_URL not set — using env vars only.")
        return
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=kv_url, credential=credential)
        fetched = 0
        for secret_name, env_name in _KV_SECRET_MAP.items():
            try:
                secret = client.get_secret(secret_name)
                if secret.value:
                    _kv_cache[env_name] = secret.value
                    fetched += 1
            except Exception as exc:
                logger.debug("KV: could not fetch %r: %s", secret_name, exc)
        logger.info("Key Vault: fetched %d/%d secrets.", fetched, len(_KV_SECRET_MAP))
    except Exception:
        logger.warning("Key Vault init failed — falling back to env vars.", exc_info=True)


def _get_secret(env_name: str, fallback: str = "") -> str:
    """Priority: KV cache → os.environ → fallback. Works for local dev too."""
    return _kv_cache.get(env_name) or os.environ.get(env_name, "") or fallback


_load_keyvault_secrets()
# Inject KV secrets into os.environ so the LiveKit framework can read them at startup
# (WorkerOptions reads LIVEKIT_API_KEY/SECRET/URL before entrypoint() is called)
for _env_name, _val in _kv_cache.items():
    os.environ.setdefault(_env_name, _val)

# ═══════════════════════════════════════════════════════════════════════════
# §7  Data models
# ═══════════════════════════════════════════════════════════════════════════

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
    order_id: str          # os.urandom(8).hex()
    problem: str
    priority: str          # default "normal"
    contact_phone: str
    preferred_time: str
    timestamp_utc: str     # datetime.now(timezone.utc).isoformat()


@dataclass
class SessionArtifacts:
    config: Dict[str, Any] = field(default_factory=dict)
    customers: List[Dict[str, str]] = field(default_factory=list)

    verified_customer: Optional[VerifiedCustomer] = None
    service_order: Optional[ServiceOrder] = None

    transcript: List[Dict[str, Any]] = field(default_factory=list)
    # entries: {"role": "agent"|"user", "text": str, "interrupted": bool}
    summary: Optional[str] = None

    caller_number: Optional[str] = None
    call_id: Optional[str] = None

    verification_attempts: int = 0

    # Fuzzy diagnostics (masked, no PII — written to call log)
    fz_decision: Optional[str] = None       # "allow" | "ask" | "block"
    fz_score: Optional[float] = None
    fz_coverage_ok: Optional[bool] = None
    fz_features: Dict[str, float] = field(default_factory=dict)
    fz_asked: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# §5  Default config
# ═══════════════════════════════════════════════════════════════════════════

def _default_config() -> Dict[str, Any]:
    return {
        "meta": {
            "saved_at": None,
            "saved_by": None,
        },
        "livekit": {
            "url": "",
            "api_key": "",
            "api_secret": "",
        },
        "azure_openai": {
            "endpoint": "",
            "api_key": "",
            "deployment": "gpt-4o-mini",
            "api_version": "2024-10-01-preview",
        },
        "azure_speech": {
            "key": "",
            "region": "",
        },
        "agent": {
            "system_prompt": (
                "Du bist ein deutscher Telefon-Serviceassistent.\n"
                "Ablauf: Begrüßen -> Kundennummer + Name des Anrufers + "
                "(Firmenname oder Standort) abfragen -> per Tool verifizieren -> "
                "Serviceauftrag aufnehmen -> am Ende kurz zusammenfassen.\n"
                "Regeln: Keine Emojis. Kurze Sätze. Immer eine Frage auf einmal."
            ),
            "welcome_message": (
                "Willkommen beim Service. Bitte nennen Sie Ihre Kundennummer "
                "ODER Firma und Ort."
            ),
            "runtime_rules_template": (
                "Laufende Konfiguration (muss eingehalten werden):\n"
                "- Name des Anrufers abfragen: {{ask_caller_name}}\n"
                "- Max. Verifikationsversuche: {{max_attempts}}\n"
                "Regeln:\n"
                "- Sprich immer deutsch.\n"
                "- Kurze Sätze. Keine Emojis.\n"
                "- Verifiziere den Kunden IMMER über das Tool verify_customer.\n"
                "- Wenn locked=true zurückkommt: freundlich abbrechen oder an "
                "menschlichen Support verweisen.\n"
                "- Nach erfolgreicher Verifikation: Serviceauftrag strukturiert "
                "aufnehmen und submit_service_order verwenden."
            ),
            "fuzzy_rules": (
                "Fuzzy-Suche & Datenschutz (verbindlich):\n"
                "- Nutze zuerst search_customers zur internen Kandidatensuche.\n"
                "- Nenne niemals Kandidaten oder Adressen aus dem System.\n"
                "- Stelle genau eine Frage pro Turn: Standort oder Straßen-Prefix.\n"
                "- Verifiziere nur, wenn decision=allow und coverage_ok=true. "
                "Sonst eine Zusatzfrage oder Eskalation.\n"
                "- Gib niemals customer_id oder interne Systemdaten an den Anrufer weiter.\n"
                "- Biete Buchstabieren an, wenn unklar."
            ),
        },
        "speech": {
            "tts_voice": "de-DE-ConradNeural",
            "language": "de-DE",
            "stt_language": "de-DE",
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
            "fuzzy": {
                "enabled": True,
                "max_candidates_internal": 5,
                "thresholds": {
                    "allow": 0.86,
                    "ask": 0.72,
                    "block": 0.50,
                },
                "weights": {
                    "name": 0.60,
                    "ort": 0.30,
                    "addr": 0.10,
                },
                "normalization": {
                    "umlaute": "ae_oe_ue",
                    "eszett": "ss",
                    "punct": "drop",
                    "spaces": "collapse",
                    "legal_suffixes": [
                        "gmbh", "mbh", "ag", "kg", "ug", "ohg",
                        "kgaa", "co", "holding", "gruppe", "group",
                    ],
                },
                "phonetic": "koelner",
                "coverage": {
                    "name": 0.80,
                    "ort": 0.80,
                    "addr": 0.60,
                    "min_positives": 1,
                },
                "conflict_penalty": {
                    "name_threshold": 0.85,
                    "ort_threshold": 0.60,
                    "same_name_penalty": 0.05,
                },
                "stt": {
                    "phrase_hints": "from_csv",
                    "phrase_hints_max": 500,
                },
                "disambiguation": {
                    "max_turns": 2,
                    "order": ["ort", "strassen_prefix"],
                },
                "privacy": {
                    "no_candidate_enumeration": True,
                    "masked_confirmations_only": True,
                },
            },
        },
        "email": {
            "enabled": False,
            "sender": "",
            "recipients": [],
            "subject_template": "Service Call {{callId}}",
        },
        "storage": {
            "customers_csv_blob": "data/customers.csv",
            "calls_prefix": "calls/",
        },
        "customization": {
            "app_name": "phone-agent",
        },
        "llm": {
            "temperature": 0.2,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# §5  Blob helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_blob_service_client() -> BlobServiceClient:
    conn_str = _get_secret("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set.")
    return BlobServiceClient.from_connection_string(conn_str)


def load_blob_text(blob_path: str) -> str:
    try:
        client = get_blob_service_client()
        blob = client.get_blob_client(container=BLOB_CONTAINER, blob=blob_path)
        return blob.download_blob().readall().decode("utf-8")
    except Exception:
        logger.debug("Blob not found or unreadable: %s", blob_path)
        return ""


def upload_blob_text(blob_path: str, content: str) -> None:
    try:
        client = get_blob_service_client()
        blob = client.get_blob_client(container=BLOB_CONTAINER, blob=blob_path)
        blob.upload_blob(content.encode("utf-8"), overwrite=True)
    except Exception:
        logger.exception("Failed to upload blob: %s", blob_path)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config() -> Dict[str, Any]:
    defaults = _default_config()
    raw = load_blob_text(CONFIG_BLOB)
    if not raw.strip():
        logger.info("No blob config found at %s — using defaults.", CONFIG_BLOB)
        return defaults
    try:
        blob_cfg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in %s — using defaults.", CONFIG_BLOB)
        return defaults
    return _deep_merge(defaults, blob_cfg)


def load_customers(blob_path: str = CUSTOMERS_BLOB) -> List[Dict[str, str]]:
    raw = load_blob_text(blob_path)
    if not raw.strip():
        logger.warning("Customer CSV is empty or missing: %s", blob_path)
        return []
    # Handle UTF-8 BOM
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    reader = csv.DictReader(io.StringIO(raw))
    rows: List[Dict[str, str]] = []
    for row in reader:
        cleaned = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        rows.append(cleaned)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# §8  Fuzzy matching system (pure Python)
# ═══════════════════════════════════════════════════════════════════════════

# ── 8.1 Helpers and normalization ───────────────────────────────────────────

_LEGAL_SUFFIX_STOPWORDS: Set[str] = {
    "gmbh", "mbh", "ag", "kg", "ug", "ohg", "kgaa",
    "co", "co.", "holding", "gruppe", "group",
}


def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _strip_punct(s: str) -> str:
    return re.sub(r"[^\w\s\-]", " ", s or "")


def _german_umlaut_norm(s: str, cfg: Dict[str, Any]) -> str:
    if not s:
        return s
    if (cfg or {}).get("umlaute", "ae_oe_ue") == "ae_oe_ue":
        s = (
            s.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
             .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        )
    if (cfg or {}).get("eszett", "ss") == "ss":
        s = s.replace("ß", "ss")
    return s


def normalize_text(s: str, norm_cfg: Dict[str, Any]) -> str:
    s = s or ""
    s = _german_umlaut_norm(s, norm_cfg)
    if (norm_cfg or {}).get("punct", "drop") == "drop":
        s = _strip_punct(s)
    s = _collapse_spaces(s)
    s = s.casefold()
    # drop legal suffixes loaded from config (or module-level default)
    legal = set(norm_cfg.get("legal_suffixes", [])) or _LEGAL_SUFFIX_STOPWORDS
    tokens = [t for t in s.split(" ") if t and t not in legal]
    return " ".join(tokens)


# ── 8.2 Similarity functions ───────────────────────────────────────────────

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
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev, dp[j] = dp[j], cur
    return 1.0 - (dp[lb] / max(la, lb))


def sim_token_set(a: str, b: str) -> float:
    ta = set(t for t in _collapse_spaces(a).split(" ") if t)
    tb = set(t for t in _collapse_spaces(b).split(" ") if t)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


# ── 8.3 Kölner Phonetik ────────────────────────────────────────────────────

def phonetic_koelner(s: str) -> str:
    s = (s or "").lower()
    s = _german_umlaut_norm(s, {"umlaute": "ae_oe_ue", "eszett": "ss"})
    s = re.sub(r"[^a-z]", "", s)
    if not s:
        return ""

    def code_char(ch: str, prev: str, nxt: str) -> str:
        if ch in "aeiouyj":  return "0"
        if ch == "h":        return ""
        if ch == "b":        return "1"
        if ch == "p":        return "1" if nxt != "h" else "3"
        if ch in "dt":       return "2"
        if ch in "fvw":      return "3"
        if ch in "gkq":      return "4"
        if ch == "x":        return "48"
        if ch == "c":        return "4"
        if ch in "szß":      return "8"
        if ch == "l":        return "5"
        if ch in "mn":       return "6"
        if ch == "r":        return "7"
        return ""

    out: List[str] = []
    for i, ch in enumerate(s):
        prev = s[i - 1] if i > 0 else ""
        nxt  = s[i + 1] if i + 1 < len(s) else ""
        out.append(code_char(ch, prev, nxt))
    dedup: List[str] = []
    for c in out:
        if not dedup or c != dedup[-1]:
            dedup.append(c)
    return "".join(dedup)


# ── 8.4 Feature scoring ────────────────────────────────────────────────────

def _compute_feature_scores(
    name_q: str, ort_q: str, street_prefix_q: str,
    row: Dict[str, str],
    norm_cfg: Dict[str, Any],
) -> Dict[str, float]:
    name_r = normalize_text(row.get("firmenname", "") or "", norm_cfg)
    ort_r  = normalize_text(row.get("standort", "") or "", norm_cfg)
    addr_r = normalize_text(row.get("adresse", "") or "", norm_cfg)

    # name: trigram + levenshtein + token-set, phonetic boost
    name_char = max(sim_trigram(name_q, name_r), sim_levenshtein(name_q, name_r))
    name_sim  = max(name_char, sim_token_set(name_q, name_r))
    if name_q and name_r and phonetic_koelner(name_q) == phonetic_koelner(name_r):
        name_sim = max(name_sim, 0.95)

    # ort: trigram + levenshtein, phonetic boost
    ort_sim = max(sim_trigram(ort_q, ort_r), sim_levenshtein(ort_q, ort_r))
    if ort_q and ort_r and phonetic_koelner(ort_q) == phonetic_koelner(ort_r):
        ort_sim = max(ort_sim, 0.95)

    # addr: prefix match or trigram
    addr_sim = 0.0
    if street_prefix_q and addr_r:
        if addr_r.startswith(street_prefix_q):
            addr_sim = 0.8
        else:
            addr_sim = 0.6 * sim_trigram(street_prefix_q, addr_r)

    return {
        "name": float(name_sim),
        "ort":  float(ort_sim),
        "addr": float(addr_sim),
    }


# ── 8.5 Weighted score, coverage, conflict penalty ─────────────────────────

def _weighted_score(features: Dict[str, float], weights: Dict[str, float]) -> float:
    score = 0.0
    total_w = 0.0
    for k, v in features.items():
        w = float(weights.get(k, 0.0))
        score   += v * w
        total_w += w
    return (score / total_w) if total_w > 0 else 0.0


def _coverage_ok(features: Dict[str, float], cov_cfg: Dict[str, Any]) -> bool:
    positives = 0
    if features.get("name", 0.0) >= float(cov_cfg.get("name", 0.80)): positives += 1
    if features.get("ort",  0.0) >= float(cov_cfg.get("ort",  0.80)): positives += 1
    if features.get("addr", 0.0) >= float(cov_cfg.get("addr", 0.60)): positives += 1
    return positives >= int(cov_cfg.get("min_positives", 1))


def _conflict_penalty(features: Dict[str, float], pen_cfg: Dict[str, Any]) -> float:
    penalty = 0.0
    name_th = float(pen_cfg.get("name_threshold", 0.85))
    ort_th  = float(pen_cfg.get("ort_threshold",  0.60))
    pen     = float(pen_cfg.get("same_name_penalty", 0.05))
    # High name match but ort very low → slight penalty (ambiguous same-name companies)
    if features.get("name", 0.0) >= name_th and features.get("ort", 0.0) < ort_th:
        penalty += pen
    return penalty


# ── 8.6 Disambiguation helper ──────────────────────────────────────────────

def _choose_ask_next(
    order: List[str],
    provided: Dict[str, str],
    asked: List[str],
    features: Dict[str, float],
) -> str:
    already = set(asked or [])
    for feat in order:
        if feat in already:
            continue
        if feat == "ort" and not provided.get("ort"):
            return "ort"
        if feat == "strassen_prefix" and not provided.get("street_prefix"):
            return "strassen_prefix"
    # fallback: weakest remaining feature
    weakest = min(features.items(), key=lambda kv: kv[1])[0] if features else ""
    return {"addr": "strassen_prefix"}.get(weakest, "")


# ═══════════════════════════════════════════════════════════════════════════
# §10  Template & config helpers
# ═══════════════════════════════════════════════════════════════════════════

def _render_template(template: str, vars: Dict[str, str]) -> str:
    def repl(m):
        return vars.get((m.group(1) or "").strip(), m.group(0))
    return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", repl, template or "")


def _get_cfg(artifacts: SessionArtifacts, *keys: str, default: Any = None) -> Any:
    obj: Any = artifacts.config or {}
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return default
        if obj is None:
            return default
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# §9  Tools (@function_tool)
# ═══════════════════════════════════════════════════════════════════════════

# ── 9.1 verify_customer ────────────────────────────────────────────────────

@function_tool
async def verify_customer(
    context: RunContext,
    customer_id: str = "",
    caller_name: str = "",
    firmenname: str = "",
) -> Dict[str, Any]:
    """
    Verifiziert Kunden gegen customers.csv.
    Pfad 1: customer_id (exakt).
    Pfad 2: firmenname (exakt, case-insensitive).
    """
    artifacts: SessionArtifacts = context.userdata
    artifacts.verification_attempts += 1

    max_attempts = int(_get_cfg(artifacts, "customer_verification", "max_attempts", default=3) or 3)
    if artifacts.verification_attempts > max_attempts:
        return {"ok": False, "locked": True,
                "reason": f"Maximale Verifikationsversuche erreicht ({max_attempts})."}

    customer_id_in   = (customer_id or "").strip()
    firmenname_in    = (firmenname  or "").strip()

    # Path 1: exact match on customer_id
    if customer_id_in:
        row = next(
            (r for r in artifacts.customers
             if any((v or "").strip() == customer_id_in for v in r.values())),
            None,
        )
        if not row:
            return {"ok": False, "reason": "Kundennummer nicht gefunden."}

    # Path 2: exact case-insensitive match on firmenname column
    elif firmenname_in:
        row = next(
            (r for r in artifacts.customers
             if (r.get("firmenname") or "").strip().lower() == firmenname_in.lower()),
            None,
        )
        if not row:
            return {"ok": False, "reason": "Firmenname nicht gefunden."}

    else:
        return {"ok": False, "reason": "Bitte nennen Sie Ihre Kundennummer oder Ihren Firmennamen."}

    ask_name = bool(_get_cfg(artifacts, "customer_verification", "ask_caller_name", default=True))
    caller_name = (caller_name or "").strip()
    if ask_name and not caller_name:
        return {"ok": False, "reason": "Name des Anrufers fehlt."}

    vc = VerifiedCustomer(
        customer_id=row.get("customer_id") or "",
        firmenname=row.get("firmenname")   or "",
        standort=row.get("standort")       or "",
        adresse=row.get("adresse")         or "",
        country=row.get("country")         or "",
        caller_name=caller_name            or "",
    )
    artifacts.verified_customer = vc
    return {"ok": True, "customer_id": vc.customer_id, "firmenname": vc.firmenname,
            "standort": vc.standort, "caller_name": vc.caller_name}


# ── 9.2 search_customers ──────────────────────────────────────────────────

@function_tool
async def search_customers(
    context: RunContext,
    firmenname: str = "",
    standort: str = "",
    street_prefix: str = "",
) -> Dict[str, Any]:
    """
    Interne fuzzy Kandidatensuche OHNE Kandidatennennung nach außen.
    Sucht nur über CSV-Spalten: firmenname, standort, adresse.
    """
    artifacts: SessionArtifacts = context.userdata
    cv_cfg    = (artifacts.config or {}).get("customer_verification") or {}
    fuzzy_cfg = (cv_cfg.get("fuzzy") or {})

    thresholds = fuzzy_cfg.get("thresholds") or {"allow": 0.86, "ask": 0.72, "block": 0.50}
    weights    = fuzzy_cfg.get("weights")    or {"name": 0.60, "ort": 0.30, "addr": 0.10}
    norm_cfg   = fuzzy_cfg.get("normalization") or {"umlaute": "ae_oe_ue", "eszett": "ss", "punct": "drop"}
    cov_cfg    = fuzzy_cfg.get("coverage")   or {"name": 0.80, "ort": 0.80, "addr": 0.60, "min_positives": 1}
    pen_cfg    = fuzzy_cfg.get("conflict_penalty") or {}
    disamb     = fuzzy_cfg.get("disambiguation") or {"max_turns": 2, "order": ["ort", "strassen_prefix"]}
    max_k      = int(fuzzy_cfg.get("max_candidates_internal", 5) or 5)

    # Enforce max disambiguation turns — hard-block when limit reached
    max_turns = int(disamb.get("max_turns", 2))
    if len(artifacts.fz_asked) >= max_turns:
        return {
            "ok": True, "decision": "block", "score": 0.0,
            "coverage_ok": False, "best_candidate_customer_id": "",
            "ask_next": "", "features": {}, "privacy": {"enumerated": False},
        }

    name_q          = normalize_text(firmenname,    norm_cfg)
    ort_q           = normalize_text(standort,      norm_cfg)
    street_prefix_q = normalize_text(street_prefix, norm_cfg)

    provided = {"ort": ort_q, "street_prefix": street_prefix_q}

    scored: List[Tuple[str, float, Dict[str, float]]] = []
    for row in artifacts.customers or []:
        cid = (row.get("customer_id") or "").strip()
        if not cid:
            continue
        feats      = _compute_feature_scores(name_q, ort_q, street_prefix_q, row, norm_cfg)
        score_base = _weighted_score(feats, weights)
        penalty    = _conflict_penalty(feats, pen_cfg)
        scored.append((cid, max(0.0, score_base - penalty), feats))

    scored.sort(key=lambda t: t[1], reverse=True)
    shortlist = scored[:max_k] if max_k > 0 else scored
    best_cid, best_score, best_feats = (shortlist[0] if shortlist else ("", 0.0, {}))

    cov_ok = _coverage_ok(best_feats, cov_cfg) if shortlist else False

    allow_th = float(thresholds.get("allow", 0.86))
    ask_th   = float(thresholds.get("ask",   0.72))

    if best_score >= allow_th and cov_ok:
        decision = "allow"
        ask_next = ""
    elif best_score >= ask_th:
        decision = "ask"
        ask_next = _choose_ask_next(
            list(disamb.get("order") or ["ort", "strassen_prefix"]),
            provided, artifacts.fz_asked, best_feats,
        )
    else:
        decision = "block"
        ask_next = _choose_ask_next(
            list(disamb.get("order") or []),
            provided, artifacts.fz_asked, best_feats,
        )

    artifacts.fz_decision    = decision
    artifacts.fz_score       = float(best_score)
    artifacts.fz_coverage_ok = bool(cov_ok)
    artifacts.fz_features    = {k: float(best_feats.get(k, 0.0)) for k in ("name", "ort", "addr")}

    newly = []
    if ort_q:           newly.append("ort")
    if street_prefix_q: newly.append("strassen_prefix")
    seen = set(artifacts.fz_asked)
    for x in newly:
        if x not in seen:
            artifacts.fz_asked.append(x)
            seen.add(x)

    return {
        "ok": True,
        "decision": decision,
        "score": float(best_score),
        "coverage_ok": bool(cov_ok),
        "best_candidate_customer_id": best_cid,   # INTERNAL ONLY; never speak this
        "ask_next": ask_next,
        "features": artifacts.fz_features,
        "privacy": {"enumerated": False},
    }


# ── 9.3 submit_service_order ──────────────────────────────────────────────

@function_tool
async def submit_service_order(
    context: RunContext,
    problem: str,
    priority: str = "normal",
    contact_phone: str = "",
    preferred_time: str = "",
) -> Dict[str, Any]:
    """Speichert einen Serviceauftrag im Session-Context."""
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


# ═══════════════════════════════════════════════════════════════════════════
# §10  System prompt builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_system_prompt(cfg: Dict[str, Any]) -> str:
    agent_cfg = cfg.get("agent", {})
    cv_cfg    = cfg.get("customer_verification", {})

    base_prompt = (
        agent_cfg.get("system_prompt")
        or agent_cfg.get("instructions")
        or _default_config()["agent"]["system_prompt"]
    )

    ask_name     = bool(cv_cfg.get("ask_caller_name", True))
    max_attempts = int(cv_cfg.get("max_attempts", 3) or 3)

    runtime_rules_tmpl = (
        agent_cfg.get("runtime_rules_template")
        or _default_config()["agent"]["runtime_rules_template"]
    )
    runtime_rules = _render_template(runtime_rules_tmpl, {
        "ask_caller_name": "ja" if ask_name else "nein",
        "max_attempts": str(max_attempts),
    })

    fuzzy_enabled = bool((cv_cfg.get("fuzzy") or {}).get("enabled", True))
    fuzzy_rules   = ""
    if fuzzy_enabled:
        fuzzy_rules = (
            agent_cfg.get("fuzzy_rules")
            or _default_config()["agent"]["fuzzy_rules"]
        )

    instructions = "\n\n".join(filter(None, [base_prompt, runtime_rules, fuzzy_rules])).strip()
    return instructions


# ═══════════════════════════════════════════════════════════════════════════
# §14  Email notification (optional)
# ═══════════════════════════════════════════════════════════════════════════

def _send_email_notification(
    call_log: Dict[str, Any],
    cfg: Dict[str, Any],
    artifacts: SessionArtifacts,
) -> None:
    try:
        if not HAS_ACS_EMAIL:
            return
        email_cfg = cfg.get("email") or {}
        if not email_cfg.get("enabled", False):
            return

        conn_str = (
            _get_secret("AZURE_COMMUNICATION_CONNECTION_STRING")
            or os.environ.get("COMMUNICATION_CONNECTION_STRING_EMAIL")
            or ""
        )
        if not conn_str:
            logger.debug("No ACS email connection string — skipping email.")
            return

        sender = email_cfg.get("sender", "")
        recipients_raw = email_cfg.get("recipients") or []
        if not sender or not recipients_raw:
            logger.debug("Email sender or recipients missing — skipping.")
            return

        call_id = artifacts.call_id or "unknown"
        customer_id = ""
        room = call_id
        if artifacts.verified_customer:
            customer_id = artifacts.verified_customer.customer_id

        subject_tmpl = email_cfg.get("subject_template", "Service Call {{callId}}")
        subject = _render_template(subject_tmpl, {
            "callId": call_id,
            "room": room,
            "customerId": customer_id,
        })

        # Plain-text body
        lines = [
            f"Call ID: {call_id}",
            f"Caller: {artifacts.caller_number or 'unknown'}",
        ]
        if artifacts.verified_customer:
            vc = artifacts.verified_customer
            lines.append(f"Customer: {vc.firmenname} (ID: {vc.customer_id})")
            lines.append(f"Caller name: {vc.caller_name}")
        if artifacts.service_order:
            so = artifacts.service_order
            lines.append(f"Order: {so.order_id} — {so.problem} (priority: {so.priority})")
        body = "\n".join(lines)

        # Attachments
        call_log_b64 = base64.b64encode(
            json.dumps(call_log, ensure_ascii=False, indent=2).encode("utf-8")
        ).decode("ascii")
        config_b64 = base64.b64encode(
            json.dumps(cfg, ensure_ascii=False, indent=2).encode("utf-8")
        ).decode("ascii")

        to_list = [{"address": r, "displayName": r} for r in recipients_raw]

        message = {
            "senderAddress": sender,
            "recipients": {"to": to_list},
            "content": {"subject": subject, "plainText": body},
            "attachments": [
                {
                    "name": f"call_{call_id}.json",
                    "contentType": "application/json",
                    "contentInBase64": call_log_b64,
                },
                {
                    "name": f"config_{call_id}.json",
                    "contentType": "application/json",
                    "contentInBase64": config_b64,
                },
            ],
        }

        email_client = EmailClient.from_connection_string(conn_str)
        poller = email_client.begin_send(message)
        poller.result()
        logger.info("Email notification sent for call %s", call_id)

    except Exception:
        logger.exception("Failed to send email notification.")


# ═══════════════════════════════════════════════════════════════════════════
# §13/14  Shutdown / persist callback
# ═══════════════════════════════════════════════════════════════════════════

async def persist_and_notify(
    artifacts: SessionArtifacts,
    cfg: Dict[str, Any],
    ctx: JobContext,
) -> None:
    try:
        call_id = artifacts.call_id or "unknown"
        room_name = ctx.room.name if ctx.room else call_id

        vc_dict = None
        if artifacts.verified_customer:
            vc = artifacts.verified_customer
            vc_dict = {
                "customer_id": vc.customer_id,
                "firmenname": vc.firmenname,
                "standort": vc.standort,
                "adresse": vc.adresse,
                "country": vc.country,
                "caller_name": vc.caller_name,
            }

        so_dict = None
        if artifacts.service_order:
            so = artifacts.service_order
            so_dict = {
                "order_id": so.order_id,
                "problem": so.problem,
                "priority": so.priority,
                "contact_phone": so.contact_phone,
                "preferred_time": so.preferred_time,
                "timestamp_utc": so.timestamp_utc,
            }

        cv_cfg = (cfg.get("customer_verification") or {})
        fuzzy_cfg = cv_cfg.get("fuzzy") or {}
        fuzzy_enabled = bool(fuzzy_cfg.get("enabled", True))

        call_log: Dict[str, Any] = {
            "call_id": call_id,
            "room": room_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "caller": artifacts.caller_number or "",
            "config_meta": cfg.get("meta") or {},
            "verified_customer": vc_dict,
            "service_order": so_dict,
            "transcript": artifacts.transcript,
            "summary": artifacts.summary,
            "fuzzy_diagnostics": {
                "enabled": fuzzy_enabled,
                "decision": artifacts.fz_decision,
                "score": artifacts.fz_score,
                "coverage_ok": artifacts.fz_coverage_ok,
                "features": artifacts.fz_features,
                "asked_features": list(artifacts.fz_asked),
                "privacy": {
                    "no_candidate_enumeration": bool(
                        (fuzzy_cfg.get("privacy") or {}).get("no_candidate_enumeration", True)
                    ),
                },
            },
        }

        storage_cfg = (cfg.get("storage") or {})
        calls_prefix = storage_cfg.get("calls_prefix") or CALLS_PREFIX
        blob_path = f"{calls_prefix}{call_id}.json"
        upload_blob_text(blob_path, json.dumps(call_log, ensure_ascii=False, indent=2))
        logger.info("Call log uploaded: %s", blob_path)

        _send_email_notification(call_log, cfg, artifacts)

    except Exception:
        logger.exception("Error in persist_and_notify — swallowed to avoid crash.")


# ═══════════════════════════════════════════════════════════════════════════
# §12  Entrypoint
# ═══════════════════════════════════════════════════════════════════════════

async def entrypoint(ctx: JobContext) -> None:
    # 1. Connect to LiveKit room (audio only)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("Connected to room: %s", ctx.room.name)

    # 2. Wait for participant, extract caller number
    participant = await ctx.wait_for_participant()
    caller_number = ""
    sip_attrs = participant.attributes or {}
    for key in ("sip.phoneNumber", "caller.number", "sip.callerNumber"):
        val = sip_attrs.get(key, "")
        if val:
            caller_number = val
            break
    logger.info("Participant joined — caller: %s", caller_number or "unknown")

    # 3. Load blob config
    cfg = load_config()

    # Bootstrap Azure OpenAI credentials: KV cache → env vars → blob config → raise
    ao_cfg = cfg.get("azure_openai") or {}
    azure_endpoint = _get_secret("AZURE_OPENAI_ENDPOINT") or ao_cfg.get("endpoint") or ""
    azure_api_key  = _get_secret("AZURE_OPENAI_API_KEY")  or ao_cfg.get("api_key")  or ""
    azure_deploy   = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or ao_cfg.get("deployment") or "gpt-4o-mini"
    azure_api_ver  = os.environ.get("OPENAI_API_VERSION") or ao_cfg.get("api_version") or "2024-10-01-preview"

    if not azure_endpoint or not azure_api_key:
        raise RuntimeError(
            "Azure OpenAI credentials missing. Set AZURE_OPENAI_ENDPOINT and "
            "AZURE_OPENAI_API_KEY as env vars or in blob config azure_openai.*"
        )

    # Bootstrap Azure Speech credentials
    az_speech_cfg = cfg.get("azure_speech") or {}
    speech_cfg    = cfg.get("speech") or {}
    speech_key    = _get_secret("AZURE_SPEECH_KEY")    or az_speech_cfg.get("key")    or ""
    speech_region = _get_secret("AZURE_SPEECH_REGION") or az_speech_cfg.get("region") or ""
    speech_lang = (
        speech_cfg.get("stt_language")
        or speech_cfg.get("language")
        or "de-DE"
    )
    tts_voice     = speech_cfg.get("tts_voice") or "de-DE-ConradNeural"

    if not speech_key or not speech_region:
        raise RuntimeError(
            "Azure Speech credentials missing. Set AZURE-SPEECH-KEY and "
            "AZURE-SPEECH-REGION in Key Vault, or azure_speech.* in blob config."
        )

    # 4. Bootstrap LiveKit credentials (non-fatal)
    lk_cfg = cfg.get("livekit") or {}
    lk_url    = os.environ.get("LIVEKIT_URL")        or lk_cfg.get("url")        or ""
    lk_key    = _get_secret("LIVEKIT_API_KEY")    or lk_cfg.get("api_key")    or ""
    lk_secret = _get_secret("LIVEKIT_API_SECRET") or lk_cfg.get("api_secret") or ""
    if lk_url:
        os.environ.setdefault("LIVEKIT_URL", lk_url)
    if lk_key:
        os.environ.setdefault("LIVEKIT_API_KEY", lk_key)
    if lk_secret:
        os.environ.setdefault("LIVEKIT_API_SECRET", lk_secret)

    # 5. Load customers CSV
    storage_cfg = cfg.get("storage") or {}
    cust_blob = storage_cfg.get("customers_csv_blob") or CUSTOMERS_BLOB
    customers = load_customers(cust_blob)
    logger.info("Loaded %d customers from CSV.", len(customers))

    # 6. Derive call_id
    call_id = ""
    if hasattr(ctx, "job") and ctx.job:
        call_id = getattr(ctx.job, "id", "") or getattr(ctx.job, "dispatch_id", "") or ""
    if not call_id:
        call_id = ctx.room.name if ctx.room else "unknown"

    # 7. Create SessionArtifacts
    artifacts = SessionArtifacts(
        config=cfg,
        customers=customers,
        caller_number=caller_number,
        call_id=call_id,
    )

    # 8. Build STT + TTS (Azure Speech) + LLM (Azure OpenAI chat completions)
    llm_cfg     = cfg.get("llm") or {}
    temperature = float(llm_cfg.get("temperature", 0.2))

    # Build STT phrase hints from company names in customers CSV
    cv_cfg_ep   = cfg.get("customer_verification") or {}
    fuzzy_cfg_ep = (cv_cfg_ep.get("fuzzy") or {})
    stt_hints_cfg = fuzzy_cfg_ep.get("stt") or {}
    phrase_list: Optional[List[str]] = None
    if stt_hints_cfg.get("phrase_hints") == "from_csv":
        max_hints = int(stt_hints_cfg.get("phrase_hints_max", 500))
        phrase_list = [
            row["firmenname"] for row in customers
            if row.get("firmenname")
        ][:max_hints] or None
        logger.info("STT phrase hints: %d company names loaded.", len(phrase_list) if phrase_list else 0)

    stt = livekit_azure.STT(
        speech_key=speech_key,
        speech_region=speech_region,
        language=speech_lang,
        phrase_list=phrase_list,
    )
    tts = livekit_azure.TTS(
        speech_key=speech_key,
        speech_region=speech_region,
        voice=tts_voice,
    )
    llm = livekit_openai.LLM.with_azure(
        model=azure_deploy,
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=azure_api_ver,
        temperature=temperature,
    )

    # 9. Build system prompt
    instructions = _build_system_prompt(cfg)

    # 10. Create Agent
    cv_cfg = cfg.get("customer_verification") or {}
    fuzzy_cfg = (cv_cfg.get("fuzzy") or {})

    fuzzy_enabled = bool(fuzzy_cfg.get("enabled", True))
    tools = [verify_customer, submit_service_order]
    if fuzzy_enabled:
        tools.append(search_customers)

    agent = Agent(
        instructions=instructions,
        tools=tools,
    )

    # 11. Register shutdown callback
    async def _shutdown() -> None:
        await persist_and_notify(artifacts, cfg, ctx)

    ctx.add_shutdown_callback(_shutdown)

    # 12. Start AgentSession (with turn-detection params from config)
    turn_cfg = cfg.get("turn") or {}
    turn_enabled = bool(turn_cfg.get("enabled", True))
    min_ep = float(turn_cfg.get("min_endpointing_delay", 1.5))
    max_ep = float(turn_cfg.get("max_endpointing_delay", 20.0))

    session_kwargs: Dict[str, Any] = dict(userdata=artifacts, stt=stt, llm=llm, tts=tts)
    if turn_enabled:
        session_kwargs["min_endpointing_delay"] = min_ep
        session_kwargs["max_endpointing_delay"] = max_ep

    session = AgentSession(**session_kwargs)
    await session.start(agent=agent, room=ctx.room)

    # 13. Say welcome message
    agent_cfg = cfg.get("agent") or {}
    welcome = (
        agent_cfg.get("welcome_message")
        or _default_config()["agent"]["welcome_message"]
    )
    await session.say(welcome, allow_interruptions=False)


# ═══════════════════════════════════════════════════════════════════════════
# §16  CLI bootstrap
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=AGENT_NAME))
