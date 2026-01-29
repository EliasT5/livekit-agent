from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import logging
import os
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
            "welcome_message": "Willkommen beim Service. Bitte nennen Sie mir Ihre Kundennummer.",
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
# Customer verification + service order tools
# -----------------------------------------------------------------------------

def _get_cfg(artifacts: SessionArtifacts, *path: str, default: Any = None) -> Any:
    cur: Any = artifacts.config
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return default if cur is None else cur


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

    try:
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

    instructions = (str(base_prompt).strip() + runtime_rules).strip()

    agent = Agent(
        instructions=instructions,
        tools=[verify_customer, submit_service_order],
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

    welcome = (agent_cfg or {}).get(
        "welcome_message",
        _default_config()["agent"]["welcome_message"],
    )
    await session.say(str(welcome), allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(server)
