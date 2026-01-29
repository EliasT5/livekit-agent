import os
import json
import csv
import io
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from livekit.agents import (
    AgentSession,
    JobContext,
    RunContext,
)
from livekit.agents.tools import function_tool
from livekit.plugins import openai as openai_plugin
from livekit.plugins import azure as azure_speech
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from azure.storage.blob import BlobServiceClient

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phone-agent")

# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "assistant")
CONFIG_BLOB = os.getenv("CONFIG_BLOB", "config/latest.json")
CUSTOMERS_BLOB = os.getenv("CUSTOMERS_BLOB", "data/customers.csv")
CALLS_PREFIX = os.getenv("CALLS_PREFIX", "calls/")

# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------

@dataclass
class VerifiedCustomer:
    customer_id: str
    firmenname: str
    standort: str
    adresse: str
    country: str
    caller_name: str


@dataclass
class SessionArtifacts:
    customers: List[Dict[str, str]]
    verified_customer: Optional[VerifiedCustomer] = None
    transcript: List[Dict[str, str]] = None
    summary: Optional[str] = None
    caller_number: Optional[str] = None


# -------------------------------------------------------------------
# Azure Blob helpers
# -------------------------------------------------------------------

_blob_service: Optional[BlobServiceClient] = None


def _blob_svc() -> BlobServiceClient:
    global _blob_service
    if _blob_service is None:
        conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not conn:
            raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING missing")
        _blob_service = BlobServiceClient.from_connection_string(conn)
    return _blob_service


def _blob_client(path: str):
    return _blob_svc().get_blob_client(container=BLOB_CONTAINER, blob=path)


def load_json_blob(path: str) -> dict:
    data = _blob_client(path).download_blob().readall()
    return json.loads(data)


def load_csv_blob(path: str) -> List[Dict[str, str]]:
    raw = _blob_client(path).download_blob().readall().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(raw)))


def write_json_blob(path: str, payload: dict):
    _blob_client(path).upload_blob(
        json.dumps(payload, ensure_ascii=False, indent=2),
        overwrite=True,
    )


# -------------------------------------------------------------------
# Customer verification tool
# -------------------------------------------------------------------

@function_tool
async def verify_customer(
    context: RunContext,
    customer_id: str,
    caller_name: str,
    firmenname: str = "",
    standort: str = "",
) -> Dict[str, Any]:
    """
    Verifiziert einen Kunden anhand:
    - customer_id (Pflicht)
    - caller_name (Pflicht)
    - firmenname ODER standort (mind. eines muss matchen)
    """

    artifacts: SessionArtifacts = context.userdata
    row = next(
        (r for r in artifacts.customers if r.get("customer_id") == customer_id),
        None,
    )

    if not row:
        return {"ok": False, "reason": "Kundennummer nicht gefunden."}

    caller_name = (caller_name or "").strip()
    if not caller_name:
        return {"ok": False, "reason": "Name des Anrufers fehlt."}

    fn_ok = False
    st_ok = False

    if firmenname:
        fn_ok = row.get("firmenname", "").strip().lower() == firmenname.strip().lower()

    if standort:
        st_ok = row.get("standort", "").strip().lower() == standort.strip().lower()

    if not (fn_ok or st_ok):
        return {
            "ok": False,
            "reason": "Firmenname oder Standort stimmt nicht überein.",
        }

    vc = VerifiedCustomer(
        customer_id=row.get("customer_id", ""),
        firmenname=row.get("firmenname", ""),
        standort=row.get("standort", ""),
        adresse=row.get("adresse", ""),
        country=row.get("country", ""),
        caller_name=caller_name,
    )

    artifacts.verified_customer = vc

    return {
        "ok": True,
        "customer_id": vc.customer_id,
        "firmenname": vc.firmenname,
        "standort": vc.standort,
        "caller_name": vc.caller_name,
    }


# -------------------------------------------------------------------
# Agent entrypoint
# -------------------------------------------------------------------

async def entrypoint(job: JobContext):
    logger.info("Starting agent for job %s", job.id)

    # ---------------------------------------------------------------
    # Load config + customers (per call!)
    # ---------------------------------------------------------------

    config = load_json_blob(CONFIG_BLOB)
    customers = load_csv_blob(CUSTOMERS_BLOB)

    artifacts = SessionArtifacts(
        customers=customers,
        transcript=[],
        caller_number=job.room.name if job.room else None,
    )

    agent_cfg = config.get("agent", {})
    speech_cfg = config.get("speech", {})
    turn_cfg = config.get("turn", {})
    llm_cfg = config.get("llm", {})

    # ---------------------------------------------------------------
    # Turn detector (optional)
    # ---------------------------------------------------------------

    turn_detection = None
    preset = turn_cfg.get("preset", "off")
    if preset not in ("off", "disabled", "false"):
        try:
            turn_detection = MultilingualModel()
        except Exception as e:
            logger.warning("Turn detector disabled: %s", e)

    # ---------------------------------------------------------------
    # Agent session
    # ---------------------------------------------------------------

    session = AgentSession(
        userdata=artifacts,

        stt=azure_speech.STT(language=["de-AT", "de-DE"]),

        tts=azure_speech.TTS(
            voice=speech_cfg.get("tts_voice", "de-DE-KatjaNeural"),
            language=speech_cfg.get("tts_language", "de-DE"),
        ),

        turn_detection=turn_detection,
        min_endpointing_delay=turn_cfg.get("min_endpointing_delay", 1.2),
        max_endpointing_delay=turn_cfg.get("max_endpointing_delay", 10.0),

        llm=openai_plugin.LLM.with_azure(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            temperature=llm_cfg.get("temperature", 0.2),
        ),
    )

    # ---------------------------------------------------------------
    # Welcome
    # ---------------------------------------------------------------

    welcome = agent_cfg.get(
        "welcome_message",
        "Willkommen beim Service. Bitte nennen Sie mir Ihre Kundennummer.",
    )
    await session.say(welcome, allow_interruptions=False)

    # ---------------------------------------------------------------
    # Main conversation loop
    # ---------------------------------------------------------------

    try:
        async for event in session:
            if event.type == "user_message":
                artifacts.transcript.append(
                    {"role": "user", "text": event.text}
                )
            elif event.type == "agent_message":
                artifacts.transcript.append(
                    {"role": "agent", "text": event.text}
                )
    finally:
        # -----------------------------------------------------------
        # Call end → write log
        # -----------------------------------------------------------

        call_id = job.id
        log_path = f"{CALLS_PREFIX}{call_id}.json"

        payload = {
            "call_id": call_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "caller": artifacts.caller_number,
            "verified_customer": (
                asdict(artifacts.verified_customer)
                if artifacts.verified_customer
                else None
            ),
            "transcript": artifacts.transcript,
            "summary": artifacts.summary,
        }

        write_json_blob(log_path, payload)
        logger.info("Call log written to %s", log_path)
