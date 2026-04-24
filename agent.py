#!/usr/bin/env python3
"""
The Compound Transcripts Agent (WhisperX Edition)
Fetches the latest episode from five Compound / Bloomberg podcasts,
transcribes audio with WhisperX,
and emails a beautifully formatted transcript.

Podcasts covered:
  - Animal Spirits
  - The Compound and Friends
  - Ask the Compound
  - Masters in Business (Bloomberg)
  - At the Money (Barry Ritholtz — episodes in Masters in Business feed)

Readability enhancements:
  - Priority 1: LLM cleanup pass (punctuation, filler word removal)
  - Priority 2: VAD filter (strips silence/noise before transcription)
  - Priority 3: Larger paragraph gap (1.5s default) for more natural breaks
"""

import os
import re
import html
import json
import time
import calendar
import shutil
import smtplib
import logging
import fcntl
import requests
import feedparser
import threading
import gc
import psutil
from datetime import datetime
from dataclasses import dataclass
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from dotenv import load_dotenv

# ── Load .env early so all os.getenv() calls at module level pick up values ──
_ENV_FILE_EARLY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_ENV_FILE_EARLY)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_DIR = os.path.join(BASE_DIR, "settings")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
STATE_DIR = os.path.join(BASE_DIR, "state")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")
STATE_FILE = os.path.join(STATE_DIR, "state.json")
STATE_BACKUP = os.path.join(STATE_DIR, "state.backup.json")
LOG_FILE = os.path.join(LOGS_DIR, "agent.log")
RUN_LOCK_FILE = os.path.join(STATE_DIR, "agent.lock")
ENV_FILE = os.path.join(os.path.dirname(BASE_DIR), ".env")
PHRASES_AND_VOCAB_DIR = os.path.join(BASE_DIR, "phrases-and-vocabulary")
KNOWN_HOSTS_FILE = os.path.join(PHRASES_AND_VOCAB_DIR, "known-hosts-per-podcast.txt")
PODCAST_FEEDS_FILE = os.path.join(SETTINGS_DIR, "podcast_feeds.txt")
TRANSCRIPTION_SETTINGS_FILE = os.path.join(SETTINGS_DIR, "transcription-settings.txt")
LLM_CLEANUP_PROMPT_FILE = os.path.join(SETTINGS_DIR, "llm-cleanup-prompt.txt")
SERVICE_ENDPOINTS_FILE = os.path.join(SETTINGS_DIR, "service-endpoints.txt")
OPENING_CATCH_PHRASES_FILE = os.path.join(PHRASES_AND_VOCAB_DIR, "opening-catch-phrases.txt")
OPENING_AD_PHRASES_FILE = os.path.join(PHRASES_AND_VOCAB_DIR, "opening-ad-phrases.txt")
CLOSING_AD_PHRASES_FILE = os.path.join(PHRASES_AND_VOCAB_DIR, "closing-ad-phrases.txt")
VOCABULARY_CORRECTIONS_FILE = os.path.join(PHRASES_AND_VOCAB_DIR, "vocabulary-corrections.txt")
SHOW_DRAWINGS_FILE = os.path.join(SETTINGS_DIR, "show-drawings.txt")
PRODUCTION_CREW_DRAWING = os.path.join(BASE_DIR, "drawings", "caricature_drawings", "production_crew_drawing.png")

# ── Logging ───────────────────────────────────────────────────────────────────

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def ensure_required_modules_installed():
    required_modules = {
        "requests": "requests",
        "feedparser": "feedparser",
        "psutil": "psutil",
        "dotenv": "python-dotenv",
        "torch": "torch",
        "whisperx": "whisperx",
        "openai": "openai",
    }

    missing_packages = []
    for module_name, package_name in required_modules.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        unique_packages = sorted(set(missing_packages))
        packages_text = ", ".join(unique_packages)
        raise RuntimeError(
            "Missing required Python packages: "
            f"{packages_text}. "
            "Install them explicitly before running the agent, e.g. "
            "`python3 -m pip install -r requirements.txt` "
            "or use your preferred managed environment setup."
        )


ensure_required_modules_installed()


def load_key_value_settings(path):
    settings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()
    return settings


SETTINGS = load_key_value_settings(TRANSCRIPTION_SETTINGS_FILE)
SERVICE_ENDPOINTS = load_key_value_settings(SERVICE_ENDPOINTS_FILE)


def load_phrase_list(path):
    phrases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            phrases.append(line)
    return phrases


def load_vocabulary_corrections(path):
    corrections = []
    if not os.path.exists(path):
        return corrections

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw_line = line.rstrip("\n")
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            if "|" not in raw_line:
                log.warning(f"Skipping malformed vocabulary correction line {line_number}: {stripped_line}")
                continue

            canonical, variants_text = raw_line.split("|", 1)
            canonical = canonical.strip()
            variants = [variant.strip() for variant in variants_text.split(",") if variant.strip()]
            delete_match = variants_text.strip() == ""
            if not canonical:
                log.warning(f"Skipping incomplete vocabulary correction line {line_number}: {stripped_line}")
                continue
            if not variants and not delete_match:
                log.warning(f"Skipping incomplete vocabulary correction line {line_number}: {stripped_line}")
                continue

            corrections.append({
                "canonical": canonical,
                "variants": variants,
                "delete_match": delete_match,
            })
    return corrections


def apply_vocabulary_corrections(text, corrections):
    if not text or not corrections:
        return text, []

    updated_text = text
    replacements_applied = []
    ordered_pairs = []
    for entry in corrections:
        canonical = entry["canonical"]
        if entry.get("delete_match"):
            ordered_pairs.append((canonical, ""))
            continue
        for variant in entry["variants"]:
            ordered_pairs.append((variant, canonical))

    ordered_pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    for variant, canonical in ordered_pairs:
        pattern = re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)", re.IGNORECASE)
        updated_text, count = pattern.subn(canonical, updated_text)
        if count:
            replacements_applied.append({
                "from": variant,
                "to": canonical,
                "count": count,
            })

    updated_text = re.sub(r"[ \t]{2,}", " ", updated_text)
    updated_text = re.sub(r"\n{3,}", "\n\n", updated_text)

    return updated_text, replacements_applied


def remove_repeated_dotted_letter_runs(text):
    """Remove obvious transcript junk like long runs of repeated dotted single letters."""
    if not text:
        return text, []

    removals = []
    pattern = re.compile(r"(?<!\w)([A-Za-z])(?:\.\s*\1){7,}\.", re.IGNORECASE)

    def repl(match):
        original = match.group(0)
        letter = match.group(1)
        count = len(re.findall(rf"{re.escape(letter)}\.", original, re.IGNORECASE))
        removals.append({
            "letter": letter,
            "count": count,
            "sample": original[:80] + ("..." if len(original) > 80 else ""),
        })
        return ""

    cleaned = pattern.sub(repl, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, removals


def detect_obvious_raw_transcript_junk(text):
    """Detect obvious transcription junk severe enough to justify retranscription."""
    if not text:
        return None

    pattern = re.compile(r"(?<!\w)([A-Za-z])(?:\.\s*\1){7,}\.", re.IGNORECASE)
    findings = []
    for match in pattern.finditer(text):
        original = match.group(0)
        letter = match.group(1)
        count = len(re.findall(rf"{re.escape(letter)}\.", original, re.IGNORECASE))
        findings.append(f"{letter}. x{count}")
        if len(findings) >= 3:
            break

    if findings:
        return "repeated dotted-letter run(s): " + ", ".join(findings)

    return None


OPENING_CATCH_PHRASES = load_phrase_list(OPENING_CATCH_PHRASES_FILE)
OPENING_AD_PHRASES = load_phrase_list(OPENING_AD_PHRASES_FILE)
CLOSING_AD_PHRASES = load_phrase_list(CLOSING_AD_PHRASES_FILE)
VOCABULARY_CORRECTIONS = load_vocabulary_corrections(VOCABULARY_CORRECTIONS_FILE)

# Episode duration filter (skip trailers/promos)
MIN_EPISODE_DURATION_SECS = int(SETTINGS["MIN_EPISODE_DURATION_SECS"])  # 5 minutes

# ── Podcast feeds ────────────────────────────────────────────────────────────
def load_podcast_feeds(path):
    podcasts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 2)
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            rss_url = parts[1].strip()
            alias = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
            podcasts.append((name, rss_url, alias))
    return podcasts


@dataclass
class RuntimeConfig:
    settings: dict
    service_endpoints: dict
    opening_catch_phrases: list
    opening_ad_phrases: list
    closing_ad_phrases: list
    vocabulary_corrections: list
    podcasts: list
    min_episode_duration_secs: int
    whisper_model: str
    whisper_batch_size: int
    whisper_compute_type: str
    whisperx_device: str
    paragraph_gap_secs: float
    max_episodes_per_run: int
    test_mode: bool
    test_mode_episode_count: int
    test_mode_month: str
    test_mode_month_number: int | None
    test_mode_skip_title_contains: str
    audio_retention_hours: int
    safe_title_max_len: int
    opening_ad_max_start_seconds: int
    podcast_start_early_char_limit: int
    download_retries: int
    download_timeout_seconds: int
    download_retry_backoff_base_seconds: int
    email_retry_delays_seconds: list[int]
    llm_request_timeout_seconds: int
    llm_provider: str | None
    llm_model: str | None
    llm_temperature: float
    force_reprocess: bool
    llm_cleanup_prompt: str
    show_drawings: dict
    smtp_host: str
    smtp_port: int


TEST_MODE_MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def parse_test_mode_month(value):
    normalized = re.sub(r"[^a-z]", "", (value or "").lower())
    if not normalized:
        return None

    month_number = TEST_MODE_MONTH_LOOKUP.get(normalized)
    if month_number is None:
        valid_values = ", ".join(calendar.month_abbr[i] for i in range(1, 13))
        raise ValueError(
            f"Invalid TEST_MODE_MONTH '{value}'. Use a month name or abbreviation like: {valid_values}."
        )
    return month_number


def get_most_recent_month_window(month_number, now=None):
    now = now or datetime.now()
    year = now.year if month_number <= now.month else now.year - 1
    start = datetime(year, month_number, 1)
    if month_number == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month_number + 1, 1)
    return start, end


def parse_int_csv(value, default=None):
    items = []
    for raw in (value or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        items.append(int(raw))
    return items if items else list(default or [])


def reset_test_mode_settings():
    if not os.path.exists(TRANSCRIPTION_SETTINGS_FILE):
        return

    updates = {
        "TEST_MODE": "false",
        "TEST_MODE_EPISODE_COUNT": "1",
        "TEST_MODE_MONTH": "",
    }

    with open(TRANSCRIPTION_SETTINGS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    seen = set()
    updated_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated_lines.append(line)
            continue
        key, _ = line.split("=", 1)
        key = key.strip()
        if key in updates:
            updated_lines.append(f"{key}={updates[key]}\n")
            seen.add(key)
        else:
            updated_lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            updated_lines.append(f"{key}={value}\n")

    with open(TRANSCRIPTION_SETTINGS_FILE, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    log.info("Reset test-mode settings to defaults: TEST_MODE=false, TEST_MODE_EPISODE_COUNT=1, TEST_MODE_MONTH=")


def load_runtime_config():
    settings = load_key_value_settings(TRANSCRIPTION_SETTINGS_FILE)
    service_endpoints = load_key_value_settings(SERVICE_ENDPOINTS_FILE)
    raw_test_mode_month = settings.get("TEST_MODE_MONTH", "").strip()
    test_mode_month_number = parse_test_mode_month(raw_test_mode_month) if raw_test_mode_month else None
    raw_test_mode_skip_title_contains = os.getenv(
        "TEST_MODE_SKIP_TITLE_CONTAINS",
        settings.get("TEST_MODE_SKIP_TITLE_CONTAINS", "")
    ).strip()

    with open(LLM_CLEANUP_PROMPT_FILE, "r", encoding="utf-8") as f:
        llm_cleanup_prompt = f.read()

    return RuntimeConfig(
        settings=settings,
        service_endpoints=service_endpoints,
        opening_catch_phrases=load_phrase_list(OPENING_CATCH_PHRASES_FILE),
        opening_ad_phrases=load_phrase_list(OPENING_AD_PHRASES_FILE),
        closing_ad_phrases=load_phrase_list(CLOSING_AD_PHRASES_FILE),
        vocabulary_corrections=load_vocabulary_corrections(VOCABULARY_CORRECTIONS_FILE),
        podcasts=load_podcast_feeds(PODCAST_FEEDS_FILE),
        min_episode_duration_secs=int(settings["MIN_EPISODE_DURATION_SECS"]),
        whisper_model=settings["WHISPER_MODEL"],
        whisper_batch_size=int(settings["WHISPER_BATCH_SIZE"]),
        whisper_compute_type=settings["WHISPER_COMPUTE_TYPE"],
        whisperx_device=settings["WHISPERX_DEVICE"],
        paragraph_gap_secs=float(settings["PARAGRAPH_GAP_SECS"]),
        max_episodes_per_run=int(settings["MAX_EPISODES_PER_RUN"]),
        test_mode=settings.get("TEST_MODE", "false").lower() == "true",
        test_mode_episode_count=int(settings.get("TEST_MODE_EPISODE_COUNT", "1")),
        test_mode_month=raw_test_mode_month,
        test_mode_month_number=test_mode_month_number,
        test_mode_skip_title_contains=raw_test_mode_skip_title_contains,
        audio_retention_hours=int(settings["AUDIO_RETENTION_HOURS"]),
        safe_title_max_len=int(settings["SAFE_TITLE_MAX_LEN"]),
        opening_ad_max_start_seconds=int(settings["OPENING_AD_MAX_START_SECONDS"]),
        podcast_start_early_char_limit=int(settings["PODCAST_START_EARLY_CHAR_LIMIT"]),
        download_retries=int(settings["DOWNLOAD_RETRIES"]),
        download_timeout_seconds=int(settings["DOWNLOAD_TIMEOUT_SECONDS"]),
        download_retry_backoff_base_seconds=int(settings["DOWNLOAD_RETRY_BACKOFF_BASE_SECONDS"]),
        email_retry_delays_seconds=parse_int_csv(settings.get("EMAIL_RETRY_DELAYS_SECONDS", "5,15"), default=[5, 15]),
        llm_request_timeout_seconds=int(settings.get("LLM_REQUEST_TIMEOUT_SECONDS", "120")),
        llm_provider=os.getenv("LLM_PROVIDER"),
        llm_model=os.getenv("LLM_MODEL"),
        llm_temperature=float(settings["LLM_TEMPERATURE"]),
        force_reprocess=os.getenv("FORCE_REPROCESS", "false").lower() == "true",
        llm_cleanup_prompt=llm_cleanup_prompt,
        show_drawings=load_show_drawings(SHOW_DRAWINGS_FILE),
        smtp_host=service_endpoints.get("SMTP_HOST", "smtp.gmail.com"),
        smtp_port=int(service_endpoints.get("SMTP_PORT", "465")),
    )


# ── Resource Monitoring ───────────────────────────────────────────────────────

def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def log_resources(phase_name):
    """Log current resource usage."""
    mem_mb = get_memory_usage_mb()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    log.info(f"  📊 [{phase_name}] Memory: {mem_mb:.1f} MB | CPU: {cpu_percent:.1f}%")


def acquire_run_lock():
    """Prevent overlapping agent runs from starting concurrently."""
    os.makedirs(STATE_DIR, exist_ok=True)
    handle = open(RUN_LOCK_FILE, "a+", encoding="utf-8")

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        log.warning(
            "Another agent run is already active; exiting to avoid overlap. "
            f"Lock file: {RUN_LOCK_FILE} | owner: {owner}"
        )
        handle.close()
        return None

    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "started_at": datetime.now().isoformat()}) + "\n")
    handle.flush()
    return handle


def release_run_lock(handle):
    if not handle:
        return
    try:
        handle.seek(0)
        handle.truncate()
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
        try:
            if os.path.exists(RUN_LOCK_FILE):
                os.remove(RUN_LOCK_FILE)
        except OSError:
            pass

# ── Host Config ──────────────────────────────────────────────────────────────

def load_show_drawings(path):
    """Load show-to-drawing mapping from a simple editable text file."""
    drawings = {}
    if not os.path.exists(path):
        return drawings

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            show_name, drawing_path = line.split("|", 1)
            drawings[show_name.strip()] = drawing_path.strip()

    return drawings


def resolve_show_drawing(podcast_name, config):
    """Resolve show-specific drawing path, with optional production-crew fallback."""
    relative_path = config.show_drawings.get(podcast_name) or config.show_drawings.get("production-crew")
    if not relative_path:
        return None

    absolute_path = os.path.join(BASE_DIR, relative_path)
    if os.path.exists(absolute_path):
        return relative_path

    log.warning(f"Configured drawing not found for '{podcast_name}': {relative_path}")
    return None


def load_known_hosts(path):
    """Load known hosts per podcast from a simple editable text file."""
    hosts_by_podcast = {}
    if not os.path.exists(path):
        return hosts_by_podcast

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            podcast_name, hosts_text = line.split("|", 1)
            hosts = [host.strip() for host in hosts_text.split(",") if host.strip()]
            hosts_by_podcast[podcast_name.strip()] = hosts

    return hosts_by_podcast


CONFIG = load_runtime_config()


def get_podcast_hosts(podcast_name):
    """Get list of known hosts for a podcast."""
    known_hosts = load_known_hosts(KNOWN_HOSTS_FILE)
    return known_hosts.get(podcast_name, [])


def extract_guest_names_from_text(text, known_hosts):
    """
    Extract potential guest names from show notes, title, or transcript intro.
    
    Patterns checked:
    - "with [Name]" / "featuring [Name]"
    - "[Name] joins" / "joining us is [Name]"
    - "guest [Name]" / "special guest [Name]"
    - "[Name] on [topic]" (common title format)
    
    Args:
        text: Combined text from title, summary, and/or transcript intro
        known_hosts: List of host names to exclude from guest detection
    
    Returns:
        list of potential guest names (may be empty)
    """
    if not text:
        return []
    
    # Normalize host names for comparison (lowercase)
    known_hosts_lower = [h.lower() for h in known_hosts]
    
    # Common words that appear capitalized but aren't names
    non_name_words = {
        'with', 'the', 'this', 'that', 'and', 'for', 'from', 'about',
        'interview', 'featuring', 'special', 'guest', 'episode', 'part',
        'how', 'why', 'what', 'when', 'where', 'who', 'which',
        'talking', 'discusses', 'joins', 'returns', 'on', 'in', 'at'
    }
    
    potential_guests = []
    
    # Patterns to find guest names
    # Strategy: Find trigger phrases, then extract the properly capitalized name that follows
    # Names must be 2-4 words where each word starts with uppercase
    
    # Trigger patterns (case-insensitive) followed by name extraction
    trigger_patterns = [
        # "with John Smith" or "featuring Jane Doe"
        (r'(?:with|featuring|feat\.?)\s+', 'after'),
        # "John Smith joins"
        (r'\s+(?:joins|join)\b', 'before'),
        # "joining us is Jane Doe" / "joining us today is Jane Doe"
        (r'joining\s+(?:us|me)\s+(?:is|today\s+is|this\s+week\s+is)\s+', 'after'),
        # "guest John Smith" or "special guest Jane Doe"
        (r'(?:special\s+)?guest\s+', 'after'),
        # "John Smith on Markets" (at start of text)
        (r'^', 'after_on'),  # Special: name before " on "
        # "interview with John Smith"
        (r'interview\s+with\s+', 'after'),
    ]
    
    # Pattern for a properly capitalized name (2-4 words)
    # Supports: John Smith, Mary Jane Watson, Patrick O'Shaughnessy, Jean-Pierre Dupont
    # Word pattern: Uppercase letter, optional apostrophe+uppercase (O'Brien), lowercase letters, optional hyphen continuation
    WORD = r"[A-Z][a-z]*(?:'[A-Z][a-z]+)?(?:-[A-Z][a-z]+)?"
    name_re = re.compile(rf'({WORD}(?:\s+{WORD}){{1,3}})')
    
    def is_valid_name(name):
        """Check if extracted text looks like a real name."""
        if not name:
            return False
        words = name.split()
        if len(words) < 2 or len(words) > 4:
            return False
        # First word shouldn't be a common non-name word
        if words[0].lower() in non_name_words:
            return False
        # All words should be at least 2 chars (filters 'A', 'I', etc.)
        if any(len(w.replace("'", "").replace("-", "")) < 2 for w in words):
            return False
        return True
    
    for trigger, direction in trigger_patterns:
        if direction == 'after':
            # Find trigger, then extract name after it
            trigger_re = re.compile(trigger, re.IGNORECASE)
            for m in trigger_re.finditer(text):
                remainder = text[m.end():]
                name_match = name_re.match(remainder)
                if name_match:
                    name = name_match.group(1).strip()
                    if name.lower() not in known_hosts_lower and is_valid_name(name):
                        if name not in potential_guests:
                            potential_guests.append(name)
        
        elif direction == 'before':
            # Find trigger, then extract name before it
            trigger_re = re.compile(trigger, re.IGNORECASE)
            for m in trigger_re.finditer(text):
                prefix = text[:m.start()]
                # Find the last capitalized name sequence before trigger
                all_names = name_re.findall(prefix)
                if all_names:
                    name = all_names[-1].strip()
                    if name.lower() not in known_hosts_lower and is_valid_name(name):
                        if name not in potential_guests:
                            potential_guests.append(name)
        
        elif direction == 'after_on':
            # Special case: "Name on Topic" at start
            # Use the same WORD pattern for consistency
            on_pattern = rf'^({WORD}(?:\s+{WORD}){{1,3}})\s+on\s+'
            on_match = re.match(on_pattern, text)
            if on_match:
                name = on_match.group(1).strip()
                if name.lower() not in known_hosts_lower and is_valid_name(name):
                    if name not in potential_guests:
                        potential_guests.append(name)
    
    return potential_guests


def extract_guest_from_transcript_intro(segments, known_hosts, max_segments=20):
    """
    Look for guest introductions in the first few segments of transcript.
    
    Hosts often say things like:
    - "Today we have [Name] joining us"
    - "Our guest is [Name]"
    - "Please welcome [Name]"
    
    Args:
        segments: List of transcript segments
        known_hosts: List of host names to exclude
        max_segments: How many segments to check (default 20 = ~first 2-3 minutes)
    
    Returns:
        list of potential guest names
    """
    if not segments:
        return []
    
    # Combine first N segments into one text block
    intro_text = ' '.join(
        seg.get('text', '') for seg in segments[:max_segments]
    )
    
    return extract_guest_names_from_text(intro_text, known_hosts)


def detect_hosts_from_intro_phrases(segments, hosts, max_segments=10):
    """
    Detect which speakers are hosts by looking for self-identification phrases,
    weighted toward segments that look like the actual podcast opening.
    Returns ALL detected host-speaker mappings, not just the first.
    """
    if not segments or not hosts:
        return {}

    hosts_lower = {h.lower(): h for h in hosts}
    detected_hosts = {}

    intro_patterns = [
        r"(?:welcome\s+to\s+[^,]+,?\s+)?I'm\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:this\s+is|I\s+am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+here\b",
        r"(?:hey|hi|hello)\s+(?:everyone|folks|guys)[,.]?\s+(?:I'm\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:and\s+)?I'm\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ]

    opening_start_index = 0
    for i, seg in enumerate(segments[:max_segments]):
        text = seg.get("text", "")
        text_lower = text.lower()
        if any(phrase.lower() in text_lower for phrase in OPENING_CATCH_PHRASES):
            opening_start_index = i
            log.info(f"  Found opening catch phrase in segment {i + 1}: {text[:80]}")
            break

    for seg in segments[opening_start_index:max_segments]:
        speaker = seg.get("speaker")
        text = seg.get("text", "")
        if not speaker or not text or speaker in detected_hosts:
            continue

        for pattern in intro_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue

            detected_name = match.group(1).strip()
            for host_lower, host_proper in hosts_lower.items():
                if (
                    detected_name.lower() == host_lower
                    or detected_name.lower() == host_lower.split()[0].lower()
                ):
                    if host_proper not in detected_hosts.values():
                        log.info(f"  Intro detection: '{detected_name}' -> {host_proper} (speaker {speaker})")
                        detected_hosts[speaker] = host_proper
                    break

    return detected_hosts


def detect_host_from_intro_phrases(segments, hosts, max_segments=10):
    """
    Legacy wrapper - returns first detected host for backwards compatibility.
    """
    detected = detect_hosts_from_intro_phrases(segments, hosts, max_segments)
    if detected:
        first_speaker = next(iter(detected))
        return first_speaker, detected[first_speaker]
    return None, None


def strip_opening_ad_segments(segments, ad_phrases, max_start_seconds):
    """Strip likely ad segments near the start of an episode."""
    if not segments or not ad_phrases:
        return segments

    filtered_segments = []
    removed_count = 0
    skipping = True

    for seg in segments:
        text = seg.get("text", "")
        start = seg.get("start", 0) or 0
        text_lower = text.lower()
        matches_ad = any(phrase.lower() in text_lower for phrase in ad_phrases)

        if skipping and start <= max_start_seconds and matches_ad:
            removed_count += 1
            continue

        skipping = False
        filtered_segments.append(seg)

    if removed_count:
        log.info(f"  Stripped {removed_count} likely opening ad segment(s)")

    return filtered_segments


def get_first_speaker(segments, min_words=10):
    """
    Get the first speaker who says more than min_words.
    Skips very short utterances that might be pre-roll or ads.
    
    Args:
        segments: List of transcript segments
        min_words: Minimum words to count as "real" speaking
    
    Returns:
        speaker_label or None
    """
    word_count_by_speaker = {}
    first_substantial_speaker = None
    
    for seg in segments:
        speaker = seg.get("speaker")
        text = seg.get("text", "")
        if not speaker:
            continue
        
        words = len(text.split())
        word_count_by_speaker[speaker] = word_count_by_speaker.get(speaker, 0) + words
        
        # First speaker to cross the threshold
        if first_substantial_speaker is None and word_count_by_speaker[speaker] >= min_words:
            first_substantial_speaker = speaker
            break
    
    return first_substantial_speaker


def map_speakers_to_names(segments, podcast_name, episode_title="", episode_summary=""):
    """
    Map generic SPEAKER_XX labels to actual host names and extracted guest names.
    
    Strategy (multi-signal approach):
    1. Intro Detection: Look for self-identification phrases ("I'm Ben Carlson")
    2. First-Speaker Heuristic: First substantial speaker is likely the host
    3. Word Count: Most talkative speaker is likely primary host (fallback)
    4. For non-hosts: Extract guest names from title/summary/transcript
    5. Fall back to 'Guest 1', 'Guest 2' if extraction fails
    
    Scoring:
    - Intro phrase detected: +5 points
    - First speaker: +2 points  
    - Highest word count: +1 point
    
    Args:
        segments: List of transcript segments with 'speaker' and 'text' keys
        podcast_name: Name of the podcast to look up hosts
        episode_title: Episode title (for guest name extraction)
        episode_summary: Episode summary/show notes (for guest name extraction)
    
    Returns:
        dict mapping SPEAKER_XX -> actual name
    """
    hosts = get_podcast_hosts(podcast_name)
    if not hosts:
        return {}
    
    # Count words per speaker
    speaker_word_counts = {}
    for seg in segments:
        speaker = seg.get("speaker")
        text = seg.get("text", "")
        if speaker and text:
            word_count = len(text.split())
            speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + word_count
    
    if not speaker_word_counts:
        return {}
    
    all_speakers = list(speaker_word_counts.keys())
    
    # === Signal 1: Intro phrase detection (may find multiple hosts) ===
    detected_hosts = detect_hosts_from_intro_phrases(segments, hosts)
    
    # === Signal 2: First speaker heuristic ===
    first_speaker = get_first_speaker(segments, min_words=10)
    
    # === Signal 3: Word count ranking ===
    speakers_by_words = sorted(all_speakers, key=lambda s: speaker_word_counts[s], reverse=True)
    
    # === Scoring: Determine speaker order ===
    speaker_scores = {s: 0 for s in all_speakers}
    
    # Intro detection: +5 points per detected host
    for speaker in detected_hosts:
        speaker_scores[speaker] += 5
        log.info(f"  Scoring: {speaker} +5 (intro phrase: {detected_hosts[speaker]})")
    
    intro_speaker = next(iter(detected_hosts), None) if detected_hosts else None
    
    # First speaker: +2 points
    if first_speaker:
        speaker_scores[first_speaker] += 2
        log.info(f"  Scoring: {first_speaker} +2 (first substantial speaker)")
    
    # Highest word count: +1 point
    if speakers_by_words:
        speaker_scores[speakers_by_words[0]] += 1
        log.info(f"  Scoring: {speakers_by_words[0]} +1 (most words: {speaker_word_counts[speakers_by_words[0]]})")
    
    # Sort speakers by score (descending), then by word count as tiebreaker
    sorted_speakers = sorted(
        all_speakers,
        key=lambda s: (speaker_scores[s], speaker_word_counts[s]),
        reverse=True
    )
    
    log.info(f"  Speaker scores: {dict(sorted(speaker_scores.items(), key=lambda x: x[1], reverse=True))}")
    log.info(f"  Final speaker order: {sorted_speakers}")
    
    # === Extract guest names from metadata and transcript ===
    # Process title and summary SEPARATELY to avoid concatenation artifacts
    guest_names = []
    
    if episode_title:
        guest_names.extend(extract_guest_names_from_text(episode_title, hosts))
    if episode_summary:
        guest_names.extend(extract_guest_names_from_text(episode_summary, hosts))
    guest_names.extend(extract_guest_from_transcript_intro(segments, hosts))
    
    # Deduplicate while preserving order, preferring shorter/cleaner names
    # Also filter out names that contain other names (e.g., "Factor Investing Cliff Asness" vs "Cliff Asness")
    seen_normalized = set()
    unique_guests = []
    
    # Sort by length (shorter first) to prefer cleaner extractions
    guest_names_sorted = sorted(guest_names, key=len)
    
    for name in guest_names_sorted:
        name_lower = name.lower()
        
        # Skip if we've seen this exact name
        if name_lower in seen_normalized:
            continue
        
        # Skip if this name contains or is contained by an already-seen name
        is_duplicate = False
        for seen_name in seen_normalized:
            if seen_name in name_lower or name_lower in seen_name:
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_normalized.add(name_lower)
            unique_guests.append(name)
    guest_names = unique_guests
    
    if guest_names:
        log.info(f"  Extracted guest names: {guest_names}")
    
    # === Build final mapping ===
    speaker_name_map = {}
    guest_index = 0
    fallback_guest_count = 0
    
    # First, apply all detected host intros (highest confidence)
    if detected_hosts:
        for speaker, host in detected_hosts.items():
            speaker_name_map[speaker] = host
        
        # Remaining hosts and speakers
        assigned_hosts = set(detected_hosts.values())
        remaining_hosts = [h for h in hosts if h not in assigned_hosts]
        remaining_speakers = [s for s in sorted_speakers if s not in detected_hosts]
        
        for i, speaker_label in enumerate(remaining_speakers):
            if i < len(remaining_hosts):
                speaker_name_map[speaker_label] = remaining_hosts[i]
            else:
                # Guest assignment
                if guest_index < len(guest_names):
                    speaker_name_map[speaker_label] = guest_names[guest_index]
                    guest_index += 1
                else:
                    fallback_guest_count += 1
                    speaker_name_map[speaker_label] = f"Guest {fallback_guest_count}"
    else:
        # No intro detection - use score-based ordering
        for i, speaker_label in enumerate(sorted_speakers):
            if i < len(hosts):
                speaker_name_map[speaker_label] = hosts[i]
            else:
                if guest_index < len(guest_names):
                    speaker_name_map[speaker_label] = guest_names[guest_index]
                    guest_index += 1
                else:
                    fallback_guest_count += 1
                    speaker_name_map[speaker_label] = f"Guest {fallback_guest_count}"
    
    log.info(f"  Speaker mapping: {speaker_name_map}")
    return speaker_name_map


HONORIFICS = {
    "mr", "mrs", "ms", "miss", "dr", "doctor", "prof", "professor",
    "president", "chairman", "chairwoman", "chair", "sen", "senator",
    "rep", "representative", "gov", "governor", "mayor", "judge",
    "justice", "sir", "madam", "dame"
}

PERSON_SUFFIXES = {
    "jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"
}

NON_PERSON_TERMS = {
    "capital", "partners", "management", "investments", "investment", "advisors",
    "advisor", "research", "media", "bank", "university", "institute", "markets",
    "market", "index", "indexes", "treasury", "bloomberg", "fidelity", "nasdaq",
    "dow", "fund", "funds", "holdings", "ventures", "group", "corp", "corporation",
    "inc", "llc", "lp", "ltd", "committee", "council", "department", "bureau",
    "street", "avenue", "road", "county", "state", "city", "country", "podcast",
    "episode", "show", "minute", "minutes", "weighted"
}

NON_PERSON_PHRASES = {
    "new york", "united states", "wall street", "the compound", "animal spirits",
    "at the money", "masters in business", "ask the compound", "s&p", "janus henderson"
}


def normalize_candidate_name(name):
    if not name:
        return ""
    cleaned = re.sub(r"\s+", " ", name).strip().rstrip(".")
    return cleaned


def tokenize_person_name(name):
    return [token for token in re.split(r"\s+", name.strip()) if token]


def is_honorific(token):
    return token.lower().rstrip(".") in HONORIFICS


def is_person_suffix(token):
    return token.lower() in PERSON_SUFFIXES


def is_valid_person_word(token):
    stripped = token.strip(".,")
    if not stripped:
        return False
    if re.search(r"\d", stripped):
        return False
    return bool(re.fullmatch(r"[A-Za-z]+(?:[\.'’-][A-Za-z]+)*", stripped))


def looks_like_person_name(name):
    normalized = normalize_candidate_name(name)
    if not normalized:
        return False

    lowered = normalized.lower()
    if lowered in NON_PERSON_PHRASES:
        return False

    tokens = tokenize_person_name(normalized)
    if not tokens:
        return False

    core_tokens = tokens[:]
    if core_tokens and is_honorific(core_tokens[0]):
        core_tokens = core_tokens[1:]

    if not core_tokens:
        return False

    if len(core_tokens) > 4:
        return False

    for token in core_tokens:
        lower_token = token.lower().rstrip(".")
        if lower_token in NON_PERSON_TERMS:
            return False

    for token in core_tokens[:-1]:
        if is_person_suffix(token):
            return False

    last_token = core_tokens[-1]
    name_tokens = core_tokens[:-1] if is_person_suffix(last_token) else core_tokens

    if not name_tokens:
        return False

    for token in name_tokens:
        if not is_valid_person_word(token):
            return False

    return True


def infer_speaker_list(podcast_name, episode_title="", episode_summary="", transcript_text=""):
    """Infer a best-effort speaker list from hosts plus guest names in show notes/title/transcript intro."""
    speakers = []
    seen = set()

    def add_name(name, validate=True):
        normalized = normalize_candidate_name(name)
        if not normalized:
            return
        if validate and not looks_like_person_name(normalized):
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        speakers.append(normalized)

    hosts = get_podcast_hosts(podcast_name) if podcast_name else []
    for host in hosts:
        add_name(host, validate=False)

    guest_candidates = []
    if episode_summary:
        guest_candidates.extend(extract_guest_names_from_text(episode_summary, hosts))
    if episode_title:
        guest_candidates.extend(extract_guest_names_from_text(episode_title, hosts))
    if transcript_text:
        intro_text = " ".join(transcript_text.split()[:400])
        guest_candidates.extend(extract_guest_names_from_text(intro_text, hosts))

    unique_candidates = []
    seen_candidates = set()
    for name in sorted(guest_candidates, key=len):
        cleaned = normalize_candidate_name(name)
        lowered = cleaned.lower()
        if not cleaned or lowered in seen_candidates:
            continue
        duplicate = False
        for existing in seen_candidates:
            if lowered in existing or existing in lowered:
                duplicate = True
                break
        if duplicate:
            continue
        seen_candidates.add(lowered)
        unique_candidates.append(cleaned)

    for guest in unique_candidates:
        add_name(guest, validate=True)

    return speakers


def format_speakers_line(speakers):
    """Format speakers for the transcript header on a single line."""
    if not speakers:
        return None
    if len(speakers) == 1:
        names_text = speakers[0]
    elif len(speakers) == 2:
        names_text = f"{speakers[0]} and {speakers[1]}"
    else:
        names_text = ", ".join(speakers[:-1]) + f", and {speakers[-1]}"
    return f"**Speakers:** {names_text}."


def parse_duration(duration_str):
    """
    Parse iTunes duration string to seconds.
    
    Handles formats:
    - "3600" (raw seconds)
    - "60:00" (MM:SS)
    - "1:00:00" (HH:MM:SS)
    
    Returns:
        int seconds, or None if parsing fails
    """
    if not duration_str:
        return None
    
    duration_str = str(duration_str).strip()
    
    # Try raw seconds first
    if duration_str.isdigit():
        return int(duration_str)
    
    # Try HH:MM:SS or MM:SS
    parts = duration_str.split(":")
    try:
        if len(parts) == 2:
            # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        pass
    
    return None


# ── State ─────────────────────────────────────────────────────────────────────

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        data["processed_guids"] = set(data.get("processed_guids", []))
        data.setdefault("episodes", [])
        data.setdefault("retry_queue", [])
        return data
    return {"processed_guids": set(), "last_run": None, "episodes": [], "retry_queue": []}

def save_state(state):
    serialisable = dict(state)
    serialisable["processed_guids"] = list(state["processed_guids"])
    if os.path.exists(STATE_FILE):
        shutil.copy(STATE_FILE, STATE_BACKUP)
    with open(STATE_FILE, "w") as f:
        json.dump(serialisable, f, indent=2)
    log.info("State saved.")


def queue_episode_for_retry(state, episode, error):
    retry_queue = [item for item in state.get("retry_queue", []) if item.get("guid") != episode.get("guid")]
    queued_episode = dict(episode)
    queued_episode["retry_queued_at"] = datetime.now().isoformat()
    queued_episode["last_error"] = str(error)
    retry_queue.append(queued_episode)
    state["retry_queue"] = retry_queue
    if episode.get("guid"):
        state["processed_guids"].discard(episode["guid"])


def remove_episode_from_retry_queue(state, guid):
    if not guid:
        return
    retry_queue = state.get("retry_queue", [])
    filtered = [item for item in retry_queue if item.get("guid") != guid]
    if len(filtered) != len(retry_queue):
        state["retry_queue"] = filtered
        log.info(f"Removed episode from retry queue after success: {guid}")

# ── RSS Fetching ──────────────────────────────────────────────────────────────

def parse_entry_published_at(entry):
    published_parsed = entry.get("published_parsed")
    if published_parsed:
        try:
            return datetime.fromtimestamp(time.mktime(published_parsed))
        except Exception:
            pass

    for field in ("published", "updated"):
        value = entry.get(field)
        if not value:
            continue
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(value)
            if dt.tzinfo is not None:
                return dt.astimezone().replace(tzinfo=None)
            return dt
        except Exception:
            continue
    return None



def get_latest_transcribed_published_at(state):
    latest = None
    for ep in state.get("episodes", []):
        published_at = ep.get("published_at")
        if published_at:
            try:
                dt = datetime.fromisoformat(published_at)
            except Exception:
                dt = None
        else:
            published_text = ep.get("published")
            if not published_text:
                dt = None
            else:
                try:
                    from email.utils import parsedate_to_datetime
                    parsed = parsedate_to_datetime(published_text)
                    dt = parsed.astimezone().replace(tzinfo=None) if parsed.tzinfo is not None else parsed
                except Exception:
                    dt = None
        if dt is None:
            continue
        if latest is None or dt > latest:
            latest = dt
    return latest



def fetch_new_episodes_for_podcast(podcast_name, rss_url, title_filter, state, config, latest_transcribed_published_at=None):
    """Fetch matching episodes for a podcast based on the current run mode."""
    log.info(f"[{podcast_name}] Fetching RSS: {rss_url}")
    t0 = time.time()
    feed = feedparser.parse(rss_url)
    rss_elapsed = time.time() - t0

    if feed.bozo:
        log.warning(f"[{podcast_name}] RSS parse warning: {feed.bozo_exception}")

    entries = feed.entries
    log.info(f"[{podcast_name}] RSS fetch complete in {rss_elapsed:.1f}s — {len(entries)} entries in feed.")

    candidates = []

    for entry in entries:
        title = entry.get("title", "Unknown Episode")

        if title_filter and title_filter.lower() not in title.lower():
            continue

        guid = entry.get("id") or entry.get("guid") or entry.get("link")
        if not guid:
            continue

        published_at = parse_entry_published_at(entry)

        if config.test_mode:
            # Test mode intentionally allows full reruns of recent published episodes.
            pass
        else:
            if guid in state["processed_guids"]:
                continue
            if latest_transcribed_published_at and published_at and published_at <= latest_transcribed_published_at:
                continue

        audio_url = None
        for enc in entry.get("enclosures", []):
            if "audio" in enc.get("type", "") or enc.get("url", "").endswith(".mp3"):
                audio_url = enc["url"]
                break

        if not audio_url:
            log.warning(f"[{podcast_name}] No audio found for: {title}")
            continue

        duration_raw = entry.get("itunes_duration") or entry.get("duration")
        duration_s = parse_duration(duration_raw)

        if duration_s is not None and duration_s < config.min_episode_duration_secs:
            log.info(f"[{podcast_name}] Skipping short episode ({duration_s}s < {config.min_episode_duration_secs}s): {title}")
            continue

        if duration_s:
            duration_min = duration_s / 60
            log.info(f"[{podcast_name}] Episode duration: {duration_min:.1f} min")

        candidates.append({
            "guid": guid,
            "podcast": podcast_name,
            "title": title,
            "published": entry.get("published", ""),
            "published_at": published_at.isoformat() if published_at else None,
            "audio_url": audio_url,
            "link": entry.get("link", ""),
            "summary": entry.get("summary", ""),
            "duration_s": duration_s,
        })

    candidates.sort(key=lambda ep: ep.get("published_at") or "", reverse=True)

    if config.test_mode:
        log.info(
            f"[{podcast_name}] TEST_MODE: added {len(candidates)} candidate episode(s) to the shared test-mode pool."
        )
        return candidates

    if not candidates:
        log.info(f"[{podcast_name}] No new episodes.")
        return []

    log.info(f"[{podcast_name}] New episodes found since latest transcribed published timestamp: {len(candidates)}")
    return candidates[:config.max_episodes_per_run]



def fetch_all_new_episodes(state, config):
    """Fetch episodes across all configured feeds based on watermark or test mode."""
    all_episodes = []
    latest_transcribed_published_at = get_latest_transcribed_published_at(state)
    if latest_transcribed_published_at:
        log.info(f"Latest transcribed episode published timestamp: {latest_transcribed_published_at.isoformat()}")
    else:
        log.info("Latest transcribed episode published timestamp: none (first-run behavior)")

    if not config.test_mode:
        retry_queue = state.get("retry_queue", [])
        if retry_queue:
            log.info(f"Pending retry episodes carried into this run: {len(retry_queue)}")
            all_episodes.extend(retry_queue)

    for podcast_name, rss_url, title_filter in config.podcasts:
        try:
            eps = fetch_new_episodes_for_podcast(
                podcast_name,
                rss_url,
                title_filter,
                state,
                config,
                latest_transcribed_published_at=latest_transcribed_published_at,
            )
            all_episodes.extend(eps)
        except Exception as e:
            log.error(f"[{podcast_name}] Failed to fetch RSS: {e}")

    deduped_episodes = []
    seen_guids = set()
    for ep in all_episodes:
        guid = ep.get("guid")
        if guid and guid in seen_guids:
            continue
        if guid:
            seen_guids.add(guid)
        deduped_episodes.append(ep)

    all_episodes = deduped_episodes
    all_episodes.sort(key=lambda ep: ep.get("published_at") or "", reverse=True)

    if config.test_mode:
        filtered_episodes = all_episodes
        if config.test_mode_month_number is not None:
            month_start, month_end = get_most_recent_month_window(config.test_mode_month_number)
            filtered_episodes = [
                ep for ep in all_episodes
                if ep.get("published_at")
                and month_start <= datetime.fromisoformat(ep["published_at"]) < month_end
            ]
            log.info(
                "TEST_MODE: limiting candidates to the most recent occurrence of "
                f"{calendar.month_name[config.test_mode_month_number]} "
                f"({month_start.strftime('%Y-%m')}) — {len(filtered_episodes)} match(es) found."
            )

        if config.test_mode_skip_title_contains:
            skip_lower = config.test_mode_skip_title_contains.lower()
            before_skip = len(filtered_episodes)
            filtered_episodes = [
                ep for ep in filtered_episodes
                if skip_lower not in ep.get("title", "").lower()
            ]
            log.info(
                "TEST_MODE: excluding episodes whose title contains "
                f"'{config.test_mode_skip_title_contains}' — {before_skip - len(filtered_episodes)} excluded."
            )

        selected = filtered_episodes[:config.test_mode_episode_count]
        if config.test_mode_month_number is not None:
            log.info(
                f"TEST_MODE: Processing {len(selected)} episode(s) from "
                f"{calendar.month_name[config.test_mode_month_number]} {month_start.year} across all matching feeds."
            )
        else:
            log.info(
                f"TEST_MODE: Processing {len(selected)} most recently published episode(s) across all matching feeds."
            )
        return selected

    log.info(f"Total new episodes to process across all podcasts: {len(all_episodes)}")
    return all_episodes

# ── Audio Download & Retention ───────────────────────────────────────────────

def make_safe_title(title, config):
    return re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:config.safe_title_max_len]

def get_audio_path(title, config):
    """Get the expected audio file path for a title."""
    safe_title = make_safe_title(title, config)
    return os.path.join(AUDIO_DIR, f"{safe_title}.mp3")


def check_cached_audio(title, config):
    """Check if audio file exists and is within retention period."""
    filepath = get_audio_path(title, config)
    if not os.path.exists(filepath):
        return None
    
    # Check if file is within retention period
    file_age_hours = (time.time() - os.path.getmtime(filepath)) / 3600
    if file_age_hours <= config.audio_retention_hours:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        log.info(f"Using cached audio: {os.path.basename(filepath)} ({size_mb:.1f} MB, {file_age_hours:.1f}h old)")
        return filepath
    else:
        log.info(f"Cached audio expired ({file_age_hours:.1f}h > {config.audio_retention_hours}h): {os.path.basename(filepath)}")
        os.remove(filepath)
        return None


def cleanup_old_audio(config):
    """Remove audio files older than retention period."""
    if not os.path.exists(AUDIO_DIR):
        return
    
    now = time.time()
    removed = 0
    for filename in os.listdir(AUDIO_DIR):
        filepath = os.path.join(AUDIO_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        file_age_hours = (now - os.path.getmtime(filepath)) / 3600
        if file_age_hours > config.audio_retention_hours:
            os.remove(filepath)
            removed += 1
            log.info(f"Removed expired audio: {filename} ({file_age_hours:.1f}h old)")
    
    if removed > 0:
        log.info(f"Audio cleanup: removed {removed} expired file(s)")


def download_audio(url, title, config, retries=None):
    """Download audio file, or return cached version if available."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    if retries is None:
        retries = config.download_retries
    
    # Check for cached audio first
    cached = check_cached_audio(title, config)
    if cached:
        size_mb = os.path.getsize(cached) / (1024 * 1024)
        return cached, 0.0, size_mb  # 0 download time for cached
    
    safe_title = make_safe_title(title, config)
    filename = os.path.join(AUDIO_DIR, f"{safe_title}.mp3")

    for attempt in range(retries):
        try:
            log.info(f"Downloading audio: {title} (attempt {attempt + 1})")
            t0 = time.time()
            with requests.get(url, stream=True, timeout=config.download_timeout_seconds) as r:
                r.raise_for_status()
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1_048_576):
                        f.write(chunk)
            elapsed = time.time() - t0
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            log.info(f"Downloaded {size_mb:.1f} MB to {filename} [{elapsed:.1f}s]")
            return filename, elapsed, size_mb
        except Exception as e:
            log.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(config.download_retry_backoff_base_seconds * (attempt + 1))
            else:
                raise Exception(f"Failed to download after {retries} attempts: {e}")

# ── Shared Model Resources ───────────────────────────────────────────────────

def detect_whisperx_device(config):
    import torch

    if config.whisperx_device == "mps" and torch.backends.mps.is_available():
        return "mps"
    if config.whisperx_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_shared_whisperx_resources(config):
    """Load and warm shared WhisperX transcription/alignment resources once per run."""
    import whisperx
    import torch

    result = {
        "model": None,
        "align_model": None,
        "align_metadata": None,
        "device": detect_whisperx_device(config),
        "error": None,
        "elapsed_s": 0,
        "model_load_s": 0,
        "align_load_s": 0,
        "warmup_s": 0,
    }

    t0 = time.time()
    try:
        device = result["device"]

        log.info(f"Loading shared WhisperX model (device={device}, compute_type={config.whisper_compute_type})...")
        t_model = time.time()
        result["model"] = whisperx.load_model(
            config.whisper_model,
            device=device,
            compute_type=config.whisper_compute_type,
            language="en",
            vad_method="silero",
            local_files_only=True,
        )
        result["model_load_s"] = time.time() - t_model
        log.info(f"Shared WhisperX model loaded [{result['model_load_s']:.1f}s]")

        log.info(f"Loading shared alignment model (device={device})...")
        t_align = time.time()
        align_model, align_metadata = whisperx.load_align_model(
            language_code="en",
            device=device,
            model_cache_only=True,
        )
        result["align_model"] = align_model
        result["align_metadata"] = align_metadata
        result["align_load_s"] = time.time() - t_align
        log.info(f"Shared alignment model loaded [{result['align_load_s']:.1f}s]")

        log.info("Warming shared WhisperX/VAD/alignment resources...")
        t_warm = time.time()
        silent_audio = [0.0] * 16000
        try:
            warm_result = result["model"].transcribe(silent_audio, batch_size=1, language="en")
            warm_segments = warm_result.get("segments", [])
            if warm_segments:
                whisperx.align(
                    warm_segments,
                    result["align_model"],
                    result["align_metadata"],
                    silent_audio,
                    device,
                    return_char_alignments=False,
                )
                result["warmup_s"] = time.time() - t_warm
                log.info(f"Shared resource warmup complete [{result['warmup_s']:.1f}s]")
            else:
                result["warmup_s"] = time.time() - t_warm
                log.warning("Shared resource warmup produced no speech segments; continuing without alignment warmup.")
        except Exception as warmup_error:
            result["warmup_s"] = time.time() - t_warm
            log.warning(f"Shared resource warmup skipped after non-fatal warmup error: {warmup_error}")

        result["elapsed_s"] = time.time() - t0
    except Exception as e:
        result["error"] = e
        log.error(f"Failed to load shared WhisperX resources: {e}")

    return result


def release_shared_whisperx_resources(shared_resources):
    if not shared_resources:
        return

    try:
        import torch
    except Exception:
        torch = None

    for key in ("model", "align_model", "align_metadata"):
        if key in shared_resources:
            shared_resources[key] = None

    gc.collect()
    if torch and shared_resources.get("device") == "mps":
        torch.mps.empty_cache()


# ── WhisperX Transcription ───────────────────────────────────────────────────

def transcribe_with_whisperx(audio_path, title, config, podcast_name=None, preloaded_model=None,
                              shared_resources=None, episode_title="", episode_summary=""):
    """
    Transcribe audio using WhisperX.
    Uses known hosts from phrases-and-vocabulary/known-hosts-per-podcast.txt for optional speaker naming.
    Returns detailed timing and resource metrics for transcription/alignment only.
    
    Args:
        audio_path: Path to audio file
        title: Safe title for output files
        podcast_name: Name of podcast to look up hosts
        preloaded_model: Deprecated pre-loaded WhisperX model dict (optional)
        shared_resources: Shared WhisperX/alignment resources loaded once per run
        episode_title: Episode title for guest name extraction
        episode_summary: Episode summary/show notes for guest name extraction
    """
    import whisperx
    import torch
    
    safe_title = make_safe_title(title, config)
    txt_path = os.path.join(TRANSCRIPT_DIR, f"raw_transcript_{safe_title}.txt")
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    
    metrics = {
        "whisper_model": config.whisper_model,
        "device": config.whisperx_device,
        "compute_type": config.whisper_compute_type,
        "batch_size": config.whisper_batch_size,
    }
    
    # Detect device
    device = detect_whisperx_device(config)
    metrics["actual_device"] = device
    
    log.info(f"WhisperX transcribing: {title}")
    log.info(f"  Config: model={config.whisper_model}, device={device}, compute_type={config.whisper_compute_type}, batch_size={config.whisper_batch_size}")
    
    mem_start = get_memory_usage_mb()
    metrics["memory_start_mb"] = round(mem_start, 1)
    
    # ── Phase 2a: Load WhisperX model (or use shared/preloaded) ──────────────
    if shared_resources and shared_resources.get("model"):
        model = shared_resources["model"]
        t_model_load = 0.0
        metrics["model_preloaded"] = True
        metrics["shared_model_reused"] = True
        log.info("  Using shared WhisperX model (loaded once at run start)")
    elif preloaded_model and preloaded_model.get("model"):
        log.info(f"  Using preloaded WhisperX model (loaded in {preloaded_model.get('elapsed_s', 0):.1f}s)")
        model = preloaded_model["model"]
        t_model_load = preloaded_model.get("elapsed_s", 0)
        metrics["model_preloaded"] = True
        metrics["shared_model_reused"] = False
    else:
        log.info("  Loading WhisperX model...")
        t0 = time.time()
        model = whisperx.load_model(
            config.whisper_model,
            device=device,
            compute_type=config.whisper_compute_type,
            language="en",
            vad_method="silero",
            local_files_only=True,
        )
        t_model_load = time.time() - t0
        metrics["model_preloaded"] = False
        metrics["shared_model_reused"] = False
    metrics["model_load_s"] = round(t_model_load, 2)
    log_resources("Model Load")
    log.info(f"  ⏱ Phase 2a (Model Load):    {t_model_load:.1f}s{' [preloaded]' if metrics.get('model_preloaded') else ''}")
    
    # ── Phase 2b: Transcription ───────────────────────────────────────────────
    log.info("  Transcribing audio...")
    t0 = time.time()
    audio = whisperx.load_audio(audio_path)
    audio_duration_s = len(audio) / 16000  # whisperx loads at 16kHz
    metrics["audio_duration_s"] = round(audio_duration_s, 1)
    metrics["audio_duration_min"] = round(audio_duration_s / 60, 1)
    
    result = model.transcribe(audio, batch_size=config.whisper_batch_size, language="en")
    t_transcribe = time.time() - t0
    metrics["transcribe_s"] = round(t_transcribe, 2)
    metrics["transcribe_rtf"] = round(audio_duration_s / t_transcribe, 2) if t_transcribe > 0 else 0
    log_resources("Transcription")
    log.info(f"  ⏱ Phase 2b (Transcribe):    {t_transcribe:.1f}s (RTF: {metrics['transcribe_rtf']:.1f}x)")
    
    # Free whisper model memory (skip if shared/preloaded - caller manages lifecycle)
    if not shared_resources and not preloaded_model:
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    
    # ── Phase 2c: Alignment ───────────────────────────────────────────────────
    log.info("  Aligning transcript (word-level timestamps)...")
    t0 = time.time()
    if shared_resources and shared_resources.get("align_model") and shared_resources.get("align_metadata"):
        model_a = shared_resources["align_model"]
        metadata = shared_resources["align_metadata"]
        metrics["shared_align_reused"] = True
        log.info("  Using shared alignment model (loaded once at run start)")
    else:
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device, model_cache_only=True)
        metrics["shared_align_reused"] = False
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    t_align = time.time() - t0
    metrics["align_s"] = round(t_align, 2)
    log_resources("Alignment")
    log.info(f"  ⏱ Phase 2c (Alignment):     {t_align:.1f}s")
    
    # Free alignment model memory if not shared
    if not (shared_resources and shared_resources.get("align_model") is model_a):
        del model_a
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    
    # ── Build transcript text ─────────────────────────────────────────────────
    segments = result.get("segments", [])
    segments = strip_opening_ad_segments(segments, config.opening_ad_phrases, max_start_seconds=config.opening_ad_max_start_seconds)
    
    # Build paragraphs without diarization speaker labels
    paragraphs = []
    current_speaker = None
    current_para = []
    prev_end = None
    
    for seg in segments:
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        
        if not text:
            continue
        
        # Start new paragraph on long pause only
        gap = (start - prev_end) if prev_end is not None else 0
        long_pause = gap >= config.paragraph_gap_secs
        
        if long_pause and current_para:
            para_text = " ".join(current_para)
            paragraphs.append(para_text)
            current_para = []
        
        current_para.append(text)
        prev_end = seg.get("end", start)
    
    # Final paragraph
    if current_para:
        para_text = " ".join(current_para)
        paragraphs.append(para_text)
    
    plain_text = "\n\n".join(paragraphs)
    metrics["paragraphs"] = len(paragraphs)
    metrics["chars"] = len(plain_text)
    metrics["words"] = len(plain_text.split())
    
    # Save raw transcript artifact only in test mode
    if config.test_mode:
        with open(txt_path, "w") as f:
            f.write(plain_text)
    
    # Total transcription time
    t_total = t_model_load + t_transcribe + t_align
    metrics["total_transcribe_s"] = round(t_total, 2)
    
    mem_end = get_memory_usage_mb()
    metrics["memory_end_mb"] = round(mem_end, 1)
    metrics["memory_delta_mb"] = round(mem_end - mem_start, 1)
    
    log.info(f"  Transcription complete: {len(paragraphs)} paragraphs, {len(plain_text)} chars [{t_total:.1f}s total]")
    
    return txt_path, plain_text, metrics, segments, audio

# ── LLM Cleanup Pass ──────────────────────────────────────────────────────────

def llm_cleanup(text: str, config: RuntimeConfig) -> tuple:
    """Run a lightweight LLM pass to clean up punctuation and filler words."""
    if config.llm_provider == "none":
        return text, {}

    log.info(f"Running LLM cleanup pass via {config.llm_provider}...")
    t0 = time.time()
    llm_timeout = max(1, config.llm_request_timeout_seconds)
    request_content = config.llm_cleanup_prompt + text
    max_output_tokens = int(os.getenv("LLM_MAX_TOKENS"))
    estimated_input_tokens = max(1, (len(request_content) + 3) // 4)

    log.info(
        "LLM cleanup request estimate — "
        f"input tokens: ~{estimated_input_tokens}, max output tokens: {max_output_tokens} "
        f"[provider={config.llm_provider}, model={config.llm_model}]"
    )

    def is_timeout_error(error):
        message = str(error).lower()
        return "timed out" in message or "timeout" in message

    def normalize_sentence(sentence):
        sentence = sentence.strip().lower()
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence.strip(" \t\n\r\"'“”‘’.,!?;:-")

    def analyze_degenerate_output(cleaned_text, usage):
        stripped = (cleaned_text or "").strip()
        if not stripped:
            return "empty cleanup output"

        sentences = [
            normalize_sentence(part)
            for part in re.split(r"(?<=[.!?])\s+", stripped)
        ]
        sentences = [s for s in sentences if s]

        max_consecutive_repeats = 1
        current_repeats = 1
        for idx in range(1, len(sentences)):
            if sentences[idx] == sentences[idx - 1]:
                current_repeats += 1
                max_consecutive_repeats = max(max_consecutive_repeats, current_repeats)
            else:
                current_repeats = 1

        if max_consecutive_repeats >= 8:
            return f"repeated sentence loop detected ({max_consecutive_repeats} consecutive repeats)"

        if len(sentences) >= 12:
            counts = {}
            for sentence in sentences:
                counts[sentence] = counts.get(sentence, 0) + 1
            repeated_sentences = sum(count for count in counts.values() if count > 1)
            repetition_ratio = repeated_sentences / len(sentences)
            if repetition_ratio >= 0.35:
                return f"repetition ratio too high ({repetition_ratio:.2f})"

        output_tokens = usage.get("output_tokens")
        ends_cleanly = stripped.endswith((".", "!", "?", '"', "'", "…”", "...”", "…"))
        if output_tokens == max_output_tokens and not ends_cleanly:
            return "cleanup output hit max tokens and appears truncated"

        return None

    def get_retryable_service_error_details(error):
        status_code = getattr(error, "status_code", None) or getattr(error, "status", None)
        body = getattr(error, "body", None)

        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                body = None

        if status_code is None:
            match = re.search(r"error code:\s*(\d{3})", str(error), re.IGNORECASE)
            if match:
                status_code = int(match.group(1))

        retryable = False
        retry_after = None
        if isinstance(body, dict):
            retryable = bool(body.get("retryable"))
            raw_retry_after = body.get("retry_after")
            if raw_retry_after is not None:
                try:
                    retry_after = max(1, int(float(raw_retry_after)))
                except Exception:
                    retry_after = None

        if retry_after is None:
            match = re.search(r"retry_after['\"]?\s*[:=]\s*(\d+)", str(error), re.IGNORECASE)
            if match:
                retry_after = max(1, int(match.group(1)))

        retryable_statuses = {500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
        is_retryable_service_error = retryable or (status_code in retryable_statuses)
        return is_retryable_service_error, retry_after, status_code

    def run_cleanup_once():
        usage = {}

        if config.llm_provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=config.llm_model,
                max_tokens=max_output_tokens,
                temperature=config.llm_temperature,
                messages=[{"role": "user", "content": request_content}]
            )
            cleaned = message.content[0].text.strip()
            usage = {
                "provider": "anthropic", "model": config.llm_model,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
            }

        elif config.llm_provider == "openai":
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0)
            response = client.chat.completions.create(
                model=config.llm_model,
                messages=[{"role": "user", "content": request_content}],
                max_tokens=max_output_tokens,
                temperature=config.llm_temperature,
                timeout=llm_timeout,
            )
            cleaned = response.choices[0].message.content.strip()
            usage = {
                "provider": "openai", "model": config.llm_model,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        elif config.llm_provider in ("grok", "xai"):
            import openai
            client = openai.OpenAI(
                api_key=os.getenv("XAI_API_KEY"),
                base_url=config.service_endpoints["XAI_BASE_URL"],
                max_retries=0,
            )
            response = client.chat.completions.create(
                model=config.llm_model,
                messages=[{"role": "user", "content": request_content}],
                max_tokens=max_output_tokens,
                temperature=config.llm_temperature,
                timeout=llm_timeout,
            )
            cleaned = response.choices[0].message.content.strip()
            usage = {
                "provider": "grok", "model": config.llm_model,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        elif config.llm_provider == "perplexity":
            import openai
            client = openai.OpenAI(
                api_key=os.getenv("PERPLEXITY_API_KEY"),
                base_url=config.service_endpoints["PERPLEXITY_BASE_URL"],
                max_retries=0,
            )
            response = client.chat.completions.create(
                model=config.llm_model,
                messages=[{"role": "user", "content": request_content}],
                max_tokens=max_output_tokens,
                temperature=config.llm_temperature,
                timeout=llm_timeout,
            )
            cleaned = response.choices[0].message.content.strip()
            usage = {
                "provider": "perplexity", "model": config.llm_model,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        else:
            log.warning(f"Unknown LLM_PROVIDER '{config.llm_provider}' — skipping cleanup.")
            return text, {}

        return cleaned, usage

    try:
        timeout_retry_used = False
        service_retry_used = False
        degenerate_retry_used = False
        degenerate_backoff_retry_used = False

        for attempt in (1, 2, 3, 4, 5):
            try:
                cleaned, usage = run_cleanup_once()
                degeneration_reason = analyze_degenerate_output(cleaned, usage)
                if degeneration_reason:
                    raise ValueError(f"LLM cleanup produced degenerate output: {degeneration_reason}")
                break
            except Exception as e:
                if str(e).startswith("LLM cleanup produced degenerate output:"):
                    if not degenerate_retry_used:
                        degenerate_retry_used = True
                        log.warning(
                            "LLM cleanup produced degenerate output; rerunning once before fallback. "
                            f"[{e} | provider={config.llm_provider}, timeout={llm_timeout}s, "
                            f"estimated_input_tokens={estimated_input_tokens}, max_output_tokens={max_output_tokens}]"
                        )
                        continue

                    if not degenerate_backoff_retry_used:
                        degenerate_backoff_retry_used = True
                        backoff_seconds = 3600
                        log.warning(
                            "LLM cleanup produced degenerate output again; waiting 1 hour before one final retry. "
                            f"[{e} | provider={config.llm_provider}, backoff={backoff_seconds}s, timeout={llm_timeout}s, "
                            f"estimated_input_tokens={estimated_input_tokens}, max_output_tokens={max_output_tokens}]"
                        )
                        time.sleep(backoff_seconds)
                        continue

                if not timeout_retry_used and is_timeout_error(e):
                    timeout_retry_used = True
                    log.warning(
                        "LLM cleanup timed out; retrying once before fallback. "
                        f"[provider={config.llm_provider}, timeout={llm_timeout}s, "
                        f"estimated_input_tokens={estimated_input_tokens}, max_output_tokens={max_output_tokens}]"
                    )
                    continue

                is_retryable_service_error, retry_after, status_code = get_retryable_service_error_details(e)
                if not service_retry_used and is_retryable_service_error and retry_after:
                    service_retry_used = True
                    log.warning(
                        "LLM cleanup hit a retryable upstream error; backing off before retry. "
                        f"[provider={config.llm_provider}, status_code={status_code}, retry_after={retry_after}s, "
                        f"estimated_input_tokens={estimated_input_tokens}, max_output_tokens={max_output_tokens}]"
                    )
                    time.sleep(retry_after)
                    continue

                raise

        elapsed = time.time() - t0
        usage["elapsed_s"] = round(elapsed, 1)
        log.info(
            f"LLM cleanup complete via {usage['provider']} ({usage['model']}) — "
            f"tokens: {usage['input_tokens']} in / {usage['output_tokens']} out / "
            f"{usage['total_tokens']} total [{elapsed:.1f}s]"
        )
        return cleaned, usage

    except ImportError as e:
        log.warning(f"LLM client library not installed ({e}) — skipping cleanup.")
        return text, {}
    except Exception as e:
        log.error(
            "LLM cleanup failed (non-fatal, using post-processed transcript): "
            f"{e} [provider={config.llm_provider}, timeout={llm_timeout}s, "
            f"estimated_input_tokens={estimated_input_tokens}, max_output_tokens={max_output_tokens}]"
        )
        return text, {}

def insert_podcast_start_marker(text, closing_ad_phrases, opening_catch_phrases, opening_ad_phrases, config):
    """Insert PODCAST START after closing-ad phrase matches in the early transcript slice, also before an opening catch phrase when present, then apply fallback insertion before paragraph cleanup."""
    if not text:
        return text, False

    def normalize_for_phrase_match(value):
        value = value.lower()
        value = re.sub(r"\bdot\s+com\b", " com ", value)
        value = re.sub(r"\bslash\b", " ", value)
        value = re.sub(r"[\/._-]+", " ", value)
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def build_normalized_index_map(source_text):
        normalized_chars = []
        index_map = []
        prev_was_space = False
        for idx, ch in enumerate(source_text):
            if ch.isalnum():
                normalized_chars.append(ch.lower())
                index_map.append(idx)
                prev_was_space = False
            else:
                if not prev_was_space and normalized_chars:
                    normalized_chars.append(" ")
                    index_map.append(idx)
                    prev_was_space = True
        if normalized_chars and normalized_chars[-1] == " ":
            normalized_chars.pop()
            index_map.pop()
        return "".join(normalized_chars), index_map

    marked = text
    inserted = False
    early_char_limit = config.podcast_start_early_char_limit

    early_slice = marked[:early_char_limit]
    normalized_early_slice, normalized_index_map = build_normalized_index_map(early_slice)
    close_matches = []
    for phrase in closing_ad_phrases:
        normalized_phrase = normalize_for_phrase_match(phrase)
        if not normalized_phrase:
            continue
        start_search = 0
        while True:
            idx = normalized_early_slice.find(normalized_phrase, start_search)
            if idx == -1:
                break
            mapped_start = normalized_index_map[idx]
            mapped_end = normalized_index_map[idx + len(normalized_phrase) - 1] + 1
            close_matches.append((mapped_start, mapped_end, phrase))
            start_search = idx + len(normalized_phrase)

    if close_matches:
        close_matches.sort(key=lambda x: x[0])
        rebuilt = []
        cursor = 0
        for start_idx, end_idx, phrase in close_matches:
            rebuilt.append(marked[cursor:end_idx])
            rebuilt.append("\n\nPODCAST START\n\n")
            cursor = end_idx
            log.info(f"  Inserted PODCAST START after closing ad phrase: {phrase}")
        rebuilt.append(marked[cursor:])
        marked = "".join(rebuilt)
        inserted = True

    lower_marked = marked.lower()
    earliest_open_idx = None
    matched_open_phrase = None
    for phrase in opening_catch_phrases:
        idx = lower_marked.find(phrase.lower())
        if idx != -1 and (earliest_open_idx is None or idx < earliest_open_idx):
            earliest_open_idx = idx
            matched_open_phrase = phrase

    if earliest_open_idx is not None:
        marked = marked[:earliest_open_idx] + "PODCAST START\n\n" + marked[earliest_open_idx:]
        inserted = True
        log.info(f"  Inserted PODCAST START before catch phrase: {matched_open_phrase}")

    if not inserted:
        marked = "PODCAST START\n\n" + marked
        inserted = True
        log.info("  Inserted PODCAST START at beginning of transcript (final fallback)")

    cleaned_paragraphs = []
    for paragraph in re.split(r"\n\s*\n", marked):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        cleaned_paragraphs.append(paragraph)
    marked = "\n\n".join(cleaned_paragraphs)
    log.info("  Cleaned transcript paragraphs while preserving paragraph blocks")

    return marked, True


# ── HTML Formatting ───────────────────────────────────────────────────────────

def _word_count_meta(plain_text):
    words = len(plain_text.split())
    read_min = round(words / 238)
    return words, read_min


def chunk_paragraphs_for_markdown(text, max_sentences=4):
    """Split long transcript paragraphs into smaller markdown-friendly chunks."""
    source_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not source_paragraphs:
        return []

    chunks = []
    sentence_split_re = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    for paragraph in source_paragraphs:
        if paragraph.strip() == "PODCAST START":
            chunks.append(paragraph)
            continue

        sentences = [s.strip() for s in sentence_split_re.split(paragraph) if s.strip()]
        if not sentences:
            continue

        current = []
        for sentence in sentences:
            current.append(sentence)
            if len(current) >= max_sentences:
                chunks.append(" ".join(current).strip())
                current = []
        if current:
            chunks.append(" ".join(current).strip())

    return chunks



def build_final_text_transcript(episode, plain_text, speakers=None):
    """Render the final plain-text transcript without image references."""
    chunks = chunk_paragraphs_for_markdown(plain_text, max_sentences=4)

    published_display = episode.get("published", "")
    published_at = parse_entry_published_at(episode)
    if published_at:
        published_display = published_at.strftime("%Y-%m-%d")

    header = [
        f"# {episode['title']}",
        "",
        f"**Published:** {published_display}",
    ]

    speakers_line = format_speakers_line(speakers or [])
    if speakers_line:
        header.extend(["", speakers_line])

    header.extend([
        "",
        "---",
        "",
    ])

    body = []
    for chunk in chunks:
        if chunk.strip() == "PODCAST START":
            body.extend(["---", ""])
            continue
        body.extend([chunk, ""])

    return "\n".join(header + body).strip() + "\n"



def build_final_markdown_transcript(episode, plain_text, speakers=None):
    """Render the final markdown transcript artifact without inline drawings."""
    chunks = chunk_paragraphs_for_markdown(plain_text, max_sentences=4)

    published_display = episode.get("published", "")
    published_at = parse_entry_published_at(episode)
    if published_at:
        published_display = published_at.strftime("%Y-%m-%d")

    header = [
        f"# {episode['title']}",
        "",
        f"**Published:** {published_display}",
    ]

    speakers_line = format_speakers_line(speakers or [])
    if speakers_line:
        header.extend(["", speakers_line])

    header.extend([
        "",
        "---",
        "",
    ])

    body = []
    for chunk in chunks:
        if chunk.strip() == "PODCAST START":
            body.extend(["---", ""])
            continue

        body.extend([chunk, ""])

    return "\n".join(header + body).strip() + "\n"



def format_transcript_html(episode, final_text, drawing_src=None, footer_src=None):
    """Render HTML email that mirrors the final cleaned transcript structure."""
    words, read_min = _word_count_meta(final_text)

    published_display = episode.get("published", "")
    published_at = parse_entry_published_at(episode)
    if published_at:
        published_display = published_at.strftime("%Y-%m-%d")

    lines = final_text.splitlines()
    body_lines = []
    divider_seen = False
    for line in lines:
        if not divider_seen and line.strip() == "---":
            divider_seen = True
            continue
        if divider_seen:
            body_lines.append(line)

    def render_paragraph(p):
        p = html.escape(p)
        p = re.sub(r'\*\*([^*]+):\*\*', r'<strong style="color:#1a1a2e">\1:</strong>', p)
        return f"        <p>{p}</p>"

    paragraph_blocks = []
    drawing_inserted = False
    content_paragraph_index = 0
    for block in "\n".join(body_lines).split("\n\n"):
        text = block.strip()
        if not text:
            continue
        if text == "---":
            paragraph_blocks.append('        <hr class="section-break">')
            continue

        content_paragraph_index += 1
        if drawing_src and content_paragraph_index == 2 and not drawing_inserted:
            safe_text = html.escape(text)
            safe_text = re.sub(r'\*\*([^*]+):\*\*', r'<strong style="color:#1a1a2e">\1:</strong>', safe_text)
            paragraph_blocks.append(f'''        <div class="image-wrap-block">
          <img class="inline-show-drawing" src="{html.escape(drawing_src, quote=True)}" alt="{html.escape(episode['podcast'])} drawing">
          <p>{safe_text}</p>
        </div>''')
            drawing_inserted = True
            continue

        paragraph_blocks.append(render_paragraph(text))

    paragraph_html = "\n".join(paragraph_blocks)

    speakers_match = re.search(r"\*\*Speakers:\*\*\s*(.+)", final_text)

    production_crew_html = ""
    if footer_src:
        production_crew_html = f'''
      <div class="footer-image-wrap">
        <img class="footer-image" src="{html.escape(footer_src, quote=True)}" alt="Production crew drawing">
      </div>'''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    body {{ margin: 0; padding: 0; background: #f5f5f0; font-family: Georgia, "Times New Roman", serif; color: #1a1a1a; }}
    .wrapper {{ max-width: 760px; margin: 40px auto; background: #ffffff; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
    .header {{ background: #1a1a2e; padding: 36px 48px 28px; color: #ffffff; }}
    .header .label {{ font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; font-size: 14px; letter-spacing: 0.8px; text-transform: uppercase; color: #ffffff; margin-bottom: 10px; }}
    .header h1 {{ margin: 0 0 14px; font-size: 24px; font-weight: normal; line-height: 1.35; color: #f0f0f8; }}
    .header .meta {{ font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; font-size: 14px; color: #9b9bc0; line-height: 1.6; }}
    .header .readtime {{ display: inline-block; margin-top: 10px; font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; font-size: 12px; color: #7b9cdf; letter-spacing: 0.5px; }}
    .header a {{ color: #7b9cdf; text-decoration: none; }}
    .divider {{ height: 3px; background: linear-gradient(90deg, #7b9cdf, #c778e0); }}
    .body {{ padding: 40px 48px 48px; }}
    .body p {{ font-size: 17px; line-height: 1.85; margin: 0 0 1.4em; color: #1a1a1a; white-space: pre-wrap; }}
    .section-break {{ border: 0; border-top: 1px solid #e8e8e4; margin: 2em 0; }}
    .image-wrap-block {{ overflow: auto; margin: 0 0 1.4em; }}
    .image-wrap-block p {{ margin: 0; }}
    .inline-show-drawing {{ float: left; max-width: 220px; width: 40%; min-width: 140px; height: auto; border-radius: 8px; margin: 0 18px 10px 0; }}
    .footer-image-wrap {{ margin: 36px 0 12px; text-align: center; }}
    .footer-image {{ max-width: 280px; width: 100%; height: auto; border-radius: 8px; }}
    .footer {{ padding: 20px 48px; background: #f5f5f0; font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; font-size: 12px; color: #999; border-top: 1px solid #e8e8e4; }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="header">
      <div class="label">{episode['podcast']} — Transcript</div>
      <h1>{episode['title']}</h1>
      <div class="meta">Published: {published_display}{('<br>Speakers: ' + html.escape(speakers_match.group(1))) if speakers_match else ''}<br><a href="{episode['link']}">Listen to episode →</a></div>
      <div class="readtime">📄 {words:,} words · ~{read_min} min read</div>
    </div>
    <div class="divider"></div>
    <div class="body">
{paragraph_html}
{production_crew_html}
    </div>
    <div class="footer">Transcript generated by WhisperX · The Compound Transcript Agent</div>
  </div>
</body>
</html>"""

# ── Email ─────────────────────────────────────────────────────────────────────

def parse_recipient_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_transcript_recipients(config: RuntimeConfig | None = None) -> list[str]:
    config = config or CONFIG

    default_recipients = parse_recipient_list(
        os.getenv("TRANSCRIPT_RECIPIENTS") or os.getenv("TRANSCRIPT_RECIPIENT_PATTY")
    )
    test_run_recipients = parse_recipient_list(
        os.getenv("TEST_RUN_RECIPIENTS") or os.getenv("TEST_RUN_RECIPIENT_PATTY")
    ) or default_recipients

    if config and config.test_mode:
        return test_run_recipients
    return default_recipients

def send_transcript_email(episode, plain_text, transcript_path, final_text=None, extra_attachment_paths=None, retry=True, config=None):
    config = config or CONFIG
    gmail_address = os.getenv("GMAIL_ADDRESS")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")
    recipients = get_transcript_recipients(config)

    if not all([gmail_address, gmail_password]) or not recipients:
        log.warning("Email credentials incomplete — skipping.")
        return 0

    words, read_min = _word_count_meta(plain_text)
    t0 = time.time()
    subject = f"📝 Transcript: {episode['podcast']} – {episode['title']} ({words:,} words)"

    plain_body = f"""{episode['podcast']}
Episode: {episode['title']}
Published: {episode['published']}
Listen: {episode['link']}
Words: {words:,} · ~{read_min} min read

{'─' * 60}

{plain_text}
"""
    drawing_rel_path = resolve_show_drawing(episode.get("podcast", ""), config)
    drawing_abs_path = os.path.join(BASE_DIR, drawing_rel_path) if drawing_rel_path else None
    drawing_cid = "show_drawing_image" if drawing_abs_path and os.path.exists(drawing_abs_path) else None
    footer_cid = "production_crew_footer_image" if os.path.exists(PRODUCTION_CREW_DRAWING) else None

    html_body = format_transcript_html(
        episode,
        final_text or plain_text,
        drawing_src=f"cid:{drawing_cid}" if drawing_cid else None,
        footer_src=f"cid:{footer_cid}" if footer_cid else None,
    )
    safe_title = make_safe_title(episode["title"], config)
    emailed_html_path = os.path.join(TRANSCRIPT_DIR, f"emailed_body_{safe_title}.html")

    msg = MIMEMultipart("related")
    msg["From"] = gmail_address
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(plain_body, "plain", "utf-8"))
    alt.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(alt)

    if drawing_cid:
        with open(drawing_abs_path, "rb") as f:
            img = MIMEImage(f.read())
        img.add_header("Content-ID", f"<{drawing_cid}>")
        img.add_header("Content-Disposition", f"inline; filename={os.path.basename(drawing_abs_path)}")
        msg.attach(img)

    if os.path.exists(PRODUCTION_CREW_DRAWING):
        with open(PRODUCTION_CREW_DRAWING, "rb") as f:
            footer_img = MIMEImage(f.read())
        footer_img.add_header("Content-ID", "<production_crew_footer_image>")
        footer_img.add_header("Content-Disposition", f"inline; filename={os.path.basename(PRODUCTION_CREW_DRAWING)}")
        msg.attach(footer_img)

    attachment_paths = [transcript_path]
    if extra_attachment_paths:
        attachment_paths.extend([p for p in extra_attachment_paths if p])

    for path in attachment_paths:
        with open(path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(path)}")
        msg.attach(part)

    def _attempt_send():
        with smtplib.SMTP_SSL(config.smtp_host, config.smtp_port) as server:
            server.login(gmail_address, gmail_password)
            server.sendmail(gmail_address, recipients, msg.as_string())

    def _to_browser_src(asset_path):
        if not asset_path or not os.path.exists(asset_path):
            return None
        return os.path.relpath(asset_path, start=os.path.dirname(emailed_html_path)).replace(os.sep, "/")

    def _write_local_html_copy():
        local_html_body = format_transcript_html(
            episode,
            final_text or plain_text,
            drawing_src=_to_browser_src(drawing_abs_path),
            footer_src=_to_browser_src(PRODUCTION_CREW_DRAWING),
        )
        with open(emailed_html_path, "w", encoding="utf-8") as f:
            f.write(local_html_body)
        log.info(f"Saved emailed HTML body to {emailed_html_path}")

    try:
        _attempt_send()
        _write_local_html_copy()
        elapsed = time.time() - t0
        log.info(f"Transcript emailed to {', '.join(recipients)} [{elapsed:.1f}s]")
        return elapsed
    except Exception as e:
        log.error(f"Email failed: {e}")
        if retry:
            for wait in config.email_retry_delays_seconds:
                log.info(f"Retrying email in {wait}s...")
                time.sleep(wait)
                try:
                    _attempt_send()
                    _write_local_html_copy()
                    elapsed = time.time() - t0
                    log.info(f"Transcript emailed to {', '.join(recipients)} on retry [{elapsed:.1f}s]")
                    return elapsed
                except Exception as retry_e:
                    log.error(f"Retry failed: {retry_e}")
        log.error("All email attempts failed — giving up.")
        return 0


def send_alert_email(subject, body, config=None):
    config = config or CONFIG
    gmail_address = os.getenv("GMAIL_ADDRESS")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")
    recipients = parse_recipient_list(os.getenv("NOTIFICATION_EMAIL")) or get_transcript_recipients(config)

    if not all([gmail_address, gmail_password]) or not recipients:
        return

    msg = MIMEMultipart()
    msg["From"] = gmail_address
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL(config.smtp_host, config.smtp_port) as server:
            server.login(gmail_address, gmail_password)
            server.sendmail(gmail_address, recipients, msg.as_string())
    except Exception as e:
        log.error(f"Alert email failed: {e}")



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_lock = acquire_run_lock()
    if not run_lock:
        return

    config = load_runtime_config()
    run_start = time.time()
    log.info("=" * 60)
    log.info("The Compound Transcript Agent (WhisperX Edition) starting...")
    log.info(f"  WhisperX: model={config.whisper_model} | device={config.whisperx_device} | compute_type={config.whisper_compute_type}")
    log.info(f"  LLM: {config.llm_provider} | paragraph_gap={config.paragraph_gap_secs}s")
    log.info(f"  Audio retention: {config.audio_retention_hours}h")
    if config.test_mode:
        if config.test_mode_month_number is not None:
            month_start, _ = get_most_recent_month_window(config.test_mode_month_number)
            mode_text = (
                f"TEST ({config.test_mode_episode_count} episode(s) from "
                f"{calendar.month_name[config.test_mode_month_number]} {month_start.year})"
            )
        else:
            mode_text = f"TEST ({config.test_mode_episode_count} recent episode(s))"
    else:
        mode_text = "NORMAL"
    log.info(f"  Mode: {mode_text} | force_reprocess={config.force_reprocess}")
    log.info(f"Podcasts: {[p[0] for p in config.podcasts]}")
    log.info("=" * 60)

    load_dotenv(ENV_FILE)
    state = load_state()
    shared_resources = None
    
    # Clean up expired audio files at start of run
    cleanup_old_audio(config)

    try:
        episodes = fetch_all_new_episodes(state, config)

        if not episodes:
            log.info("No new episodes found — nothing to do.")
            state["last_run"] = datetime.now().isoformat()
            save_state(state)
            return

        log.info("Loading shared WhisperX resources once for this run...")
        shared_resources = load_shared_whisperx_resources(config)
        if shared_resources.get("error"):
            raise shared_resources["error"]
        log.info(
            "Shared resource warmup ready: "
            f"model={shared_resources.get('model_load_s', 0):.1f}s | "
            f"align={shared_resources.get('align_load_s', 0):.1f}s | "
            f"warmup={shared_resources.get('warmup_s', 0):.1f}s"
        )

        for ep in episodes:
            ep_start = time.time()
            log.info(f"\nProcessing: {ep['title']} ({ep['published']})")
            audio_path = None

            try:
                # ── Phase 1: Download / cache audio ───────────────────────────
                log.info("  Starting episode with shared model/alignment resources already warm")
                audio_path, t_download, size_mb = download_audio(ep["audio_url"], ep["title"], config)
                if t_download > 0:
                    log.info(f"  ⏱ Phase 1 (Download):       {t_download:.1f}s ({size_mb:.1f} MB)")
                else:
                    log.info(f"  ⏱ Phase 1 (Cached):         0.0s ({size_mb:.1f} MB)")
                log_resources("Download")

                # ── Phase 2: WhisperX Transcription ───────────────────────────
                raw_retranscribe_used = False
                raw_redownload_used = False
                while True:
                    transcript_path, plain_text, whisperx_metrics, aligned_segments, audio_for_diarization = transcribe_with_whisperx(
                        audio_path, ep["title"], config,
                        podcast_name=ep.get("podcast"),
                        shared_resources=shared_resources,
                        episode_title=ep.get("title", ""),
                        episode_summary=ep.get("summary", "")
                    )

                    raw_junk_reason = detect_obvious_raw_transcript_junk(plain_text)
                    if not raw_junk_reason:
                        break

                    if not raw_retranscribe_used:
                        raw_retranscribe_used = True
                        log.warning(
                            "Detected obvious raw transcript junk; retranscribing once from cached audio before fallback. "
                            f"[{raw_junk_reason}]"
                        )
                        continue

                    if not raw_redownload_used:
                        raw_redownload_used = True
                        if audio_path and os.path.exists(audio_path):
                            os.remove(audio_path)
                            log.info(f"Removed cached audio before forced re-download: {audio_path}")
                        redownload_elapsed, redownload_size_mb = 0.0, size_mb
                        audio_path, redownload_elapsed, redownload_size_mb = download_audio(ep["audio_url"], ep["title"], config)
                        t_download += redownload_elapsed
                        size_mb = redownload_size_mb
                        log.info(
                            "Detected obvious raw transcript junk again; re-downloaded audio and retranscribing one final time. "
                            f"[{raw_junk_reason}]"
                        )
                        log.info(f"  ⏱ Phase 1 (Re-download):    {redownload_elapsed:.1f}s ({redownload_size_mb:.1f} MB)")
                        log_resources("Re-download")
                        continue

                    raise ValueError(
                        "Raw transcript junk persisted after retranscribe and re-download: "
                        f"{raw_junk_reason}"
                    )

                t_transcribe_total = whisperx_metrics.get("total_transcribe_s", 0)
                log.info(f"  ⏱ Phase 2 (Total):          {t_transcribe_total:.1f}s")

                # ── Phase 3: Vocabulary Corrections ──────────────────────────
                t_vocab_start = time.time()
                corrected_text, vocabulary_replacements = apply_vocabulary_corrections(
                    plain_text,
                    config.vocabulary_corrections,
                )
                corrected_text, dotted_letter_run_removals = remove_repeated_dotted_letter_runs(corrected_text)
                t_vocab = time.time() - t_vocab_start
                whisperx_metrics["vocabulary_corrections_s"] = round(t_vocab, 2)
                whisperx_metrics["vocabulary_corrections_count"] = sum(item["count"] for item in vocabulary_replacements)
                whisperx_metrics["dotted_letter_run_removals_count"] = len(dotted_letter_run_removals)
                if vocabulary_replacements:
                    replacement_summary = ", ".join(
                        f"{item['from']}→{item['to']} x{item['count']}" for item in vocabulary_replacements
                    )
                    log.info(f"  Applied vocabulary corrections: {replacement_summary}")
                else:
                    log.info("  Applied vocabulary corrections: none")
                if dotted_letter_run_removals:
                    removal_summary = ", ".join(
                        f"{item['letter']}. x{item['count']}" for item in dotted_letter_run_removals
                    )
                    log.info(f"  Removed repeated dotted-letter junk runs: {removal_summary}")
                else:
                    log.info("  Removed repeated dotted-letter junk runs: none")
                log.info(f"  ⏱ Phase 3 (Vocabulary):     {t_vocab:.1f}s")

                # ── Phase 4: Note Podcast Start in Transcript ─────────────────
                t_marker_start = time.time()
                marked_text, inserted_marker = insert_podcast_start_marker(
                    corrected_text,
                    config.closing_ad_phrases,
                    config.opening_catch_phrases,
                    config.opening_ad_phrases,
                    config,
                )
                t_marker = time.time() - t_marker_start
                whisperx_metrics["podcast_start_marker_inserted"] = inserted_marker
                whisperx_metrics["podcast_start_marker_s"] = round(t_marker, 2)
                log.info(f"  ⏱ Phase 4 (Podcast Start):  {t_marker:.1f}s")

                # Save post-processed transcript starting after the last PODCAST START marker
                safe_title = make_safe_title(ep["title"], config)
                marker_matches = list(re.finditer(r"podcast start", marked_text, re.IGNORECASE))
                post_processed_text = marked_text
                if marker_matches:
                    post_processed_text = marked_text[marker_matches[-1].end():].lstrip()
                post_processed_txt_path = os.path.join(TRANSCRIPT_DIR, f"post_processed_{safe_title}.txt")
                if config.test_mode:
                    with open(post_processed_txt_path, "w", encoding="utf-8") as f:
                        f.write(post_processed_text)

                # ── Phase 5: LLM Cleanup ──────────────────────────────────────
                t_llm_start = time.time()
                plain_transcript_text, llm_usage = llm_cleanup(post_processed_text, config)
                t_llm = time.time() - t_llm_start
                log.info(f"  ⏱ Phase 5 (LLM cleanup):    {t_llm:.1f}s")
                log_resources("LLM Cleanup")

                # Save primary cleaned transcript
                plain_transcript_path = os.path.join(TRANSCRIPT_DIR, f"llm_cleaned_transcript_{safe_title}.txt")
                cleaned_txt_path  = os.path.join(TRANSCRIPT_DIR, f"final_cleaned_{safe_title}.txt")
                cleaned_md_path   = os.path.join(TRANSCRIPT_DIR, f"final_cleaned_{safe_title}.md")
                if config.test_mode:
                    with open(plain_transcript_path, "w", encoding="utf-8") as f:
                        f.write(plain_transcript_text)

                inferred_speakers = infer_speaker_list(
                    ep.get("podcast", ""),
                    episode_title=ep.get("title", ""),
                    episode_summary=ep.get("summary", ""),
                    transcript_text=plain_transcript_text,
                )
                cleaned_text = build_final_text_transcript(ep, plain_transcript_text, inferred_speakers)
                cleaned_markdown = build_final_markdown_transcript(
                    ep,
                    plain_transcript_text,
                    inferred_speakers,
                )
                with open(cleaned_txt_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                with open(cleaned_md_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_markdown)

                # ── Phase 6: Email ────────────────────────────────────────────
                t_email = send_transcript_email(
                    ep,
                    plain_transcript_text,
                    cleaned_txt_path,
                    final_text=cleaned_text,
                    config=config,
                )
                log.info(f"  ⏱ Phase 6 (Email):          {t_email:.1f}s")

                # ── Summary ───────────────────────────────────────────────────
                ep_elapsed = time.time() - ep_start
                log.info(f"  {'─' * 45}")
                log.info(f"  ⏱ Total episode time:       {ep_elapsed:.1f}s ({ep_elapsed/60:.1f} min)")

                remove_episode_from_retry_queue(state, ep["guid"])
                state["processed_guids"].add(ep["guid"])
                state["episodes"].append({
                    "guid": ep["guid"],
                    "title": ep["title"],
                    "published": ep["published"],
                    "published_at": ep.get("published_at"),
                    "processed_at": datetime.now().isoformat(),
                    "timing": {
                        "download_s":   round(t_download, 1),
                        "transcribe_total_s": round(t_transcribe_total, 1),
                        "model_load_s": whisperx_metrics.get("model_load_s", 0),
                        "transcribe_s": whisperx_metrics.get("transcribe_s", 0),
                        "align_s":      whisperx_metrics.get("align_s", 0),
                        "llm_s":        round(t_llm, 1),
                        "podcast_start_s": round(t_marker, 1),
                        "email_s":      round(t_email, 1),
                        "total_s":      round(ep_elapsed, 1)
                    },
                    "whisperx_metrics": whisperx_metrics,
                    "llm_usage": llm_usage
                })
                save_state(state)
                log.info(f"✅ Episode complete: [{ep['podcast']}] {ep['title']}")

            except Exception as e:
                log.error(f"Failed to process episode '{ep['title']}': {e}", exc_info=True)
                deleted_audio = False
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
                    deleted_audio = True
                    log.info(f"Deleted source audio after failure so next run will re-download it: {audio_path}")

                queue_episode_for_retry(state, ep, e)
                save_state(state)
                send_alert_email(
                    f"⚠️ Transcript Agent Error: {ep['title']}",
                    (
                        f"Failed to process episode:\n{ep['title']}\n\n"
                        f"Podcast:\n{ep['podcast']}\n\n"
                        f"Error:\n{e}\n\n"
                        f"Queued for retry on next run: yes\n"
                        f"Deleted cached source audio: {'yes' if deleted_audio else 'no'}\n"
                        f"Continuing to next episode in this run: yes\n\n"
                        f"Check {LOG_FILE}"
                    ),
                    config=config,
                )
            finally:
                # Audio retained for AUDIO_RETENTION_HOURS (cleanup happens at start of run)
                if audio_path and os.path.exists(audio_path):
                    log.info(f"Audio retained: {audio_path} (will expire in {config.audio_retention_hours}h)")

        run_elapsed = time.time() - run_start
        log.info("=" * 60)
        log.info(f"Run complete in {run_elapsed:.1f}s ({run_elapsed/60:.1f} min)")
        log.info(f"Episodes processed: {len(episodes)}")
        log.info("=" * 60)

        state["last_run"] = datetime.now().isoformat()
        save_state(state)

    except Exception as e:
        log.error(f"Hard failure: {e}", exc_info=True)
        send_alert_email(
            "⚠️ Transcript Agent Hard Failure",
            f"The Compound transcript agent failed:\n\n{e}\n\nCheck {LOG_FILE}",
            config=config,
        )
        raise
    finally:
        release_shared_whisperx_resources(shared_resources)
        if config.test_mode:
            reset_test_mode_settings()
        release_run_lock(run_lock)

if __name__ == "__main__":
    main()
