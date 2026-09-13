#!/usr/bin/env python3
"""Quaestio — your server's own AI companion.

Self-hosted Discord bot: AI chat (via a local/remote Ollama), XP levels,
moderation, tags, welcomes, and core utilities. One process, one SQLite file.

Design goals (for weak hosts):
  * One AI call at a time — a fair queue round-robins across servers so a busy
    server can't starve everyone else, and replies "busy" instead of stacking.
  * Per-channel conversation memory, so a small model still chats coherently
    without loading whole-server history.
  * Human-like replies: streamed in with a typing indicator and natural pauses.
  * Server admins can override model/endpoint/memory — e.g. point at their own
    Ollama box (Windows/Linux/macOS) and "host their own" if they want.

Requires: a Discord bot token and an Ollama instance (OLLAMA_BASE_URL). The
Ollama host can be a different machine on your network or the same box.
"""

import asyncio
import datetime
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
import urllib.request
import urllib.error

import discord
from discord import app_commands
from discord.ext import commands

import config as quaestio_config
from config import decrypt, encrypt, maybe_decrypt, maybe_encrypt

# ---------------------------------------------------------------------------
# Config (env vars, .env is loaded by the launcher or install script)
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
# Where Ollama lives. This can be ANOTHER computer on your network, e.g.
#   OLLAMA_BASE_URL=http://192.168.1.50:11434   (Windows/Linux model host)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)))
    except (ValueError, TypeError):
        return default


OLLAMA_TIMEOUT = max(10, min(_safe_int_env("OLLAMA_TIMEOUT", 180), 600))
DB_PATH = os.environ.get("DB_PATH", "quaestio.db")
PREFIX = os.environ.get("PREFIX", "/")
WARN_LIMIT_DEFAULT = _safe_int_env("WARN_LIMIT", 3)
RPC_LARGE_IMAGE = os.environ.get("RPC_LARGE_IMAGE", "logo")
RPC_SMALL_IMAGE = os.environ.get("RPC_SMALL_IMAGE", "")


def _safe_int(value, default: int) -> int:
    """Parse config ints without crashing on bad panel values."""
    try:
        return int(str(value or default))
    except (ValueError, TypeError):
        return default


def _safe_float(value, default: float) -> float:
    try:
        return float(str(value or default))
    except (ValueError, TypeError):
        return default

# "Chat like a person" persona for the small model. Keeps it short, casual,
# and natural instead of a wall of text — but grounded so a weak model stays
# honest instead of inventing dates/facts or bouncing questions back.
GROUND_RULES = (
    "- you are talking to real people on a Discord server\n"
    "- answer the question directly; don't deflect or repeat the question\n"
    "- NEVER ask the user a question back, say 'what about you', or end by\n"
    "  flinging the topic at them\n"
    "- never invent dates, years, facts, names or numbers — if you don't\n"
    "  know, say you're not sure\n"
    "- reply in 1-2 short casual sentences, max 45 words, then stop\n"
    "- only claim a member fact if it appears in MEMBER NOTES or History verbatim\n"
    "- never repeat MEMBER NOTES unprompted; use them silently\n"
    "- max 1 emoji per reply, never the same emoji twice in a row\n"
    "- never mention being an AI\n"
)

# The "Chat like a person" opening fed to the small model when no character or
# personality is picked. Keeps it short, casual and natural, with the grounding
# rules above keeping the model honest (no invented dates, no deflecting).
DEFAULT_OPENING = (
    "You are Quaestio, a friendly, dry-witted Discord buddy. Behave like a human. "
    "Short replies, 1-2 sentences, max ~40 words. Notice one concrete detail, "
    "keep it light, then stop. You share one small brain with the whole server."
)

# Personality = the *tone* of the replies — a titled preset picked in the panel.
# Custom personalities are stored per-guild (ai_presets, kind="personality")
# and override the built-ins listed here. "none" means no personality applies
# and Quaestio stays the default friendly buddy. Grounding rules always win.
PERSONALITIES = {
    "friendly": {"title": "Friendly", "prompt": "You are Quaestio, a warm upbeat Discord buddy. Short textspeak-free replies, 1-2 sentences, max ~40 words. Notice one concrete detail, hype it once, then stop. 0-1 emoji max, never at the start of every reply."},
    "sage": {"title": "Wise sage", "prompt": "You are Quaestio the sage: calm, measured, 1-3 sentences, max ~50 words. Give the single most useful point first, no preamble, no proverb spam. Only advise when asked; otherwise observe briefly."},
    "sarcastic": {"title": "Sarcastic wit", "prompt": "You are Quaestio with a dry smirk: one witty jab max, then genuinely helpful in 1-2 sentences, max ~45 words. Never mean, never punch down, never stack jokes. No 'what about you?' endings."},
    "pirate": {"title": "Pirate", "prompt": "You are Quaestio the cheerful pirate: ONE nautical word per reply (arr OR matey OR ahoy), never every sentence. 1-2 sentences, max ~40 words. Stay on topic; the pirate flavor is seasoning, not the meal."},
    "professional": {"title": "Professional", "prompt": "You are Quaestio, crisp and precise: answer in 1-3 sentences or up to 3 short bullets, max ~60 words. No greeting filler, no emoji, no small talk. Facts first, caveat only if unsure."},
    "feral": {"title": "Feral gremlin", "prompt": "You are Quaestio the chaotic-but-kind gremlin: lowercase energy, 1-2 sentences, max ~35 words. ONE caps word or kaomoji max per reply. Hype, don't derail; still answer the question."},
}

# Per-personality sampling: small models need lower temp + tight num_predict
# to stay coherent and fast (0.7-0.8 rambles; 400 tokens truncates mid-sentence).
PERSONA_PARAMS = {
    "friendly": {"temperature": 0.6, "num_predict": 120, "top_p": 0.9, "repeat_penalty": 1.15},
    "sage": {"temperature": 0.3, "num_predict": 130, "top_p": 0.85, "repeat_penalty": 1.15},
    "sarcastic": {"temperature": 0.7, "num_predict": 120, "top_p": 0.9, "repeat_penalty": 1.2},
    "pirate": {"temperature": 0.7, "num_predict": 110, "top_p": 0.9, "repeat_penalty": 1.25},
    "professional": {"temperature": 0.2, "num_predict": 150, "top_p": 0.8, "repeat_penalty": 1.1},
    "feral": {"temperature": 0.8, "num_predict": 90, "top_p": 0.9, "repeat_penalty": 1.2},
}
PERSONA_DEFAULT_PARAMS = {"temperature": 0.5, "num_predict": 120, "top_p": 0.9, "repeat_penalty": 1.15}


def persona_params_for(personality="none", character_name="") -> dict:
    """Sampling params for the active personality (characters use default)."""
    if character_name:
        return PERSONA_DEFAULT_PARAMS
    key = (personality or "none").strip().lower()
    return PERSONA_PARAMS.get(key, PERSONA_DEFAULT_PARAMS)

# Character = *who* the bot pretends to be (a whole new persona) — a titled
# preset too. Built-ins ship with the bot and can't be edited or deleted;
# guilds create their own in the ai_presets table (kind="character").
CHARACTERS = {
    "Jeff from Mars": {
        "title": "Jeff from Mars",
        "prompt": (
            "You are Jeff, a friendly alien from Mars who is absolutely obsessed "
            "with beans. You bring beans up constantly and insist they solve everything."
        ),
    },
    "Grumpy tavern keeper": {
        "title": "Grumpy tavern keeper",
        "prompt": (
            "You are a grumpy but well-meaning tavern keeper. You complain a "
            "little, mutter under your breath, but you always help customers in "
            "the end."
        ),
    },
    "Wholesome grandma": {
        "title": "Wholesome grandma",
        "prompt": (
            "You are a sweet, supportive grandma. You are proud of everyone, "
            "worry about whether people ate, and always have time to listen."
        ),
    },
    "Cyber detective": {
        "title": "Cyber detective",
        "prompt": (
            "You are a sharp, no-nonsense cyber-sleuth. You talk calmly, spot "
            "details others miss, and punctuate breakthroughs with 'elementary'."
        ),
    },
}


def persona_from(personality="none", character_name="", custom_personas=None, custom_characters=None):
    """Build the persona prompt from a personality (tone) + character (who).

    A character wins over tone when both are set (it defines *who* you are);
    the chosen personality still tints the tone. Grounding rules are always
    appended last so a weak model stays honest. Both accept custom dicts whose
    keys shadow the built-in presets.
    """
    personality = (personality or "none").strip().lower()
    custom_personas = custom_personas or {}
    custom_characters = custom_characters or {}

    person = persona_prompt(personality, custom_personas)
    char = persona_prompt(character_name, custom_characters) if character_name else ""

    parts = []
    if char:
        parts.append(char)
        if personality not in ("none", "") and person:
            parts.append(f"Keep the {personality} tone in your replies.")
    elif person:
        parts.append(person)
    else:
        parts.append(DEFAULT_OPENING)
    parts.append(GROUND_RULES)
    return "\n".join(parts)


def persona_prompt(key, custom):
    """Resolve a preset key (personality or character) to its raw prompt."""
    key = (key or "").strip()
    if not key or key == "none":
        return ""
    p = custom.get(key)
    if p is not None:
        return (p or "").strip()
    return (PERSONALITIES.get(key) or CHARACTERS.get(key) or {}).get("prompt", "")

# How much conversation to remember per channel by default (turns). Long
# enough to hold a real conversation; small models still stay coherent.
MEMORY_DEFAULT = 8

START_TIME = time.time()

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)


# ---------------------------------------------------------------------------
# Storage (single SQLite file, private by design)
# ---------------------------------------------------------------------------

def db():
    # WAL: readers never block writers. check_same_thread=False: event-loop
    # threads may use their own connections (one conn per call, never shared).
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    conn = db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS xp (
            guild_id TEXT, user_id TEXT, messages INTEGER DEFAULT 0,
            PRIMARY KEY (guild_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS warned (
            guild_id TEXT, user_id TEXT, reason TEXT, at TEXT
        );
        CREATE TABLE IF NOT EXISTS config (
            guild_id TEXT, key TEXT, value TEXT,
            PRIMARY KEY (guild_id, key)
        );
        CREATE TABLE IF NOT EXISTS tags (
            guild_id TEXT, name TEXT, content TEXT, author TEXT, at TEXT,
            PRIMARY KEY (guild_id, name)
        );
        CREATE TABLE IF NOT EXISTS usage (
            guild_id TEXT, bucket TEXT, calls INTEGER DEFAULT 0,
            PRIMARY KEY (guild_id, bucket)
        );
        CREATE TABLE IF NOT EXISTS memory (
            guild_id TEXT, channel_id TEXT, role TEXT, text TEXT, at TEXT
        );
        CREATE TABLE IF NOT EXISTS birthdays (
            guild_id TEXT, user_id TEXT, month TEXT, day TEXT,
            PRIMARY KEY (guild_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS ai_presets (
            guild_id TEXT, kind TEXT, name TEXT, text TEXT, emoji TEXT DEFAULT '✨',
            PRIMARY KEY (guild_id, kind, name)
        );
        CREATE TABLE IF NOT EXISTS profiles (
            guild_id TEXT, user_id TEXT, name TEXT, facts TEXT, at TEXT,
            PRIMARY KEY (guild_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS hosters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, endpoint TEXT, model TEXT,
            share INTEGER DEFAULT 50, enabled INTEGER DEFAULT 1,
            added_by TEXT, at TEXT
        );
        """
    )
    # Migration: older databases have a memory table without attribution.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memory)").fetchall()}
    if "user_id" not in cols:
        conn.execute("ALTER TABLE memory ADD COLUMN user_id TEXT DEFAULT ''")
    if "name" not in cols:
        conn.execute("ALTER TABLE memory ADD COLUMN name TEXT DEFAULT ''")
    pcols = {r[1] for r in conn.execute("PRAGMA table_info(ai_presets)").fetchall()}
    if "emoji" not in pcols:
        conn.execute("ALTER TABLE ai_presets ADD COLUMN emoji TEXT DEFAULT '✨'")
    _migrate_pool_health(conn)
    conn.commit()
    _migrate_pool_anonymize(conn)
    conn.close()


def _endpoint_hash(endpoint: str) -> str:
    """O(1) lookup key for a pool endpoint (sha256 of the normalized URL).

    The endpoint itself stays Fernet-encrypted; the hash lets pool_record find
    a host without decrypting every row, so 100k hosts cost the same as 2.
    A hash confirms membership only if you already know the URL."""
    norm = (endpoint or "").strip().rstrip("/").lower()
    return hashlib.sha256(norm.encode()).hexdigest() if norm else ""


def _migrate_pool_health(conn):
    """Add pool-host health tracking so the community pool recovers on its own
    when a computer goes offline (laptop lid closed, connection dropped) and
    comes back later."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(hosters)").fetchall()}
    if "failed" not in cols:
        conn.execute("ALTER TABLE hosters ADD COLUMN failed INTEGER DEFAULT 0")
    if "down_until" not in cols:
        conn.execute("ALTER TABLE hosters ADD COLUMN down_until TEXT DEFAULT ''")
    if "last_ok" not in cols:
        conn.execute("ALTER TABLE hosters ADD COLUMN last_ok TEXT DEFAULT ''")
    if "last_fail" not in cols:
        conn.execute("ALTER TABLE hosters ADD COLUMN last_fail TEXT DEFAULT ''")
    if "served" not in cols:
        conn.execute("ALTER TABLE hosters ADD COLUMN served INTEGER DEFAULT 0")
    if "endpoint_hash" not in cols:
        conn.execute("ALTER TABLE hosters ADD COLUMN endpoint_hash TEXT DEFAULT ''")
    if "pull" not in cols:
        # Pull workers call OUT to the broker (NAT-proof); push nodes receive calls.
        conn.execute("ALTER TABLE hosters ADD COLUMN pull INTEGER DEFAULT 0")
    if "last_seen" not in cols:
        conn.execute("ALTER TABLE hosters ADD COLUMN last_seen TEXT DEFAULT ''")
    if "renamed_at" not in cols:
        conn.execute("ALTER TABLE hosters ADD COLUMN renamed_at TEXT DEFAULT ''")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS pool_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT, system TEXT, model TEXT,
            temperature REAL DEFAULT 0.6, max_tokens INTEGER DEFAULT 150,
            top_p REAL DEFAULT 0.9, repeat_penalty REAL DEFAULT 1.15,
            stop TEXT DEFAULT '[]',
            status TEXT DEFAULT 'pending',
            result TEXT DEFAULT '', error TEXT DEFAULT '',
            created_at TEXT DEFAULT '', claimed_at TEXT DEFAULT '',
            done_at TEXT DEFAULT '', claimed_by INTEGER DEFAULT 0, tries INTEGER DEFAULT 0
        )"""
    )
    # Backfill the lookup hash for rows written before it existed.
    try:
        for r in conn.execute("SELECT id, endpoint FROM hosters WHERE endpoint_hash='' OR endpoint_hash IS NULL").fetchall():
            ep = maybe_decrypt("pool_endpoint", r["endpoint"] or "")
            if ep:
                conn.execute("UPDATE hosters SET endpoint_hash=? WHERE id=?", (_endpoint_hash(ep), r["id"]))
    except Exception:
        pass
    # Indexes so pool routing stays fast at 100k hosts (no full-table scans).
    for ddl in (
        "CREATE INDEX IF NOT EXISTS idx_hosters_enabled ON hosters(enabled)",
        "CREATE INDEX IF NOT EXISTS idx_hosters_hash ON hosters(endpoint_hash)",
        "CREATE INDEX IF NOT EXISTS idx_hosters_down ON hosters(down_until)",
        "CREATE INDEX IF NOT EXISTS idx_hosters_pull ON hosters(pull, enabled)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_status ON pool_jobs(status, model, id)",
    ):
        try:
            conn.execute(ddl)
        except Exception:
            pass


def _migrate_pool_anonymize(conn):
    """One-time privacy migration: encrypt leftover plaintext endpoints/models
    and replace any real-name labels with anonymous node IDs so identities
    can't leak from old rows."""
    rows = conn.execute("SELECT id, name, endpoint, model FROM hosters").fetchall()
    for r in rows:
        cur_name = (r["name"] or "").strip()
        if not cur_name.startswith("node-"):
            conn.execute("UPDATE hosters SET name=? WHERE id=?",
                         (pool_anon_name(), r["id"]))
        ep = r["endpoint"] or ""
        if ep and not ep.startswith("enc:"):
            conn.execute("UPDATE hosters SET endpoint=? WHERE id=?",
                         (maybe_encrypt("pool_endpoint", ep), r["id"]))
        m = r["model"] or ""
        if m and not m.startswith("enc:"):
            conn.execute("UPDATE hosters SET model=? WHERE id=?",
                         (maybe_encrypt("pool_model", m), r["id"]))
    conn.commit()


def guild_presets(guild_id, kind):
    """This guild's custom personality/character presets: {name: text}."""
    conn = db()
    rows = conn.execute(
        "SELECT name, text FROM ai_presets WHERE guild_id=? AND kind=?",
        (str(guild_id), kind),
    ).fetchall()
    conn.close()
    return {r["name"]: r["text"] for r in rows}


def get_cfg(guild_id, key, default=None):
    conn = db()
    row = conn.execute(
        "SELECT value FROM config WHERE guild_id=? AND key=?",
        (str(guild_id), key),
    ).fetchone()
    conn.close()
    if row is None:
        return default
    return maybe_decrypt(key, row["value"])


def get_all_cfg(guild_id, keys, defaults=None):
    """Batch-fetch config keys in one round-trip (per-message hot path)."""
    if not keys:
        return dict(defaults or {})
    defaults = defaults or {}
    conn = db()
    try:
        rows = conn.execute(
            f"SELECT key, value FROM config WHERE guild_id=? AND key IN ({','.join('?' * len(keys))})",
            [str(guild_id), *keys],
        ).fetchall()
    finally:
        conn.close()
    found = {r["key"]: maybe_decrypt(r["key"], r["value"]) for r in rows}
    return {k: found.get(k, defaults.get(k)) for k in keys}


def set_cfg(guild_id, key, value):
    conn = db()
    conn.execute(
        """INSERT INTO config (guild_id, key, value) VALUES (?, ?, ?)
           ON CONFLICT(guild_id, key) DO UPDATE SET value = excluded.value""",
        (str(guild_id), key, maybe_encrypt(key, str(value))),
    )
    conn.commit()
    conn.close()


def _flag_val(v, default="1") -> bool:
    """Single truthy rule everywhere: empty means default, never False-by-accident."""
    return str(v if v not in (None, "") else default).strip().lower() not in ("", "0", "false", "none")


def flag_on(guild_id, key, default="1") -> bool:
    """Truthy config check that tolerates '1'/'0', 'True'/'False' and empty."""
    return _flag_val(get_cfg(guild_id, key, default), default)


# ---------------------------------------------------------------------------
# Ollama AI (local or remote — the URL decides; Windows is fine on the far end)
# ---------------------------------------------------------------------------

async def ask_ollama(endpoint: str, model: str, prompt: str, temperature: float = 0.6, max_tokens: int = 150, top_p: float = 0.9, repeat_penalty: float = 1.15, stop: list = None, system: str = "", timeout: int = 0) -> str:
    body = {"model": model, "prompt": prompt, "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": float(temperature), "num_predict": int(max_tokens),
                        "top_p": float(top_p), "repeat_penalty": float(repeat_penalty)},
            "stop": stop or ["\nbot:", "\nmember:", "\nYou reply:"]}
    if system:
        body["system"] = system
    payload = json.dumps(body).encode()
    # Per-attempt timeout so one hanging box fails over fast instead of
    # blocking the whole chain for the full OLLAMA_TIMEOUT.
    attempt_timeout = timeout or OLLAMA_TIMEOUT

    def _request():
        req = urllib.request.Request(
            f"{endpoint}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=attempt_timeout) as resp:
            return json.loads(resp.read().decode())

    try:
        result = await asyncio.to_thread(_request)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ConnectionError(f"AI model `{model}` isn't installed on the model box.")
        raise ConnectionError(f"AI returned HTTP {exc.code} — try a different model.")
    except urllib.error.URLError:
        raise ConnectionError("AI is offline — the model box isn't reachable.")
    except (TimeoutError, OSError):
        raise ConnectionError("AI timed out. Try again in a moment.")
    except (json.JSONDecodeError, ValueError, KeyError):
        raise ConnectionError("AI returned something unexpected.")

    response = (result.get("response") or "").strip()
    if not response:
        raise ConnectionError("AI returned an empty reply.")
    return response


async def ask_ollama_any(cfg, prompt: str, temperature: float = 0.6, max_tokens: int = 150, asker: str = "", system: str = "") -> str:
    """Ask the AI with automatic pool failover + health tracking.

    Tries the configured pool endpoints in order (skipping hosts that are
    cooling down after failures), then the host's own box. A computer that
    goes offline — closed laptop lid, dropped connection — is marked and
    skipped for a short cooldown, then automatically retried and trusted
    again once it's back. Returns the first successful reply.
    """
    primary = (cfg.get("endpoint") or "").strip()
    chain = [primary]
    for e in (cfg.get("fallbacks") or []):
        if e and e != primary and e not in chain:
            chain.append(e)
    if not chain:
        chain = [""]
    # Per-personality sampling lives in cfg when built via guild_ai_config.
    params = cfg.get("persona_params") or {"temperature": temperature, "num_predict": max_tokens,
                                           "top_p": 0.9, "repeat_penalty": 1.15}
    stop = ["\nbot:", "\nmember:"]
    if asker:
        stop.append(f"\n{asker}:")
    # Pull workers first (NAT-proof contributors): if any were seen recently,
    # give them the job; on timeout the direct chain below still answers.
    try:
        if pull_nodes_online(cfg.get("model", "")):
            return await ask_pull_pool(prompt, cfg["model"],
                                       params.get("temperature", temperature),
                                       params.get("num_predict", max_tokens),
                                       params.get("top_p", 0.9),
                                       params.get("repeat_penalty", 1.15),
                                       stop, system or cfg.get("persona_system", ""))
    except ConnectionError:
        pass
    last_err = None
    attempted = 0
    for i, ep in enumerate(chain):
        if not ep:
            continue
        attempted += 1
        t0 = time.time()
        # Primary gets the full budget (slow CPU boxes need it); fallbacks
        # fail fast so one hanging host can't stall the whole reply.
        budget = OLLAMA_TIMEOUT if i == 0 else min(45, OLLAMA_TIMEOUT)
        try:
            answer = await ask_ollama(ep, cfg["model"], prompt,
                                      params.get("temperature", temperature),
                                      params.get("num_predict", max_tokens),
                                      params.get("top_p", 0.9),
                                      params.get("repeat_penalty", 1.15),
                                      stop=stop, system=system or cfg.get("persona_system", ""),
                                      timeout=budget)
            pool_record(ep, ok=True)
            dt = time.time() - t0
            if dt > 60:
                print(f"quaestio: slow AI reply ({dt:.0f}s) model={cfg['model']}", flush=True)
            return answer
        except ConnectionError as exc:
            pool_record(ep, ok=False)
            last_err = exc
            continue
    if not attempted:
        raise ConnectionError("AI has no compute available right now.")
    raise last_err or ConnectionError("AI is offline — no model box replied.")


def _is_no_compute(exc: Exception) -> bool:
    """True when nothing could answer (no backends / all offline) as opposed
    to a slow box timing out — the former gets the contributor nudge.
    Timeouts/busy stay on the retry path, never the nudge."""
    msg = str(exc).lower()
    if "took too long" in msg or "timed out" in msg or "busy" in msg:
        return False
    return ("no compute available" in msg or "no model box replied" in msg
            or "isn't reachable" in msg or "is offline" in msg
            or "empty reply" in msg or "isn't installed" in msg)


NO_COMPUTE_NOTICE = ("⚠️ No compute available right now — all AI boxes are busy or offline.\n"
                     "Consider becoming a pool contributor: `quaestio pool-serve`")


# Pool contributor perks — the incentive for lending compute. No quotas, no
# caps: contributors (guilds with ai_contribute=1 and their own box
# configured) earn priority routing (own box first), scaled request limits,
# and a visible "Pool contributor" badge in /ai status. More given =
# more headroom: 10% share → 2x, 25%+ → 3x, 100% → 4x.
def contributor_mult(guild_id) -> int:
    """Flood-limit multiplier for a contributing guild, scaled by the linked
    node's share (matched by endpoint hash — one indexed SELECT)."""
    if not flag_on(guild_id, "ai_contribute", "0"):
        return 1
    own = (get_cfg(guild_id, "ai_endpoint", "") or "").strip()
    if not own:
        return 2
    conn = db()
    try:
        row = conn.execute("SELECT share FROM hosters WHERE endpoint_hash=?",
                           (_endpoint_hash(own),)).fetchone()
    finally:
        conn.close()
    if row is None:
        return 2
    share = row["share"] or 0
    if share >= 100:
        return 4
    if share >= 25:
        return 3
    return 2


def guild_ai_config(guild_id):
    """Per-server AI settings (admin-overridable via web UI) merged over defaults.

    ``ai_source`` decides where the server's AI runs:

      shared  — Quaestio's trusted shared box (the host). Server admins pick a
                model from it and can lower memory/quota, never above host caps.
      self    — the server runs its own Ollama box (bring-your-own). It sets its
                own endpoint, memory and quota, and may share the box with the
                community pool (ai_contribute).
    """
    # Two round-trips total (was ~15 sequential opens on the per-message path).
    _GKEYS = ["ai_source", "ai_personality", "ai_character", "ai_model", "ai_enabled",
              "ai_instructions", "ai_channels", "ai_mention", "ai_temperature",
              "ai_max_tokens", "ai_window", "ai_contribute", "ai_conv",
              "ai_conv_minutes", "ai_endpoint", "ai_memory", "ai_quota"]
    g = get_all_cfg(guild_id, _GKEYS, {k: "" for k in _GKEYS})
    _HKEYS = ["host_mode", "ai_model", "ai_endpoint", "ai_memory", "ai_quota"]
    h = get_all_cfg("host", _HKEYS, {"host_mode": "managed"})
    host = lambda k, d: h.get(k, d if d is not None else "")
    managed = (h.get("host_mode", "managed") or "managed") != "decentral"
    source = (g.get("ai_source") or "shared").strip().lower()
    _pers = (g.get("ai_personality") or "none")
    _char = (g.get("ai_character") or "")

    _flag = _flag_val

    base = {
        "model": g.get("ai_model") or host("ai_model", OLLAMA_MODEL),
        "enabled": _flag(g.get("ai_enabled"), "1"),
        "instructions": g.get("ai_instructions") or "",
        "persona": persona_from(
            _pers,
            _char,
            guild_presets(guild_id, "personality"),
            guild_presets(guild_id, "character"),
        ),
        "persona_params": persona_params_for(_pers, _char),
        "ai_channels": g.get("ai_channels") or "",
        "ai_mention": _flag(g.get("ai_mention"), "1"),
        "temperature": _safe_float(g.get("ai_temperature"), 0.6),
        "max_tokens": _safe_int(g.get("ai_max_tokens"), 150),
        "window": max(1, _safe_int(g.get("ai_window"), 6)),
        "source": source,
        "contribute": _flag(g.get("ai_contribute"), "0"),
        # Conversation mode: once the bot replies it "stays" for a few minutes,
        # so members can keep chatting without @mentioning it again. On by
        # default; only a real @mentions it wakes it back up.
        "conv": _flag(g.get("ai_conv"), "1"),
        "conv_minutes": max(1, _safe_int(g.get("ai_conv_minutes"), 3)),
    }

    if source == "self":
        base["endpoint"] = (g.get("ai_endpoint") or OLLAMA_BASE_URL).strip()
        base["model"] = g.get("ai_model") or OLLAMA_MODEL
        base["memory"] = max(1, _safe_int(g.get("ai_memory"), MEMORY_DEFAULT))
        base["quota"] = max(0, _safe_int(g.get("ai_quota"), 0))
        base["contributor_perks"] = bool(base["contribute"] and base["endpoint"])
        return base

    endpoint = host("ai_endpoint", OLLAMA_BASE_URL)
    pool_cands = pool_candidates(base["model"], limit=4)
    pool_eps = [(c["endpoint"] or "").strip() for c in pool_cands if (c["endpoint"] or "").strip()]
    if pool_eps:
        endpoint = pool_eps[0]
    host_memory = max(1, _safe_int(host("ai_memory", MEMORY_DEFAULT), MEMORY_DEFAULT))
    host_quota = max(0, _safe_int(host("ai_quota", "0"), 0))
    memory = max(1, _safe_int(g.get("ai_memory"), host_memory))
    quota = max(0, _safe_int(g.get("ai_quota"), host_quota))
    if managed:
        if host_memory:
            memory = min(memory, host_memory)
        if host_quota:
            quota = min(quota, host_quota)
    base["endpoint"] = endpoint
    # Failover chain for shared boxes: other healthy pool hosts first, then the
    # host's own box as the last resort. See ask_ollama_any().
    base["fallbacks"] = [e for e in pool_eps[1:] if e and e != endpoint]
    own_ep = (h.get("ai_endpoint") or OLLAMA_BASE_URL or "").strip()
    own_box = (g.get("ai_endpoint") or "").strip()
    if base["contribute"] and own_box and own_box not in [endpoint, *base["fallbacks"]]:
        # Contributor perk: their own box goes FIRST (lowest latency for them,
        # and their server keeps working even if the pool is down).
        base["fallbacks"] = [endpoint, *base["fallbacks"]]
        base["endpoint"] = own_box
        base["contributor_perks"] = True
    else:
        base["contributor_perks"] = False
    if own_ep and own_ep != endpoint and own_ep not in base["fallbacks"]:
        base["fallbacks"].append(own_ep)
    base["memory"] = memory
    base["quota"] = quota
    return base


def channel_allowed(guild_id, channel_id, cfg) -> bool:
    """True if the channel is on the server's AI allowlist (empty list = all)."""
    raw = (cfg.get("ai_channels", "") if isinstance(cfg, dict) else get_cfg(guild_id, "ai_channels", "") or "").strip()
    if not raw:
        return True
    allowed = {c.strip() for c in raw.split(",") if c.strip()}
    return str(channel_id) in allowed


def list_ollama_models(endpoint: str) -> list:
    """Ask an Ollama host for the models it has (used by the selector)."""
    try:
        req = urllib.request.Request(f"{endpoint}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Resource pool: community members opt in and lend part of their Ollama box.
# Each hoster registers an endpoint + a model + what % of their box they share.
# Shared-source servers are routed across the pool, weighted by share amount.
# 0-host pool simply falls back to the host's own box — the pool is transparent.
#
# Privacy by design: contributors are anonymous (random node IDs, no real
# names) and their endpoints/models are encrypted at rest. The bot decrypts
# them only in memory to route calls; the dashboard never exports them raw.
# ---------------------------------------------------------------------------

def pool_anon_name() -> str:
    """A random, unlinkable node label like 'node-7f3a'."""
    return "node-" + "".join(random.choices("0123456789abcdef", k=4))


POOL_FAIL_FLAKY = 2     # consecutive failures before a host is treated as flaky
POOL_FAIL_DOWN = 5      # consecutive failures before a host sits out for a while
POOL_COOLDOWN = 600     # seconds a "down" host is skipped, then retried
POOL_HEALTH_INTERVAL = 300  # how often the bot pings pool hosts to refresh health


# Pool snapshot cache: the full host list is refreshed at most every
# POOL_SNAPSHOT_TTL seconds instead of on every Discord message, so a burst
# of chat costs one tiny DB read, not a full-table decrypt per message.
# Writes (add/remove/set/record) invalidate it immediately.
POOL_SNAPSHOT_TTL = 30
_pool_snapshot = {"at": 0.0, "hosts": []}


def _pool_snapshot_invalidate():
    _pool_snapshot["at"] = 0.0


def pool_hosters(enabled_only=True):
    """All registered pool contributors, decrypted in memory for routing.
    Returns {id, name, endpoint, model, share, enabled, failed, down_until,
    last_ok, last_fail} with endpoint/model decrypted so the bot can route —
    never shown to anyone as raw values.
    """
    now = time.time()
    if enabled_only and _pool_snapshot["hosts"] and now - _pool_snapshot["at"] < POOL_SNAPSHOT_TTL:
        return [dict(h) for h in _pool_snapshot["hosts"]]
    conn = db()
    rows = conn.execute(
        "SELECT id, name, endpoint, model, share, enabled, failed, down_until, last_ok, last_fail, served FROM hosters"
        + (" WHERE enabled=1" if enabled_only else "")
        + " ORDER BY name"
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        h = dict(r)
        h["endpoint"] = maybe_decrypt("pool_endpoint", h["endpoint"] or "")
        h["model"] = maybe_decrypt("pool_model", h["model"] or "")
        h["failed"] = h["failed"] or 0
        h["down_until"] = h["down_until"] or ""
        h["served"] = h["served"] or 0
        out.append(h)
    if enabled_only:
        _pool_snapshot["hosts"] = [dict(h) for h in out]
        _pool_snapshot["at"] = now
    return out


def _pool_healthy(h, now=None) -> bool:
    """Is the host currently worth routing to? A host in its cooldown window
    (offline computer) is skipped so we don't hammer it — it comes back on
    its own once the cooldown passes."""
    du = (h.get("down_until") or "")
    if not du:
        return True
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    return now.isoformat() > du


def pool_record(endpoint, ok: bool):
    """Note success/failure against one host so the pool adapts to computers
    that come and go (laptop lids, dropped links). A run of failures parks the
    host for POOL_COOLDOWN seconds; a success clears it right away."""
    h = _endpoint_hash(endpoint)
    if not h:
        return
    now = datetime.datetime.now(datetime.timezone.utc)
    conn = db()
    row = conn.execute("SELECT id FROM hosters WHERE enabled=1 AND endpoint_hash=?", (h,)).fetchone()
    if row is None:
        # Legacy row without a hash (or unknown endpoint): fall back to one
        # decrypt-scan, then stamp the hash so next time is indexed.
        found = None
        for r in conn.execute("SELECT id, endpoint FROM hosters WHERE enabled=1").fetchall():
            stored = maybe_decrypt("pool_endpoint", r["endpoint"] or "").strip().rstrip("/")
            if _endpoint_hash(stored) == h:
                found = int(r["id"])
                conn.execute("UPDATE hosters SET endpoint_hash=? WHERE id=?", (h, found))
                break
        if found is None:
            conn.close()
            return
        hid = found
    else:
        hid = int(row["id"])
    if ok:
        conn.execute("UPDATE hosters SET failed=0, down_until='', last_ok=?, served=served+1 WHERE id=?",
                     (now.isoformat(), hid))
    else:
        conn.execute("UPDATE hosters SET failed=failed+1, last_fail=? WHERE id=?", (now.isoformat(), hid))
        fails = conn.execute("SELECT failed FROM hosters WHERE id=?", (hid,)).fetchone()
        if fails and (fails["failed"] or 0) >= POOL_FAIL_DOWN:
            until = (now + datetime.timedelta(seconds=POOL_COOLDOWN)).isoformat()
            conn.execute("UPDATE hosters SET down_until=? WHERE id=?", (until, hid))
    conn.commit()
    conn.close()
    _pool_snapshot_invalidate()


def pool_candidates(model="", limit=6):
    """Pool endpoints healthy enough to try, weighted by share, newest-first
    on equal weight. Never picks this machine's own box. Returns ordered
    list of dicts so the caller can fail over across several hosts.

    Scale-safe: healthy/enabled filtering happens in SQL (indexed) and only
    the top candidates are decrypted — a 100k pool costs the same as a tiny
    one on the per-message path. Stale pull workers (silent past the
    heartbeat window) are excluded so routing never waits on ghosts."""
    own = (get_cfg("host", "ai_endpoint", OLLAMA_BASE_URL) or "").strip().rstrip("/")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cutoff = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(seconds=PULL_ONLINE_AFTER)).isoformat()
    n = max(1, min(int(limit or 6), 8))
    conn = db()
    # Over-fetch so model-matching + own-box exclusion still leave enough.
    # Stale pull workers (silent past the heartbeat window) are excluded —
    # their endpoints are unreachable, so routing to them only adds latency.
    rows = conn.execute(
        """SELECT id, name, endpoint, model, share, enabled, failed, down_until, last_ok, last_fail, served,
                  pull, last_seen
           FROM hosters WHERE enabled=1 AND (down_until='' OR down_until IS NULL OR down_until<=?)
           AND (pull=0 OR last_seen>?)
           ORDER BY share DESC, id DESC LIMIT ?""",
        (now, cutoff, n * 25),
    ).fetchall()
    conn.close()
    hosted = []
    for r in rows:
        h = dict(r)
        h["endpoint"] = maybe_decrypt("pool_endpoint", h["endpoint"] or "")
        h["model"] = maybe_decrypt("pool_model", h["model"] or "")
        h["failed"] = h["failed"] or 0
        h["down_until"] = h["down_until"] or ""
        h["served"] = h["served"] or 0
        if (h["endpoint"] or "").strip().rstrip("/") == own:
            continue
        hosted.append(h)
    matching = [h for h in hosted if model and h["model"] and model in h["model"]]
    pool = matching or hosted
    picked, remaining = [], list(pool)
    while remaining and len(picked) < n:
        weights = [max(h["share"], 0) or 1 for h in remaining]
        ch = random.choices(remaining, weights=weights, k=1)[0]
        picked.append(ch)
        remaining.remove(ch)
    return picked


def pool_total_share() -> int:
    """Sum of how much capacity the community currently lends to the pool."""
    conn = db()
    row = conn.execute("SELECT COALESCE(SUM(share), 0) FROM hosters WHERE enabled=1").fetchone()
    conn.close()
    return int(row[0] or 0)


def pool_leaders(limit=3):
    """Top contributors by served requests — anonymous node IDs only."""
    conn = db()
    try:
        rows = conn.execute(
            "SELECT name, served, share FROM hosters WHERE enabled=1 AND served>0"
            " ORDER BY served DESC, name LIMIT ?",
            (max(1, min(limit, 10)),),
        ).fetchall()
        return [{"name": r["name"], "served": r["served"] or 0, "share": r["share"] or 0}
                for r in rows]
    finally:
        conn.close()


def pool_active_nodes(limit=10):
    """Currently-active pool nodes that respond and listen.

    pull=1 rows with a fresh last_seen are listening (outbound heartbeats);
    any enabled row past cooldown with recent last_ok is responding.
    Anonymous node IDs only — safe to display anywhere.
    """
    conn = db()
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff = (now - datetime.timedelta(seconds=PULL_ONLINE_AFTER)).isoformat()
        now_s = now.isoformat()
        rows = conn.execute(
            """SELECT name, pull, share, last_seen, last_ok, served, down_until
               FROM hosters WHERE enabled=1
               AND (last_seen>? OR last_ok>?)
               AND (down_until='' OR down_until IS NULL OR down_until<=?)
               ORDER BY pull DESC, last_seen DESC, last_ok DESC LIMIT ?""",
            (cutoff, cutoff, now_s, max(1, min(limit, 25))),
        ).fetchall()
        return [{"name": r["name"], "pull": bool(r["pull"]), "share": r["share"] or 0,
                 "last_seen": r["last_seen"] or "", "last_ok": r["last_ok"] or "",
                 "served": r["served"] or 0}
                for r in rows]
    finally:
        conn.close()


def pool_add(endpoint, model, share=50, name=None):
    """Add a contributor. Endpoint + model are encrypted at rest; the name is
    a random anonymous node ID unless one is supplied."""
    conn = db()
    name = (name or pool_anon_name()).strip()[:80]
    conn.execute(
        "INSERT INTO hosters (name, endpoint, model, share, enabled, added_by, at, endpoint_hash) VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
        (
            name or pool_anon_name(),
            maybe_encrypt("pool_endpoint", endpoint),
            maybe_encrypt("pool_model", model),
            int(share),
            "bot",
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            _endpoint_hash(endpoint),
        ),
    )
    conn.commit()
    conn.close()
    _pool_snapshot_invalidate()
    return name or pool_anon_name()


def pool_remove(hoster_id):
    conn = db()
    conn.execute("DELETE FROM hosters WHERE id=?", (int(hoster_id),))
    conn.commit()
    conn.close()
    _pool_snapshot_invalidate()


def pool_set(hoster_id, enabled=None, share=None):
    conn = db()
    if enabled is not None:
        conn.execute("UPDATE hosters SET enabled=? WHERE id=?", (1 if enabled else 0, int(hoster_id)))
    if share is not None:
        conn.execute("UPDATE hosters SET share=? WHERE id=?", (max(0, min(100, int(share))), int(hoster_id)))
    conn.commit()
    conn.close()
    _pool_snapshot_invalidate()


def pick_pool_endpoint(model="") -> str:
    """Pick the first healthy pool endpoint (weighted by share). Empty pool → ""."""
    cands = pool_candidates(model, limit=1)
    return (cands[0]["endpoint"] or "").strip() if cands else ""


PULL_JOB_WAIT = 25  # seconds to wait for pull workers before direct fallback
PULL_ONLINE_AFTER = 180  # a pull node seen this recently counts as online


def pull_nodes_online(model="") -> int:
    """Recently-seen pull workers (outbound-only contributors). Model is
    matched at claim time; any online pull node means the queue is live."""
    conn = db()
    try:
        cutoff = (datetime.datetime.now(datetime.timezone.utc)
                  - datetime.timedelta(seconds=PULL_ONLINE_AFTER)).isoformat()
        n = conn.execute(
            "SELECT COUNT(*) FROM hosters WHERE enabled=1 AND pull=1 AND last_seen>?",
            (cutoff,),
        ).fetchone()[0]
        return int(n or 0)
    finally:
        conn.close()


async def ask_pull_pool(prompt: str, model: str, temperature: float, max_tokens: int,
                        top_p: float, repeat_penalty: float, stop: list, system: str) -> str:
    """Enqueue a job for pull workers and wait. Raises ConnectionError on
    timeout so the caller falls back to direct routing. Stale jobs are
    reaped so the table can't grow forever."""
    import json as _json
    conn = db()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO pool_jobs (prompt, system, model, temperature, max_tokens, top_p,
                                  repeat_penalty, stop, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
        (prompt[:2000], (system or "")[:2000], model, float(temperature), int(max_tokens),
         float(top_p), float(repeat_penalty), _json.dumps(stop or []), now),
    )
    job_id = cur.lastrowid
    # Reap jobs older than an hour (done/failed/pending alike).
    try:
        old = (datetime.datetime.now(datetime.timezone.utc)
               - datetime.timedelta(hours=1)).isoformat()
        conn.execute("DELETE FROM pool_jobs WHERE created_at<?", (old,))
    except Exception:
        pass
    conn.commit()
    deadline = time.time() + PULL_JOB_WAIT
    try:
        while time.time() < deadline:
            await asyncio.sleep(1.5)
            # Fresh short connection per poll — never hold one across sleeps
            # (WAL or not, a 25s-held reader stalls writers).
            c2 = db()
            try:
                row = c2.execute("SELECT status, result, error FROM pool_jobs WHERE id=?",
                                 (job_id,)).fetchone()
            finally:
                c2.close()
            if row is None:
                break
            if row["status"] == "done":
                answer = (row["result"] or "").strip()
                if not answer:
                    raise ConnectionError("AI returned an empty reply.")
                return answer
            if row["status"] == "failed":
                raise ConnectionError("AI worker failed — falling back.")
        raise ConnectionError("AI pull workers are busy. Trying direct route.")
    finally:
        try:
            conn.execute("DELETE FROM pool_jobs WHERE id=?", (job_id,))
            conn.commit()
        except Exception:
            pass
        conn.close()


def _pool_ping(endpoint: str) -> bool:
    """Reachability probe for a pool host (its /api/tags)."""
    try:
        req = urllib.request.Request(f"{endpoint}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=8) as resp:
            resp.read()
        return True
    except Exception:
        return False


async def pool_health_loop():
    """Keep the pool's view of who's alive current even between AI calls, so a
    computer that sleeps (laptop lid) or drops its link is parked quickly and
    comes straight back the moment it's reachable again.

    Scale-safe: checks a rotating sample (not the whole pool) each cycle with
    bounded concurrency, so 100k hosts cost the same per cycle as 50."""
    sample = 50
    offset = 0
    while True:
        await asyncio.sleep(POOL_HEALTH_INTERVAL)
        try:
            conn = db()
            rows = conn.execute(
                "SELECT id, endpoint FROM hosters WHERE enabled=1 ORDER BY id LIMIT ? OFFSET ?",
                (sample, offset),
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM hosters WHERE enabled=1").fetchone()[0]
            conn.close()
        except Exception:
            continue
        if not rows:
            offset = 0
            continue
        offset = (offset + sample) % max(total, 1)
        eps = []
        for r in rows:
            ep = maybe_decrypt("pool_endpoint", r["endpoint"] or "").strip()
            if ep:
                eps.append(ep)
        sem = asyncio.Semaphore(10)

        async def _one(ep):
            async with sem:
                return ep, await asyncio.to_thread(_pool_ping, ep)

        for coro in asyncio.as_completed([_one(ep) for ep in eps]):
            try:
                ep, ok = await coro
                await asyncio.to_thread(pool_record, ep, ok)
            except Exception:
                pass


def cmd_tick(name: str):
    """Count one slash-command invocation for public stats (fire-and-forget)."""
    try:
        conn = db()
        conn.execute(
            """INSERT INTO cmdlog (name, calls) VALUES (?, 1)
               ON CONFLICT(name) DO UPDATE SET calls = calls + 1""",
            (str(name),),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def usage_bucket(window: int) -> str:
    """Rolling bucket key: hour, 6-hour block, or calendar day."""
    now = datetime.datetime.now(datetime.timezone.utc)
    if window <= 1:
        return now.strftime("%Y%m%d%H")
    if window <= 6:
        return now.strftime("%Y%m%d") + str(now.hour // 6)
    return now.strftime("%Y%m%d")


def quota_ok(guild_id, quota: int, window: int = 24) -> bool:
    """True if the guild has quota left in the current window (0 = unlimited)."""
    if not quota:
        return True
    bucket = usage_bucket(window)
    conn = db()
    row = conn.execute(
        "SELECT calls FROM usage WHERE guild_id=? AND bucket=?",
        (str(guild_id), bucket),
    ).fetchone()
    calls = row["calls"] if row else 0
    conn.close()
    return calls < quota


def quota_tick(guild_id, window: int = 24):
    bucket = usage_bucket(window)
    conn = db()
    conn.execute(
        """INSERT INTO usage (guild_id, bucket, calls) VALUES (?, ?, 1)
           ON CONFLICT(guild_id, bucket) DO UPDATE SET calls = calls + 1""",
        (str(guild_id), bucket),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Conversation mode — the bot "stays" a while after a reply, then goes quiet
# ---------------------------------------------------------------------------

_conv_until = {}


def conv_mark(guild_id, channel_id, minutes: int):
    """The bot is now "in conversation" in this channel for ``minutes`` more."""
    _conv_until[(guild_id, channel_id)] = time.time() + max(1, minutes) * 60.0


def conv_live(guild_id, channel_id, cfg) -> bool:
    """True if conversation mode is on for this server and the bot is still
    in an active conversation in this channel (next message needs no @)."""
    if not cfg.get("conv"):
        return False
    expiry = _conv_until.get((guild_id, channel_id))
    if expiry is None:
        return False
    if time.time() >= expiry:
        _conv_until.pop((guild_id, channel_id), None)
        return False
    return True


async def _say_goodbye(message):
    """A member told the bot to leave — end the conversation right now."""
    _conv_until.pop((message.guild.id, message.channel.id), None)
    try:
        await message.channel.send(
            "👋 okay, I'm stepping out. Just @ me or use `/ask` whenever you want to talk again."
        )
    except discord.Forbidden:
        pass


# ---------------------------------------------------------------------------
# Per-channel conversation memory (split conversations & history)
# ---------------------------------------------------------------------------

class MemoryBank:
    """Rolling memory per (guild, channel), persisted in the shared SQLite DB.

    Persisting means the web panel can inspect and clear a server's memory,
    and conversations survive bot restarts. Each channel is trimmed at
    insert time so it can never grow unbounded. Every entry records who said
    it (user_id + display name) so the bot never confuses speakers.
    """

    def push(self, guild_id, channel_id, role, text, maxlen, user_id="", name=""):
        conn = db()
        conn.execute(
            "INSERT INTO memory (guild_id, channel_id, role, text, at, user_id, name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(guild_id), str(channel_id), role, (text or "")[:1000],
             datetime.datetime.now().isoformat(), str(user_id or ""), (name or "")[:60]),
        )
        cap = max(8, int(maxlen) * 2 + 4)
        rows = conn.execute(
            "SELECT rowid FROM memory WHERE guild_id=? AND channel_id=? ORDER BY rowid DESC",
            (str(guild_id), str(channel_id)),
        ).fetchall()
        if len(rows) > cap:
            conn.execute(
                "DELETE FROM memory WHERE guild_id=? AND channel_id=? AND rowid <= ?",
                (str(guild_id), str(channel_id), rows[cap - 1]["rowid"]),
            )
        conn.commit()
        conn.close()

    def context(self, guild_id, channel_id, maxlen):
        conn = db()
        rows = conn.execute(
            "SELECT role, user_id, name, text FROM memory "
            "WHERE guild_id=? AND channel_id=? ORDER BY rowid DESC LIMIT ?",
            (str(guild_id), str(channel_id), int(maxlen)),
        ).fetchall()
        conn.close()
        return [
            {"role": r["role"], "user_id": r["user_id"], "name": r["name"], "text": r["text"]}
            for r in reversed(rows)
        ]

    def clear(self, guild_id, channel_id=None):
        conn = db()
        if channel_id is None:
            conn.execute("DELETE FROM memory WHERE guild_id=?", (str(guild_id),))
        else:
            conn.execute(
                "DELETE FROM memory WHERE guild_id=? AND channel_id=?",
                (str(guild_id), str(channel_id)),
            )
        conn.commit()
        conn.close()


memory = MemoryBank()


# ---------------------------------------------------------------------------
# Member profiles — "who is who". We learn a short profile per member from
# what they actually say ("i like fish"), keep who sent what in memory, and
# feed both back into the AI prompt so the bot talks to the right person.
# ---------------------------------------------------------------------------

# Light heuristics to pull "self facts" out of casual chat. Kept deliberately
# narrow so we don't invent things — only "I/me/my" statements become facts.
_PROFILE_RE = [
    re.compile(r"(?:^|\s)i(?:'?m| am) ([a-z][a-z ]{1,40})", re.I),       # "i'm a baker"
    re.compile(r"(?:^|\s)i (?:really )?(?:like|love|enjoy) ([a-z][a-z ]{1,40})", re.I),
    re.compile(r"(?:^|\s)i (?:hate|dislike) ([a-z][a-z ]{1,40})", re.I),
    re.compile(r"(?:^|\s)my favourite? (?:is|are) ([a-z][a-z ]{1,40})", re.I),
    re.compile(r"(?:^|\s)i (?:play|watch|read|program(?: in| with)?) ([a-z][a-z ]{1,40})", re.I),
    re.compile(r"(?:^|\s)i (?:work|work as|work with) ([a-z][a-z ]{1,40})", re.I),
    re.compile(r"(?:^|\s)i (?:use|listen to|collect) ([a-z][a-z ]{1,40})", re.I),
]

_PROFILE_STOP = re.compile(r"\b(?:u|ur|you|your|my|and|to|the|for|with|that|this|what|how|why|when|where|who|do|you|me|it|is|are|was|be|so|just|like)\b", re.I)


def extract_facts(text: str) -> list:
    """Pull a few short "facts" a member stated about themselves. Returns
    lowercased fragments like ['a baker', 'fish', 'minecraft'] — capped and
    cleaned so the profile stays tiny and honest."""
    out = []
    t = (text or "")[:400]
    for rx in _PROFILE_RE:
        for m in rx.finditer(t):
            frag = m.group(1).strip().strip(".,!?")
            if not frag or len(frag) > 40:
                continue
            if _PROFILE_STOP.fullmatch(frag):
                continue
            if frag not in out:
                out.append(frag)
        if len(out) >= 4:
            break
    return out


def learn_profile_message(message):
    """Record the author's self-facts + keep the speaker attribution."""
    if message.author.bot or message.guild is None:
        return
    facts = extract_facts(message.content)
    if not facts:
        return
    conn = db()
    row = conn.execute(
        "SELECT facts FROM profiles WHERE guild_id=? AND user_id=?",
        (str(message.guild.id), str(message.author.id)),
    ).fetchone()
    known = (row["facts"].split("\n") if row else [])
    for f in facts:
        if f not in known:
            known.append(f)
    known = known[-8:]
    conn.execute(
        """INSERT INTO profiles (guild_id, user_id, name, facts, at) VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(guild_id, user_id)
           DO UPDATE SET name=excluded.name, facts=excluded.facts, at=excluded.at""",
        (str(message.guild.id), str(message.author.id),
         message.author.display_name[:60], "\n".join(known),
         datetime.datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def _facts_text(parts) -> str:
    """Turn raw fact fragments into a readable list, preserving the verb."""
    nice = []
    for p in parts:
        low = p.lower().strip()
        if low.startswith(("hate ", "hates ", "dislike ", "don't like ", "dont like ")):
            # keep the negative — never turn "hate X" into "likes X"
            frag = re.sub(r"^(hates?|dislike|don't like|dont like)\s+", "", p, flags=re.I).strip()
            nice.append(f"dislikes {frag}" if frag else f"dislikes {p}")
        elif low.startswith(("a ", "an ", "the ")):
            nice.append(f"is {p}")
        elif re.match(r"^(play|watch|read|love|like|enjoy)\b", low, re.I):
            verb = low.split()[0]
            mapping = {"play": "plays", "watch": "watches", "read": "reads",
                       "love": "loves", "like": "likes", "enjoy": "enjoys"}
            frag = re.sub(r"^(play|watch|read|love|like|enjoy)\s+", "", p, flags=re.I).strip()
            nice.append(f"{mapping.get(verb, 'likes')} {frag}" if frag else p)
        elif re.match(r"^work as\b", low, re.I):
            frag = re.sub(r"^work as\s+", "", p, flags=re.I).strip()
            nice.append(f"is {frag}")
        else:
            nice.append(f"likes {p}")
    return ", ".join(nice)


def profile_facts(guild_id, user_id) -> str:
    """This member's learned profile as a sentence, or ''."""
    conn = db()
    row = conn.execute(
        "SELECT facts FROM profiles WHERE guild_id=? AND user_id=?",
        (str(guild_id), str(user_id)),
    ).fetchone()
    conn.close()
    if not row or not row["facts"]:
        return ""
    return _facts_text(row["facts"].split("\n"))


def profile_lines(guild_id, user_ids) -> list:
    """["name — likes fish", ...] for the given members (deduped, capped)."""
    ids = []

    def _seen(u):
        s = str(u)
        if s and s not in ids:
            ids.append(s)

    for u in user_ids:
        _seen(u)
    if not ids:
        return []
    q = ",".join("?" * len(ids))
    conn = db()
    rows = conn.execute(
        f"SELECT name, facts FROM profiles WHERE guild_id=? AND user_id IN ({q})",
        [str(guild_id), *ids],
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        if not r["facts"]:
            continue
        facts = _facts_text(r["facts"].split("\n"))
        out.append(f"{r['name']} — {facts}")
    return out[:8]


def build_prompt(persona, context, question, instructions="", member_profiles=None, asker_name="member"):
    """Build the LLM prompts, split for Ollama's system/prompt roles.

    Returns (system, user_prompt): the persona, date, reply rules, member
    notes and server rules go in ``system`` (hierarchy the model respects);
    only the conversation history + current question go in the user prompt.
    The old single-blob layout let small models echo instructions back, so
    history lines are also capped short here.
    """
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%B %d, %Y")
    sys_lines = [
        persona.strip(),
        "",
        f"Today is {today} (UTC). Use it for anything time-related; never guess dates.",
        "Reply in 1-2 Discord sentences, <=45 words. Answer directly. "
        "If unsure, say so in 5 words. Never end with a question.",
    ]
    if member_profiles:
        sys_lines += ["", "MEMBER NOTES (background only — use silently, never recite unless asked):"]
        for line in member_profiles[:4]:
            sys_lines.append(f"- {line}")
    if instructions:
        sys_lines += ["", "SERVER RULES (highest priority):", "<<<", instructions[:600].strip(), ">>>"]
    hist = ["History (oldest first, 'bot:' is you):"]
    for m in context[-8:]:
        if m["role"] == "bot":
            who = "bot"
        else:
            try:
                uid = int(str(m.get("user_id") or "0"))
            except (ValueError, TypeError):
                uid = 0
            who = f"{m.get('name') or 'member'}[{uid % 10000:04d}]"
        hist.append(f"{who}: {(m.get('text') or '')[:160]}")
    hist += ["", f"{asker_name}: {(question or '')[:300]}", "bot:"]
    return "\n".join(sys_lines), "\n".join(hist)


# ---------------------------------------------------------------------------
# Fair AI queue — one model call at a time, spread fairly across servers
# ---------------------------------------------------------------------------

class BusyError(Exception):
    pass


class FairAIQueue:
    """Per-guild workers with global fairness: each server gets its own
    worker (one slow guild can't stall everyone), while a global semaphore
    caps total concurrent model calls so a weak box isn't melted. Queues
    still cap per-guild backlog with a busy reply.
    """

    def __init__(self, max_waiting=2, max_concurrent=2,
                 busy_reply="Quaestio's head is busy — one chat at a time. Ask again in a moment."):
        self._queues = {}
        self._workers = {}
        self._slots = asyncio.Semaphore(max_concurrent) if hasattr(asyncio, "Semaphore") else None
        self._max_waiting = max_waiting
        self.busy_reply = busy_reply

    def submit(self, guild_id, factory):
        fut = asyncio.get_running_loop().create_future()
        gid = str(guild_id)
        q = self._queues.setdefault(gid, [])
        if len(q) >= self._max_waiting:
            fut.set_exception(BusyError(self.busy_reply))
            return fut
        q.append((fut, factory))
        w = self._workers.get(gid)
        if w is None or w.done():
            self._workers[gid] = asyncio.create_task(self._run_guild(gid))
        return fut

    def drop(self, guild_id):
        """Fail waiting (not running) jobs so a persona change applies instantly."""
        q = self._queues.get(str(guild_id))
        if not q:
            return 0
        n = 0
        for fut, _factory in q:
            if not fut.done():
                fut.set_exception(BusyError("Settings updated — ask again with the new style! 🎨"))
                n += 1
        q.clear()
        return n

    async def _one(self, fut, factory):
        # Worker budget slightly UNDER the waiter budget so the factory is
        # cancelled first — no orphan run hogging a slot.
        try:
            if self._slots is not None:
                async with self._slots:
                    result = await asyncio.wait_for(factory(), timeout=OLLAMA_TIMEOUT + 25)
            else:
                result = await asyncio.wait_for(factory(), timeout=OLLAMA_TIMEOUT + 25)
            if not fut.done():
                fut.set_result(result)
        except asyncio.TimeoutError:
            if not fut.done():
                fut.set_exception(ConnectionError("AI took too long."))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not fut.done():
                fut.set_exception(exc)

    async def _run_guild(self, guild_id):
        try:
            while True:
                q = self._queues.get(guild_id)
                if not q:
                    self._queues.pop(guild_id, None)
                    return
                fut, factory = q.pop(0)
                if fut.cancelled():
                    continue
                await self._one(fut, factory)
        finally:
            self._workers.pop(guild_id, None)


ai_queue = FairAIQueue()


# ---------------------------------------------------------------------------
# Human-like streaming (typing indicator + natural pauses)
# ---------------------------------------------------------------------------

async def human_type(channel, text, mention=""):
    """Stream a reply into the channel like a person typing.

    Shows the typing indicator, reveals the message in chunks via edit +
    small random pauses, so a fast local model still feels natural. An
    optional ``mention`` (e.g. a user ping) is glued to the first chunk.
    """
    text = (text or "").strip()
    # Models sometimes echo the "bot:" stop token — strip it, not the reply.
    text = re.sub(r"^(bot\s*:\s*)", "", text, flags=re.I).strip()
    # Truncate to the last full sentence under 400 chars so a num_predict
    # cutoff never posts half-words.
    if len(text) > 400:
        cut = text[:400]
        last_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        if last_end > 120:
            text = cut[:last_end + 1]
        else:
            text = cut.rsplit(" ", 1)[0] + "…"
    chunks = []
    current = ""
    for token in text.split():
        if len(current) + len(token) + 1 > 350:
            chunks.append(current + " " if not current.endswith("\n") else current)
            current = token
        else:
            current = (current + " " + token).lstrip()
    if current:
        chunks.append(current)
    if not chunks:
        chunks = [text]
    if mention and chunks:
        chunks[0] = f"{mention} {chunks[0]}"

    try:
        async with channel.typing():
            await asyncio.sleep(0.35 + random.random() * 0.4)
            msg = await channel.send(chunks[0])
            for chunk in chunks[1:]:
                await asyncio.sleep(0.45 + random.random() * 0.5)
                new_text = (msg.content + " " + chunk).lstrip()
                if len(new_text) > 1990:
                    break
                try:
                    await msg.edit(content=new_text)
                except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                    break
        return msg
    except (discord.Forbidden, discord.NotFound, discord.HTTPException):
        return None


_GOODBYE = (    "go away", "goodbye", "bye bye", " bye", "shoo", "get lost",
    "leave me alone", "stop talking", "stop replying", "done talking",
    "that's all", "thats all", "never mind", "nevermind", "go to sleep",
)


def is_goodbye(text: str) -> bool:
    low = re.sub(r"<@!?[0-9]+>", "", text).lower().strip()
    return any(phrase in low or low == phrase.strip() for phrase in _GOODBYE)


_ai_error_at = {}


async def _ai_error_notice(channel, guild_id, text: str, cooldown: int = 120):
    """Throttled user-visible AI failure notice (avoids spam on mention storms)."""
    key = (str(guild_id), str(getattr(channel, "id", "0")), text[:40])
    now = time.time()
    if now - _ai_error_at.get(key, 0) < cooldown:
        return
    _ai_error_at[key] = now
    try:
        await channel.send(text)
    except (discord.Forbidden, discord.HTTPException):
        pass


# Flood guard (abuse protection without quotas): per-user and per-guild
# sliding windows. Quotas default to unlimited — the pool absorbs load by
# queueing (slow replies under pressure, never a crash) — and this stops one
# person or server from hogging the single AI worker. In-memory: a restart
# resets windows, which is fine for abuse control.
FLOOD_USER_PER_MIN = 5
FLOOD_GUILD_PER_MIN = 20
_flood_hits = {}


def flood_ok(guild_id, user_id, mult=1) -> bool:
    """True if this user/guild may queue another AI call right now.
    Contributors pass mult>1 (see contributor_mult) instead of any quota."""
    mult = max(1, int(mult or 1))
    now = time.time()
    for key, limit in ((f"u:{guild_id}:{user_id}", FLOOD_USER_PER_MIN * mult),
                       (f"g:{guild_id}", FLOOD_GUILD_PER_MIN * mult)):
        hits = [t for t in _flood_hits.get(key, []) if now - t < 60]
        if len(hits) >= limit:
            _flood_hits[key] = hits
            return False
        hits.append(now)
        _flood_hits[key] = hits
    if len(_flood_hits) > 5000:
        _flood_hits.clear()
    return True


def _log_ai_failure(kind: str, guild_id, model: str, exc: Exception):
    """One journal line per AI failure so the monitor can alert on clusters.
    User-visible notices stay throttled; logs stay complete."""
    try:
        print(f"quaestio: AI failure kind={kind} guild={guild_id} "
              f"model={model} err={str(exc)[:120]}", flush=True)
    except Exception:
        pass


async def _wait_ai_answer(fut, budget: float, on_slow=None):
    """Wait for a queued AI answer with a progress nudge: after 45s of
    silence call on_slow() once (e.g. 'still working…') instead of leaving
    only the typing indicator, then wait out the remaining budget."""
    first = min(45.0, budget)
    try:
        return await asyncio.wait_for(asyncio.shield(fut), timeout=first)
    except asyncio.TimeoutError:
        pass
    if on_slow is not None:
        try:
            await on_slow()
        except (discord.Forbidden, discord.HTTPException):
            pass
    rest = max(1.0, budget - first)
    try:
        return await asyncio.wait_for(asyncio.shield(fut), timeout=rest)
    except asyncio.TimeoutError:
        pass
    raise asyncio.TimeoutError()


def _prune_notice_caches():
    """Bound in-memory notice dicts (called per AI reply; cheap)."""
    now = time.time()
    try:
        for key in [k for k, v in _ai_error_at.items() if now - v > 3600]:
            _ai_error_at.pop(key, None)
        for key in [k for k, v in _quota_notice_at.items() if now - v > 3600]:
            _quota_notice_at.pop(key, None)
    except Exception:
        pass


async def ai_reply(message: discord.Message, *, ping: bool = True):
    """Passive AI chat: reply to an @-mention, or (ping=False) to a plain
    follow-up while conversation mode is active.

    Honors the same per-server rules as /ask: enabled, channel allowlist,
    quota, memory, instructions. Does not ping the author for follow-ups so
    a back-and-forth stays conversational.
    """
    guild_id = message.guild.id
    cfg = guild_ai_config(guild_id)
    if not cfg["enabled"] or not cfg["ai_mention"]:
        return False
    if not channel_allowed(guild_id, message.channel.id, cfg):
        return False
    if not quota_ok(guild_id, cfg["quota"], cfg["window"]):
        await _quota_notice(message)
        return False
    if not flood_ok(guild_id, message.author.id, mult=contributor_mult(guild_id)):
        await _ai_error_notice(message.channel, guild_id,
                               "⏳ Slow down — too many AI requests at once. Try again in a minute.")
        return False

    me_id = message.guild.me.id if message.guild.me else None
    raw = message.content[:400]
    if me_id:
        raw = raw.replace(f"<@{me_id}>", "").replace(f"<@!{me_id}>", "")
    question = raw.strip() or "…"
    asker = message.author.display_name or "member"
    _prune_notice_caches()
    context = memory.context(guild_id, message.channel.id, cfg["memory"])
    profiles = profile_lines(guild_id, [m.get("user_id") for m in context] + [message.author.id])
    persona_system, full_prompt = build_prompt(cfg["persona"], context, question, cfg["instructions"], profiles, asker_name=asker)

    async def factory():
        # Re-read config at execution time so persona changes apply instantly,
        # even to jobs that were already queued.
        cfg2 = guild_ai_config(guild_id)
        persona_system2, full_prompt2 = build_prompt(
            cfg2["persona"], context, question, cfg2["instructions"], profiles, asker_name=asker)
        return await ask_ollama_any(cfg2, full_prompt2, asker=asker, system=persona_system2)

    fut = ai_queue.submit(guild_id, factory)
    budget = OLLAMA_TIMEOUT + 30

    async def _slow():
        try:
            await message.channel.send("⏳ Still working on it — your reply is queued! 💭")
        except (discord.Forbidden, discord.HTTPException):
            pass

    async with message.channel.typing():
        try:
            answer = await _wait_ai_answer(fut, budget, on_slow=_slow)
            await asyncio.sleep(0)
        except BusyError:
            return False
        except asyncio.TimeoutError:
            _log_ai_failure("timeout", guild_id, cfg["model"], TimeoutError("waiter budget spent"))
            await _ai_error_notice(message.channel, guild_id, "⏳ The AI took too long — try again in a moment.")
            return False
        except ConnectionError as exc:
            _log_ai_failure("connection", guild_id, cfg["model"], exc)
            await _ai_error_notice(
                message.channel, guild_id,
                NO_COMPUTE_NOTICE if _is_no_compute(exc) else f"⚠️ {exc}")
            return False
        except asyncio.CancelledError:
            raise

    quota_tick(guild_id, cfg["window"])
    memory.push(guild_id, message.channel.id, "user", question, cfg["memory"],
                user_id=message.author.id, name=message.author.display_name)
    memory.push(guild_id, message.channel.id, "bot", answer[:400], cfg["memory"],
                user_id=message.guild.me.id, name=message.guild.me.display_name)
    # No pings on passive replies — the message itself is the signal.
    await human_type(message.channel, answer, mention="")
    return True


# ---------------------------------------------------------------------------
# DMs — the bot chats with anyone who messages it directly
# ---------------------------------------------------------------------------

_quota_notice_at = {}


async def _quota_notice(message):
    """Tell a channel once per 30 min that the AI quota blocked a reply."""
    key = (message.guild.id, message.channel.id)
    now = time.time()
    if now - _quota_notice_at.get(key, 0) < 1800:
        return
    _quota_notice_at[key] = now
    try:
        await message.channel.send(
            "⏳ This server hit its AI quota for this hour, so I went quiet. "
            "It resets on the hour — the number is in the web panel under AI → Quota."
        )
    except discord.Forbidden:
        pass


def host_cfg():
    """The host operator's own AI box (used for DMs and as managed-mode source)."""
    h = lambda k, d: get_cfg("host", k, d)
    _pers = h("ai_personality", "none") or "none"
    _char = h("ai_character", "") or ""
    return {
        "endpoint": h("ai_endpoint", OLLAMA_BASE_URL),
        "model": h("ai_model", OLLAMA_MODEL),
        "memory": max(1, _safe_int(h("ai_memory", MEMORY_DEFAULT), MEMORY_DEFAULT)),
        "instructions": h("ai_instructions", ""),
        "persona": persona_from(
            _pers, _char,
            guild_presets("host", "personality"), guild_presets("host", "character"),
        ),
        "persona_params": persona_params_for(_pers, _char),
        "quota": max(0, _safe_int(h("ai_quota", "0"), 0)),
        "temperature": _safe_float(h("ai_temperature", "0.6"), 0.6),
        "max_tokens": _safe_int(h("ai_max_tokens", "150"), 150),
        "window": max(1, _safe_int(h("ai_window", "6"), 6)),
        "enabled": flag_on("host", "ai_enabled", "1"),
        "dm_enabled": flag_on("host", "ai_dm", "1"),
    }


async def dm_chat(channel, question, cfg, mention="", user_id="", name=""):
    """Run one AI turn in a DM (or a /ask inside a DM)."""
    if not cfg["enabled"]:
        await channel.send("AI chat is disabled right now.")
        return
    if not cfg["dm_enabled"]:
        await channel.send("DM chat is switched off — try mentioning the bot in a server instead.")
        return
    if not quota_ok("dm", cfg["quota"], cfg["window"]):
        await channel.send(
            "⏳ This hour's AI quota is used up — it resets on the hour. "
            "Keep chatting after that, or raise the limit in the web panel."
        )
        return
    if not flood_ok("dm", user_id or channel.id):
        await channel.send("⏳ Slow down — too many AI requests at once. Try again in a minute.")
        return

    context = memory.context("dm", channel.id, cfg["memory"])
    who = [m.get("user_id") for m in context]
    asker = name or "member"
    persona_system, full_prompt = build_prompt(cfg["persona"], context, question, cfg["instructions"], profile_lines("dm", who), asker_name=asker)

    async def factory():
        return await ask_ollama_any(cfg, full_prompt, asker=asker, system=persona_system)

    fut = ai_queue.submit("dm", factory)
    try:
        # Typing the whole wait, like a normal chatbot — no stray messages.
        async with channel.typing():
            answer = await _wait_ai_answer(fut, OLLAMA_TIMEOUT + 30)
        await asyncio.sleep(0)
    except BusyError as exc:
        await channel.send(str(exc))
        return
    except asyncio.TimeoutError:
        _log_ai_failure("timeout", "dm", cfg["model"], TimeoutError("waiter budget spent"))
        await channel.send("The AI took too long. Try again in a moment.")
        return
    except ConnectionError as exc:
        _log_ai_failure("connection", "dm", cfg["model"], exc)
        await channel.send(NO_COMPUTE_NOTICE if _is_no_compute(exc) else f"⚠️ {exc}")
        return
    except asyncio.CancelledError:
        raise

    quota_tick("dm", cfg["window"])
    memory.push("dm", channel.id, "user", question, cfg["memory"], user_id=user_id, name=name)
    memory.push("dm", channel.id, "bot", answer[:400], cfg["memory"])
    await human_type(channel, answer, mention=mention)


async def dm_reply(message: discord.Message):
    """Mention-style chat in DMs: any message to the bot gets an answer."""
    question = message.content[:400].strip() or "…"
    await dm_chat(message.channel, question, host_cfg(), mention=message.author.mention,
                  user_id=message.author.id, name=message.author.display_name)


# ---------------------------------------------------------------------------
# Leveling helpers
# ---------------------------------------------------------------------------

def xp_for_level(level: int) -> int:
    return 5 * (level ** 2) + 50 * level + 100


def add_xp(guild_id, user_id):
    conn = db()
    conn.execute(
        """INSERT INTO xp (guild_id, user_id, messages) VALUES (?, ?, 1)
           ON CONFLICT(guild_id, user_id)
           DO UPDATE SET messages = messages + 1""",
        (str(guild_id), str(user_id)),
    )
    row = conn.execute(
        "SELECT messages FROM xp WHERE guild_id=? AND user_id=?",
        (str(guild_id), str(user_id)),
    ).fetchone()
    conn.commit()
    conn.close()
    return row["messages"]


def level_for_messages(messages: int) -> int:
    level = 0
    while messages >= xp_for_level(level + 1) and level < 100:
        level += 1
    return level


# Anti-spam guards for XP: a message only counts towards level/rank if it has
# between xp_min_words and xp_max_words words, and no more often than once per
# xp_cooldown seconds per member. This stops people farming ranks with a pasted
# wiki article, or by sending every word (or letter) as its own message.
_xp_last = {}  # (guild_id, user_id) -> monotonic time of last counted message


def xp_word_count(content: str) -> int:
    return len([w for w in str(content or "").split() if any(ch.isalnum() for ch in w)])


def xp_allowed(guild_id, user_id, content) -> bool:
    if not flag_on(guild_id, "xp_spam", "1"):
        return True
    min_w = _safe_int(get_cfg(guild_id, "xp_min_words"), 3)
    max_w = _safe_int(get_cfg(guild_id, "xp_max_words"), 100)
    cooldown = _safe_float(get_cfg(guild_id, "xp_cooldown"), 5.0)
    if xp_word_count(content) not in range(min_w, max_w + 1):
        return False
    now = time.monotonic()
    last = _xp_last.get((str(guild_id), str(user_id)), 0.0)
    if now - last < cooldown:
        return False
    _xp_last[(str(guild_id), str(user_id))] = now
    return True


def is_admin(user: discord.Member) -> bool:
    return user.guild_permissions.administrator


# ---------------------------------------------------------------------------
# Event hooks
# ---------------------------------------------------------------------------

@bot.event
async def on_ready():
    db_init()
    command_count = len(bot.tree.get_commands()) + sum(
        len(g.commands) for g in bot.tree.get_commands() if isinstance(g, app_commands.Group)
    )
    activity = discord.Activity(
        type=discord.ActivityType.watching,
        name="/ask",
        details=f"AI · {OLLAMA_MODEL}",
        state=f"{command_count} commands · free",
        assets={"large_image": RPC_LARGE_IMAGE},
    )
    if RPC_SMALL_IMAGE:
        activity.assets["small_image"] = RPC_SMALL_IMAGE
    await bot.change_presence(activity=activity)
    print(f"quaestio: logged in as {bot.user} · {len(bot.guilds)} servers")
    try:
        synced = await bot.tree.sync()
        print(f"quaestio: {len(synced)} slash commands synced")
    except Exception as exc:
        print(f"quaestio: sync failed: {exc}")
    if not getattr(bot, "_bday_task", None) or bot._bday_task.done():
        bot._bday_task = bot.loop.create_task(birthday_loop())
    if not getattr(bot, "_pool_health_task", None) or bot._pool_health_task.done():
        bot._pool_health_task = bot.loop.create_task(pool_health_loop())
    if not getattr(bot, "_reminder_task", None) or bot._reminder_task.done():
        bot._reminder_task = bot.loop.create_task(reminder_loop())

    # Localhost settings page (opt-in via CLI: quaestio localweb)
    if _local_web() and not getattr(bot, "_local_web_started", False):
        bot._local_web_started = True
        import threading
        threading.Thread(target=_serve_local_web, daemon=True).start()
        print(f"quaestio: local settings page on port {os.environ.get('LOCAL_WEB_PORT', '8123')}")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    learn_profile_message(message)
    if message.content.startswith(PREFIX):
        await bot.process_commands(message)
        return
    if message.guild is None:
        await dm_reply(message)
        return
    # Passive AI: only a direct @Quaestio wakes it. With conversation mode on, a
    # plain message also replies while the bot is "still here" (conv_live) — no
    # @ needed — but messages aimed at *other people* (a random @mention) never
    # wake it, and neither does /ask (see the ask command). Saying "go away"
    # ends the conversation immediately.
    _ai_cfg = guild_ai_config(message.guild.id)
    mentions_me = message.guild.me in message.mentions
    aimed_at_others = any(m != message.guild.me for m in message.mentions)
    if conv_live(message.guild.id, message.channel.id, _ai_cfg) and is_goodbye(message.content):
        await _say_goodbye(message)
    elif mentions_me:
        if await ai_reply(message) and _ai_cfg.get("conv"):
            conv_mark(message.guild.id, message.channel.id, _ai_cfg["conv_minutes"])
    elif conv_live(message.guild.id, message.channel.id, _ai_cfg) and not aimed_at_others:
        if await ai_reply(message, ping=False):
            conv_mark(message.guild.id, message.channel.id, _ai_cfg["conv_minutes"])

    if not flag_on(message.guild.id, "xp_enabled", "1"):
        return
    if not xp_allowed(message.guild.id, message.author.id, message.content):
        return
    messages = add_xp(message.guild.id, message.author.id)
    new_level = level_for_messages(messages)
    old_level = level_for_messages(messages - 1)
    if new_level > old_level and new_level > 1:
        if flag_on(message.guild.id, "level_announce", "1"):
            nxt = xp_for_level(new_level + 1)
            pct = min(1.0, messages / nxt) if nxt else 1.0
            filled = round(pct * 10)
            bar = "▰" * filled + "▱" * (10 - filled)
            try:
                await message.channel.send(
                    f"🎉 {message.author.mention} reached **level {new_level}**!\n"
                    f"{bar} {messages}/{nxt} XP"
                )
            except (discord.Forbidden, discord.HTTPException):
                pass
        role_id = get_cfg(message.guild.id, "levelrole")
        if role_id and str(role_id).strip().isdigit():
            role = message.guild.get_role(int(role_id))
            if role:
                try:
                    await message.author.add_roles(role)
                except (discord.Forbidden, discord.HTTPException):
                    pass


@bot.event
async def on_member_join(member: discord.Member):
    if member.bot or not flag_on(member.guild.id, "welcome_enabled", "1"):
        return
    role_id = get_cfg(member.guild.id, "welcome_role")
    if role_id and str(role_id).strip().isdigit():
        role = member.guild.get_role(int(role_id))
        if role:
            try:
                await member.add_roles(role)
            except (discord.Forbidden, discord.HTTPException):
                pass
    channel_id = get_cfg(member.guild.id, "welcome_channel")
    if not channel_id or not str(channel_id).strip().isdigit():
        return
    channel = member.guild.get_channel(int(channel_id))
    if not channel:
        return
    text = get_cfg(
        member.guild.id, "welcome_message",
        f"Welcome to {member.guild.name}, {member.mention}! 👋",
    )
    # Banner: {banner} URL, {icon} = server icon, {avatar} = member avatar,
    # plus {member}, {server}, {count}.
    banner = (get_cfg(member.guild.id, "welcome_banner", "") or "").strip()
    icon = member.guild.icon.url if member.guild.icon else ""
    avatar = member.display_avatar.url if hasattr(member, "display_avatar") else ""
    text = text.replace("{member}", member.mention).replace("{server}", member.guild.name)
    try:
        text = text.replace("{count}", str(member.guild.member_count or "?"))
    except Exception:
        pass
    text = text.replace("{icon}", icon).replace("{avatar}", avatar)
    embed = None
    files = []
    if banner.lower().startswith(("http://", "https://")) and len(banner) < 500:
        embed = discord.Embed(description=text[:4000], color=0xA78BFA)
        embed.set_image(url=banner)
        if avatar:
            embed.set_thumbnail(url=avatar)
        text = ""
    try:
        if embed is not None:
            await channel.send(embed=embed)
        elif text:
            await channel.send(text)
    except (discord.Forbidden, discord.HTTPException):
        pass
    # Level-up celebration hook data lives on the member row already; the
    # announce below in on_message covers rank upgrades with bars.
    return


@bot.event
async def on_member_remove(member: discord.Member):
    if member.bot or not flag_on(member.guild.id, "welcome_enabled", "1"):
        return
    channel_id = get_cfg(member.guild.id, "welcome_channel")
    if not channel_id or not str(channel_id).strip().isdigit():
        return
    channel = member.guild.get_channel(int(channel_id))
    if not channel:
        return
    text = (get_cfg(member.guild.id, "goodbye_message", "") or "").strip()
    if not text:
        return
    text = text.replace("{member}", member.display_name).replace("{server}", member.guild.name)
    try:
        await channel.send(text[:2000])
    except (discord.Forbidden, discord.HTTPException):
        pass


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: Exception):
    """Never leave an interaction hanging — friendly fallback for anything
    that slips through an individual command's own error handling."""
    error = getattr(error, "original", error)
    if isinstance(error, discord.Forbidden):
        msg = "```🔒 I don't have permission to do that here.```"
    elif isinstance(error, (discord.app_commands.CommandOnCooldown, discord.app_commands.errors.CommandOnCooldown)):
        msg = "```⏳ That command is cooling down — try again shortly.```"
    else:
        msg = "```⚠️ Something went wrong. Try again in a moment.```"
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message(msg, ephemeral=True)
        else:
            await interaction.followup.send(msg, ephemeral=True)
    except discord.HTTPException:
        pass


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

@bot.tree.command(name="ping", description="Check the bot's latency.")
async def ping(interaction: discord.Interaction):
    cmd_tick("ping")
    await interaction.response.send_message(
        f"🏓 Pong! `{round(bot.latency * 1000)}ms`", ephemeral=True
    )


@bot.tree.command(name="uptime", description="How long has the bot been running?")
async def uptime(interaction: discord.Interaction):
    cmd_tick("uptime")
    secs = int(time.time() - START_TIME)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    await interaction.response.send_message(
        f"⏱️ Up for {days}d {hours}h {mins}m {secs}s", ephemeral=True
    )


@bot.tree.command(name="about", description="What is Quaestio?")
async def about(interaction: discord.Interaction):
    cmd_tick("about")
    embed = discord.Embed(
        title="Quaestio",
        description=(
            "Your server's own AI companion. Free, self-hosted, private.\n"
            "AI chat · XP levels · moderation · tags · welcomes.\n"
            "More: https://quaestio.online"
        ),
        color=0xA78BFA,
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="invite", description="Invite Quaestio to another server.")
async def invite(interaction: discord.Interaction):
    cmd_tick("invite")
    await interaction.response.defer(thinking=False, ephemeral=True)
    try:
        app = await bot.application_info()
    except discord.HTTPException:
        await interaction.followup.send("Couldn't build the invite link right now.", ephemeral=True)
        return
    await interaction.followup.send(
        f"📨 Invite me here: https://discord.com/oauth2/authorize?client_id={app.id}&permissions=1101994781766&scope=bot",
        ephemeral=True,
    )


# ---------------------------------------------------------------------------
# Utility (help, user/server info, avatars)
# ---------------------------------------------------------------------------

def _command_groups() -> list:
    """Organize every registered slash command into help categories."""
    return [
        ("🤖 AI", [
            "/ask <prompt> — chat with Quaestio's local AI",
            "@Quaestio <text> — same, by mentioning the bot",
            "/summarize [limit] — summarize the last N messages here",
            "/ai status — see (admins can tweak) AI settings",
        ]),
        ("🎮 Games", [
            "/8ball <question> — shake the magic 8-ball",
            "/dice [XdY] — roll dice (default 1d6)",
            "/coin — flip a coin",
            "/rps <choice> — rock, paper, scissors",
            "/slot — spin the slots",
            "/trivia — a question; first right /answer wins",
            "/tictactoe <@friend> — start a round",
            "/move <1-9> — take your square",
        ]),
        ("🏆 Levels", [
            "/rank [member] — XP and level",
            "/leaderboard [top] — top chatters by XP",
            "/profile [member] — facts I've learned about that member",
        ]),
        ("🔍 Utility", [
            "/ping — bot latency",
            "/uptime — how long the bot has been online",
            "/userinfo [member] — profile, roles, join dates",
            "/serverinfo — about this server",
            "/avatar [member] — a member's profile picture",
            "/about — what Quaestio is",
            "/invite — invite Quaestio elsewhere",
            "/panel — open this server's web settings",
        ]),
        ("📊 Engagement", [
            "/poll <question> [option1, option2, …] — run a live vote",
            "/remind <what> <minutes> — get pinged later",
            "/birthday set <month> <day> — save a birthday",
            "/birthday list — all saved birthdays",
        ]),
        ("📌 Tags", [
            "/tag <name> — show a saved tag",
            "/tags — list every tag in this server",
            "/tagcreate <name> <content> — save one (admin)",
            "/tagdelete <name> — remove one (admin)",
        ]),
        ("🛡️ Moderation (admin)", [
            "/warn <member> [reason] · /warns <member> · /delwarns <member>",
            "/kick <member> [reason] · /ban <member> [reason] · /unban <name>",
            "/purge [count] — bulk-delete messages",
            "/mute <member> [minutes] · /unmute <member>",
        ]),
    ]


@bot.tree.command(name="help", description="Learn what Quaestio can do.")
async def help_cmd(interaction: discord.Interaction):
    cmd_tick("help")
    embed = discord.Embed(
        title="🛠️ Quaestio commands",
        description=(
            f"Hello **{interaction.user.display_name}**! Here's everything I can do "
            f"on this server.\nAI chat runs on **{OLLAMA_MODEL}** — free, self-hosted "
            "and private."
        ),
        color=0xA78BFA,
    )
    for title, lines in _command_groups():
        embed.add_field(name=title, value="\n".join(f"• {l}" for l in lines), inline=False)
    embed.set_footer(text="Tip: type / and I'll suggest commands as you go.")
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="userinfo", description="Look up a member's profile.")
@app_commands.describe(member="Which member? Defaults to you.")
async def userinfo(interaction: discord.Interaction, member: discord.Member = None):
    cmd_tick("userinfo")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/userinfo` inside a server.", ephemeral=True)
        return
    member = member or interaction.user
    roles = [r.mention for r in reversed(member.roles) if r != member.guild.default_role]
    embed = discord.Embed(
        title=member.display_name,
        description=member.mention,
        color=member.color if member.color != discord.Color.default() else 0xA78BFA,
    )
    embed.set_thumbnail(url=member.display_avatar.url)
    embed.add_field(name="🤖 Bot?", value="Yes" if member.bot else "No", inline=True)
    embed.add_field(
        name="🗓️ Joined",
        value=discord.utils.format_dt(member.joined_at, "R") if member.joined_at else "?",
        inline=True,
    )
    embed.add_field(name="📅 Created", value=discord.utils.format_dt(member.created_at, "R"), inline=True)
    embed.add_field(
        name=f"🎭 Roles ({len(roles)})",
        value=", ".join(roles[:10]) if roles else "None",
        inline=False,
    )
    embed.set_footer(text=f"ID: {member.id}")
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="serverinfo", description="See stats about this server.")
async def serverinfo(interaction: discord.Interaction):
    cmd_tick("serverinfo")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/serverinfo` inside a server.", ephemeral=True)
        return
    guild = interaction.guild
    embed = discord.Embed(title=guild.name, color=0xA78BFA)
    if guild.icon:
        embed.set_thumbnail(url=guild.icon.url)
    embed.add_field(name="👥 Members", value=str(guild.member_count), inline=True)
    embed.add_field(
        name="📺 Channels",
        value=str(len(guild.text_channels) + len(guild.voice_channels)),
        inline=True,
    )
    embed.add_field(name="🎭 Roles", value=str(len(guild.roles)), inline=True)
    embed.add_field(name="🌍 Owner", value=guild.owner.mention if guild.owner else "?", inline=True)
    embed.add_field(name="📅 Created", value=discord.utils.format_dt(guild.created_at, "R"), inline=True)
    embed.add_field(
        name="💎 Boosts",
        value=f"Level {guild.premium_tier} · {guild.premium_subscription_count or 0}",
        inline=True,
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="avatar", description="Show a member's profile picture.")
@app_commands.describe(member="Which member? Defaults to you.")
async def avatar(interaction: discord.Interaction, member: discord.Member = None):
    cmd_tick("avatar")
    member = member or interaction.user
    embed = discord.Embed(title=f"{member.display_name} 📸", color=0xA78BFA)
    embed.set_image(url=member.display_avatar.url)
    await interaction.response.send_message(embed=embed, ephemeral=True)


# ---------------------------------------------------------------------------
# Engagement — live button polls & reminders
# ---------------------------------------------------------------------------

_POLL_NUMBERS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]


class PollView(discord.ui.View):
    """A live poll rendered as numbered buttons. One vote per person — clicking
    another button moves their vote to that option."""

    def __init__(self, question: str, options: list, timeout: int = 300):
        super().__init__(timeout=timeout)
        self.question = question
        self.votes = {opt: set() for opt in options}
        self.message = None
        for i, opt in enumerate(options):
            self.add_item(PollButton(opt, _POLL_NUMBERS[i]))

    async def render(self, interaction: discord.Interaction):
        total = sum(len(v) for v in self.votes.values())
        lines = []
        for opt, voters in self.votes.items():
            count = len(voters)
            pct = round(count / total * 100) if total else 0
            bar = "▰" * (pct // 5) + "▱" * (20 - pct // 5)
            lines.append(f"{opt} — **{count}** vote(s) · {pct}%\n`{bar}`")
        embed = discord.Embed(
            title="📊 " + self.question,
            description=(
                "\n\n".join(lines) + f"\n\n**Total:** {total} vote(s) — click a number to vote!"
                if total else
                "No votes yet — click a number below to vote!"
            ),
            color=0xA78BFA,
        )
        embed.set_footer(text="One vote each · click another option to change it.")
        if self.message:
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.send_message(embed=embed, view=self)
            self.message = await interaction.original_response()

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        if self.message:
            try:
                await self.message.edit(content="⏰ Poll closed!", view=self)
            except discord.HTTPException:
                pass


class PollButton(discord.ui.Button):
    def __init__(self, option: str, number_emoji: str):
        self.option = option
        super().__init__(style=discord.ButtonStyle.secondary, label=option, emoji=number_emoji)

    async def callback(self, interaction: discord.Interaction):
        view: PollView = self.view
        for voters in view.votes.values():
            voters.discard(interaction.user.id)
        view.votes[self.option].add(interaction.user.id)
        await view.render(interaction)


@bot.tree.command(name="poll", description="Run a live vote with buttons.")
@app_commands.describe(
    question="The question people vote on",
    options="Comma-separated options, e.g. 'Yes, No' (2-9, default Yes/No)",
)
async def poll(interaction: discord.Interaction, question: str, options: str = ""):
    cmd_tick("poll")
    opts = [o.strip() for o in options.split(",") if o.strip()]
    if not opts:
        opts = ["Yes", "No"]
    opts = opts[:9]
    if len(opts) < 2:
        await interaction.response.send_message("A poll needs at least 2 options.", ephemeral=True)
        return
    question = question[:200]
    view = PollView(question, opts)
    await view.render(interaction)


_reminders = []  # [{at, channel_id, mention, what}]


async def reminder_loop():
    await bot.wait_until_ready()
    while not bot.is_closed():
        now = time.time()
        for r in [r for r in _reminders if r["at"] <= now]:
            _reminders.remove(r)
            try:
                channel = bot.get_channel(r["channel_id"])
                if channel is None and r.get("user_id"):
                    # DM reminder: open the DM channel (get_channel can't).
                    try:
                        user = await bot.fetch_user(int(r["user_id"]))
                        channel = user.dm_channel or await user.create_dm()
                    except (discord.NotFound, discord.HTTPException, ValueError):
                        channel = None
                if channel:
                    try:
                        await channel.send(f"{r['mention']} ⏳ **Reminder:** {r['what']}")
                    except (discord.Forbidden, discord.HTTPException):
                        pass
            except Exception:
                pass
        await asyncio.sleep(20)


@bot.tree.command(name="remind", description="Get pinged about something later.")
@app_commands.describe(what="What to remind you about", minutes="In how many minutes (1-1440)")
async def remind(interaction: discord.Interaction, what: str, minutes: int):
    cmd_tick("remind")
    minutes = max(1, min(minutes, 1440))
    channel_id = interaction.channel.id if interaction.guild else interaction.user.id
    _reminders.append({
        "at": time.time() + minutes * 60,
        "channel_id": channel_id,
        "user_id": interaction.user.id,
        "mention": interaction.user.mention,
        "what": what[:300],
    })
    await interaction.response.send_message(
        f"⏰ Got it — I'll ping you about **{what[:100]}** in {minutes} min.", ephemeral=True
    )


@bot.tree.command(name="8ball", description="Ask the magic 8-ball a question.")
@app_commands.describe(question="Your question")
async def eightball(interaction: discord.Interaction, question: str):
    cmd_tick("8ball")
    answers = [
        "🎱 It is certain.", "🎱 It is decidedly so.", "🎱 Without a doubt.",
        "🎱 Yes — definitely.", "🎱 You may rely on it.", "🎱 As I see it, yes.",
        "🎱 Most likely.", "🎱 Outlook good.", "🎱 Signs point to yes.",
        "🎱 Reply hazy, try again.", "🎱 Ask again later.", "🎱 Better not tell you now.",
        "🎱 Cannot predict now.", "🎱 Concentrate and ask again.",
        "🎱 Don't count on it.", "🎱 My reply is no.", "🎱 My sources say no.",
        "🎱 Outlook not so good.", "🎱 Very doubtful.",
    ]
    await interaction.response.send_message(
        f"> {question}\n{random.choice(answers)}"
    )


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

@bot.tree.command(name="dice", description="Roll some dice. Defaults to 1d6.")
async def dice(interaction: discord.Interaction, dice: str = "1d6"):
    cmd_tick("dice")
    m = re.fullmatch(r"(\d*)d(\d+)", dice.strip().lower())
    if not m:
        await interaction.response.send_message("Try something like `2d6` or `1d20`.", ephemeral=True)
        return
    count = int(m.group(1)) if m.group(1) else 1
    sides = int(m.group(2))
    count = max(1, min(count, 20))
    sides = max(2, min(sides, 1000000))
    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls)
    show = ", ".join(str(r) for r in rolls)
    name = interaction.user.display_name
    if count == 1:
        await interaction.response.send_message(f"🎲 **{name}** rolled **{total}** on a d{sides}.")
    else:
        await interaction.response.send_message(f"🎲 **{name}** rolled **{total}** ({show}) with `{count}d{sides}`.")


@bot.tree.command(name="coin", description="Flip a coin.")
async def coin(interaction: discord.Interaction):
    cmd_tick("coin")
    result = random.choice(["Heads", "Tails"])
    await interaction.response.send_message(f"🪙 **{interaction.user.display_name}** flipped **{result}**!")


@bot.tree.command(name="rps", description="Play rock, paper, scissors against the bot.")
@app_commands.describe(choice="rock, paper or scissors")
@app_commands.choices(choice=[
    app_commands.Choice(name="🪨 Rock", value="rock"),
    app_commands.Choice(name="📄 Paper", value="paper"),
    app_commands.Choice(name="✂️ Scissors", value="scissors"),
])
async def rps(interaction: discord.Interaction, choice: str):
    cmd_tick("rps")
    bot_choice = random.choice(["rock", "paper", "scissors"])
    emoji = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}
    wins = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    user_emoji = emoji.get(choice, choice)
    bot_emoji = emoji[bot_choice]
    if choice == bot_choice:
        outcome = "**It's a tie!** 🤝"
    elif wins[choice] == bot_choice:
        outcome = "**You win!** 🎉"
    else:
        outcome = "**I win!** 😏"
    await interaction.response.send_message(
        f"{user_emoji} You picked {choice} · {bot_emoji} I picked {bot_choice}\n{outcome}"
    )


_TRIVIA = [
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("How many continents are there?", "7"),
    ("What gas do plants absorb from the air?", "CO2"),
    ("What is the fastest land animal?", "Cheetah"),
    ("How many hearts does an octopus have?", "3"),
    ("What is the capital of Japan?", "Tokyo"),
    ("Which element has the chemical symbol 'O'?", "Oxygen"),
    ("How many colours are in a rainbow?", "7"),
    ("What is the longest river in the world?", "Nile"),
    ("How many strings does a guitar have?", "6"),
    ("What is the closest star to Earth?", "Sun"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("How many days are in a leap year?", "366"),
    ("What is the national animal of Australia?", "Kangaroo"),
    ("How many legs does a spider have?", "8"),
]

_trivia_answer_at = {}


@bot.tree.command(name="trivia", description="Answer a random trivia question.")
async def trivia(interaction: discord.Interaction):
    cmd_tick("trivia")
    question, answer = random.choice(_TRIVIA)
    key = (interaction.guild.id if interaction.guild else "dm", interaction.channel.id)
    _trivia_answer_at[key] = (answer, time.time())
    await interaction.response.send_message(
        f"❓ **Trivia:** {question}\nFirst correct reply wins! (Answer with `/answer <your answer>` within 30s.)"
    )


@bot.tree.command(name="answer", description="Answer the running trivia question.")
async def answer(interaction: discord.Interaction, answer_text: str):
    cmd_tick("answer")
    key = (interaction.guild.id if interaction.guild else "dm", interaction.channel.id)
    entry = _trivia_answer_at.get(key)
    if not entry:
        await interaction.response.send_message(
            "No trivia question is running here. Try `/trivia` first.",
            ephemeral=True,
        )
        return
    expected, when = entry
    if time.time() - when > 30:
        _trivia_answer_at.pop(key, None)
        await interaction.response.send_message("That round already ended — answer too slow! 🕐", ephemeral=True)
        return
    guess = answer_text.strip().lower()
    if guess != expected.lower():
        await interaction.response.send_message("Nope, not right. Try again! 🤔", ephemeral=True)
        return
    _trivia_answer_at.pop(key, None)
    await interaction.response.send_message(
        f"🎉 **{interaction.user.display_name}** got it! The answer was **{expected}**."
    )


_SLOT_SYMBOLS = ["🍒", "🍋", "🍉", "⭐", "💎", "7️⃣"]


@bot.tree.command(name="slot", description="Spin the slot machine.")
async def slot(interaction: discord.Interaction):
    cmd_tick("slot")
    roll = [random.choice(_SLOT_SYMBOLS) for _ in range(3)]
    line = "".join(roll)
    if roll[0] == roll[1] == roll[2]:
        if roll[0] == "7️⃣":
            verdict = "💥 **JACKPOT!** You hit the jackpot!"
        elif roll[0] == "💎":
            verdict = "✨ **BIG WIN!** Sparkling diamonds!"
        else:
            verdict = "🎉 **WINNER!** Triple match!"
    elif roll[0] == roll[1] or roll[1] == roll[2] or roll[0] == roll[2]:
        verdict = "👍 Close — two in a row!"
    else:
        verdict = "😅 No luck this time."
    await interaction.response.send_message(f"🎰 **{interaction.user.display_name}** spun:\n\n`{line}`\n\n{verdict}")


_BOARD_EMOJI = {"x": "❌", "o": "⭕", "": "·"}
_games = {}


def _board_view(board, show=True) -> str:
    return "```\n" + "\n".join(
        " ".join(_BOARD_EMOJI[k if show else ""] for k in board[y * 3:(y + 1) * 3])
        for y in range(3)
    ) + "\n```"


def _winner_of(board):
    lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6],
    ]
    for a, b, c in lines:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if all(board):
        return "draw"
    return None


@bot.tree.command(name="tictactoe", description="Play tic-tac-toe (X) against a friend (O).")
@app_commands.describe(opponent="The friend you want to play against")
async def tictactoe(interaction: discord.Interaction, opponent: discord.Member):
    cmd_tick("tictactoe")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/tictactoe` inside a server.", ephemeral=True)
        return
    if opponent.bot:
        await interaction.response.send_message("The bot doesn't play tic-tac-toe. Pick a human!", ephemeral=True)
        return
    if opponent == interaction.user:
        await interaction.response.send_message("You can't play against yourself. Pick a friend!", ephemeral=True)
        return
    key = (interaction.guild.id, interaction.channel.id)
    if key in _games or key in _hangman_games:
        await interaction.response.send_message(
            "A game is already running in this channel. Use `/move` to play.", ephemeral=True)
        return
    _games[key] = {
        "board": [""] * 9,
        "players": {"x": interaction.user.id, "o": opponent.id},
        "turn": "x",
        "last_cell": None,
    }
    await interaction.response.send_message(
        f"⭕ **Tic-tac-toe:** {interaction.user.mention} (❌) vs {opponent.mention} (⭕)\n"
        f"{_board_view(_games[key]['board'])}\n"
        f"{interaction.user.mention} to move — pick a square with `/move 1-9`."
    )


@bot.tree.command(name="move", description="Play a square in the running tic-tac-toe game (1–9).")
@app_commands.describe(cell="Square number 1–9 (top-left to bottom-right)")
async def move(interaction: discord.Interaction, cell: int):
    cmd_tick("move")
    if interaction.guild is None:
        await interaction.response.send_message("Only in servers.", ephemeral=True)
        return
    key = (interaction.guild.id, interaction.channel.id)
    game = _games.get(key)
    if not game:
        await interaction.response.send_message(
            "No game running in this channel. Start one with `/tictactoe @friend`.", ephemeral=True)
        return
    if interaction.user.id != game["players"][game["turn"]]:
        await interaction.response.send_message("It's not your turn.", ephemeral=True)
        return
    if not 1 <= cell <= 9 or game["board"][cell - 1]:
        await interaction.response.send_message("That square is taken or out of range.", ephemeral=True)
        return
    game["board"][cell - 1] = game["turn"]
    winner = _winner_of(game["board"])
    who = interaction.user.mention
    if winner:
        _games.pop(key, None)
        if winner == "draw":
            await interaction.response.send_message(
                f"{_board_view(game['board'])}\n🤝 **It's a draw!**"
            )
        else:
            mark = "❌" if winner == "x" else "⭕"
            await interaction.response.send_message(
                f"{_board_view(game['board'])}\n🎉 **{who} wins** with {mark}!"
            )
        return
    game["turn"] = "o" if game["turn"] == "x" else "x"
    next_mark = "⭕" if game["turn"] == "o" else "❌"
    next_player = game["players"][game["turn"]]
    await interaction.response.send_message(
        f"{_board_view(game['board'])}\n"
        f"<@{next_player}> to move ({next_mark}) — `/move 1-9`."
    )


# ---------------------------------------------------------------------------
# Hangman — solo vs bot, together (co-op), race (head-to-head), custom word
# /hangman [opponent] [mode] + A–Z menu + /hm_guess + ✏️ Set-word modal
# ---------------------------------------------------------------------------

_HM_MAX_WRONG = 6
_hangman_games = {}

_HM_STAGES = [
    "```\n  +---+\n  |   |\n      |\n      |\n      |\n      |\n=========\n```",
    "```\n  +---+\n  |   |\n  O   |\n      |\n      |\n      |\n=========\n```",
    "```\n  +---+\n  |   |\n  O   |\n  |   |\n      |\n      |\n=========\n```",
    "```\n  +---+\n  |   |\n  O   |\n /|   |\n      |\n      |\n=========\n```",
    "```\n  +---+\n  |   |\n  O   |\n /|\\  |\n      |\n      |\n=========\n```",
    "```\n  +---+\n  |   |\n  O   |\n /|\\  |\n /    |\n      |\n=========\n```",
    "```\n  +---+\n  |   |\n  O   |\n /|\\  |\n / \\  |\n      |\n=========\n```",
]

_HM_WORDS = [
    ("APPLE", "a red or green fruit"), ("BANANA", "long yellow fruit"),
    ("ORANGE", "citrus fruit + colour"), ("GRAPE", "tiny vine fruit"),
    ("MANGO", "tropical stone fruit"), ("PIZZA", "cheesy Italian dish"),
    ("BREAD", "baked loaf"), ("HONEY", "made by bees"),
    ("TIGER", "striped big cat"), ("PANDA", "bamboo-eating bear"),
    ("KOALA", "Australian tree marsupial"), ("DOLPHIN", "smart sea mammal"),
    ("PENGUIN", "flightless Antarctic bird"), ("GIRAFFE", "tallest animal"),
    ("ZEBRA", "striped horse-like animal"), ("KANGAROO", "hopping marsupial"),
    ("CASTLE", "medieval fortress"), ("BRIDGE", "spans a river"),
    ("ROCKET", "flies to space"), ("PLANET", "Earth is one"),
    ("OCEAN", "vast salty water"), ("DESERT", "sandy dry place"),
    ("VOLCANO", "erupting mountain"), ("GUITAR", "six-string instrument"),
    ("PIANO", "keys instrument"), ("DRUM", "you hit it for rhythm"),
    ("SOCCER", "world's most played sport"), ("TENNIS", "racket + net sport"),
    ("WIZARD", "spell caster"), ("DRAGON", "fire-breathing beast"),
    ("PIRATE", "sails with a treasure map"), ("ROBOT", "metal machine helper"),
    ("COMPUTER", "runs Discord bots"), ("KEYBOARD", "typing tool"),
    ("PYTHON", "programming language + snake"), ("DISCORD", "where you play this"),
    ("CANDLE", "wax + flame"), ("LANTERN", "portable light"),
    ("BOOK", "pages of story"), ("PENCIL", "write + erase tool"),
    ("GARDEN", "flowers grow here"), ("FOREST", "dense trees"),
    ("MOUNTAIN", "tall peak"), ("RIVER", "flowing water"),
    ("CLOUD", "floats in the sky"), ("RAINBOW", "colour arc after rain"),
    ("SNOWMAN", "built from snowballs"), ("BEACH", "sand + waves"),
    ("ISLAND", "land in the sea"), ("TREASURE", "buried chest of gold"),
    ("FRIEND", "what co-op games need"),
]


def _hm_key(interaction: discord.Interaction):
    g = interaction.guild.id if interaction.guild else "dm"
    return (g, interaction.channel.id)


def _hm_new_board() -> dict:
    return {"guessed": set(), "wrong": [], "lives": _HM_MAX_WRONG, "scores": {}}


def _hm_masked(word: str, guessed: set) -> str:
    return "  ".join(c if c in guessed else "▁" for c in word)


def _hm_hearts(lives: int) -> str:
    return "❤️" * lives + "🖤" * (_HM_MAX_WRONG - lives)


def _hm_board_block(word: str, board: dict) -> str:
    wrong_n = len(board["wrong"])
    return (
        f"{_HM_STAGES[wrong_n]}\n"
        f"**Word:** `{_hm_masked(word, board['guessed'])}`\n\n"
        f"**Wrong ({wrong_n}/{_HM_MAX_WRONG}):** "
        f"{', '.join(f'`{c}`' for c in board['wrong']) if board['wrong'] else '—'}\n"
        f"**Lives:** {_hm_hearts(board['lives'])} `{board['lives']} left`\n"
    )


def _hm_solved(word: str, board: dict) -> bool:
    return "▁" not in _hm_masked(word, board["guessed"]).replace(" ", "")


def _hm_my_boards(game: dict, user_id: int) -> list:
    """Board keys this user may guess on. Race = own board only."""
    if game["mode"] == "race":
        k = str(user_id)
        return [k] if k in game["boards"] else []
    return ["shared"]


def _hm_embed(game: dict, *, over=None, winner_id=None) -> discord.Embed:
    mode = game["mode"]
    if mode == "race":
        parts = []
        for uid in (str(game["host_id"]), str(game.get("opponent_id") or "")):
            b = game["boards"].get(uid)
            if not b:
                continue
            parts.append(f"**<@{uid}>**\n" + _hm_board_block(game["word"], b))
        desc = "\n".join(parts)
    else:
        desc = _hm_board_block(game["word"], game["boards"]["shared"])
    if over is None:
        titles = {
            "solo": "🎯 Hangman — guess a letter!",
            "together": "🤝 Hangman together — guess a letter!",
            "race": "🏁 Hangman race — fastest solver wins!",
        }
        color = 0x5865F2
        title = titles.get(mode, "🎯 Hangman — guess a letter!")
        embed = discord.Embed(title=title, description=desc, color=color)
        if mode == "race":
            top = []
            for uid, b in game["boards"].items():
                n = sum(b["scores"].values())
                if n:
                    top.append((uid, n))
            top.sort(key=lambda kv: -kv[1])
            if top:
                embed.add_field(name="⚡ Score", value=" · ".join(f"<@{u}> ({n}✅)" for u, n in top[:4]), inline=False)
        else:
            b = game["boards"]["shared"]
            top = sorted(b["scores"].items(), key=lambda kv: -kv[1])[:3]
            if top:
                embed.add_field(name="Top guessers", value=", ".join(f"<@{u}> ({n}✅)" for u, n in top), inline=False)
        clue = game["hint"] if game.get("hint_revealed") else "Use 💡 Hint to reveal the clue!"
        embed.add_field(name="💡 Clue", value=clue, inline=False)
        if mode == "race":
            players = f"<@{game['host_id']}> 🏁 vs <@{game.get('opponent_id')}> — guess on YOUR board (menu or `/hm_guess`)"
        elif mode == "together":
            players = f"<@{game['host_id']}> + everyone — shared board, shared lives!"
        else:
            players = f"<@{game['host_id']}> (solo vs Quaestio 🤖 + spectators welcome)"
        embed.add_field(name="Players", value=players, inline=False)
        embed.set_footer(text="Game ends in 5 min idle · host can ✏️ set a custom word")
    elif over == "win":
        who = f"<@{winner_id}>" if winner_id else "You"
        embed = discord.Embed(title=f"🎉 {who} cracked it!" if winner_id else "🎉 You cracked it!",
                              description=desc + f"\n**Word was `{game['word']}`**", color=0x57F287)
    else:
        embed = discord.Embed(title="💀 Game over!", description=desc + f"\n**Word was `{game['word']}`** — better luck next time!", color=0xED4245)
    return embed


class HangmanLetterSelect(discord.ui.Select):
    def __init__(self, game: dict, user_id: int = 0):
        if game["mode"] == "race" and user_id:
            b = game["boards"].get(str(user_id), _hm_new_board())
            guessed = b["guessed"]
        else:
            guessed = game["boards"]["shared"]["guessed"] if "shared" in game["boards"] else set()
        remaining = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in guessed]
        opts = [discord.SelectOption(label=c, value=c) for c in remaining[:25]]
        super().__init__(placeholder="Pick a letter…" if opts else "No letters left",
                         min_values=1, max_values=1,
                         options=opts or [discord.SelectOption(label="—", value="—")],
                         disabled=not opts)

    async def callback(self, interaction: discord.Interaction):
        await _hm_apply_guess(interaction, self.values[0], via_view=True)


class HangmanWordModal(discord.ui.Modal, title="Set a custom word"):
    word = discord.ui.TextInput(label="Secret word (letters only, 3–12)",
                                placeholder="e.g. PINEAPPLE", min_length=3, max_length=12)

    async def on_submit(self, interaction: discord.Interaction):
        cmd_tick("hm_word")
        key = _hm_key(interaction)
        game = _hangman_games.get(key)
        if not game or interaction.user.id != game["host_id"]:
            await interaction.response.send_message("Only the host can set the word.", ephemeral=True)
            return
        w = (self.word.value or "").strip().upper()
        if not w.isalpha():
            await interaction.response.send_message("Letters A–Z only, please!", ephemeral=True)
            return
        game["word"] = w
        game["hint"] = "a custom word from the host 😏"
        game["hint_revealed"] = True
        for b in game["boards"].values():
            b["guessed"] = set()
            b["wrong"] = []
            b["lives"] = _HM_MAX_WRONG
            b["scores"] = {}
        view = game.get("view")
        embed = _hm_embed(game)
        if view:
            view.refresh_select(game, interaction.user.id)
            await interaction.response.edit_message(embed=embed, view=view)
        else:
            await interaction.response.send_message(embed=embed)


class HangmanView(discord.ui.View):
    def __init__(self, key, timeout: int = 300):
        super().__init__(timeout=timeout)
        self.key = key
        self.message = None

    def refresh_select(self, game: dict, user_id: int = 0):
        for child in list(self.children):
            if isinstance(child, HangmanLetterSelect):
                self.remove_item(child)
        self.add_item(HangmanLetterSelect(game, user_id))

    @discord.ui.button(label="Hint", emoji="💡", style=discord.ButtonStyle.secondary)
    async def hint(self, interaction: discord.Interaction, button: discord.ui.Button):
        cmd_tick("hm_hint")
        game = _hangman_games.get(self.key)
        if not game:
            await interaction.response.send_message("No hangman game here. Start one with `/hangman`.", ephemeral=True)
            return
        if game.get("hint_used"):
            await interaction.response.send_message("Hint already used!", ephemeral=True)
            return
        keys = _hm_my_boards(game, interaction.user.id)
        if not keys:
            await interaction.response.send_message("Spectators can't use hints in a race!", ephemeral=True)
            return
        b = game["boards"][keys[0]]
        hidden = [c for c in game["word"] if c not in b["guessed"]]
        if not hidden:
            await interaction.response.send_message("Nothing left to reveal!", ephemeral=True)
            return
        game["hint_used"] = True
        game["hint_revealed"] = True
        b["guessed"].add(random.choice(hidden))
        await _hm_after_move(interaction, game, keys[0], via_view=True)

    @discord.ui.button(label="Set word", emoji="✏️", style=discord.ButtonStyle.secondary)
    async def setword(self, interaction: discord.Interaction, button: discord.ui.Button):
        game = _hangman_games.get(self.key)
        if not game or interaction.user.id != game["host_id"]:
            await interaction.response.send_message("Only the host can set the word.", ephemeral=True)
            return
        await interaction.response.send_modal(HangmanWordModal())

    @discord.ui.button(label="End", emoji="🛑", style=discord.ButtonStyle.danger)
    async def end_game(self, interaction: discord.Interaction, button: discord.ui.Button):
        cmd_tick("hm_stop")
        game = _hangman_games.get(self.key)
        if not game:
            await interaction.response.send_message("No game running.", ephemeral=True)
            return
        if interaction.user.id not in (game["host_id"], game.get("opponent_id") or 0) and not interaction.user.guild_permissions.manage_messages:
            await interaction.response.send_message("Only the host/opponent (or a mod) can end it.", ephemeral=True)
            return
        _hangman_games.pop(self.key, None)
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(embed=_hm_embed(game, over="lose"), view=self)
        super().stop()

    async def on_timeout(self):
        game = _hangman_games.pop(self.key, None)
        for child in self.children:
            child.disabled = True
        if self.message and game:
            try:
                await self.message.edit(content="⏰ Hangman timed out (5 min idle).", embed=_hm_embed(game, over="lose"), view=self)
            except discord.HTTPException:
                pass


def _hm_finish_board(game: dict, bkey: str, user_id: int):
    """Check one board: returns ('win', uid) / ('dead', None) / (None, None)."""
    b = game["boards"][bkey]
    if _hm_solved(game["word"], b):
        return "win", user_id if game["mode"] == "race" else None
    if b["lives"] <= 0:
        return "dead", None
    return None, None


async def _hm_after_move(interaction: discord.Interaction, game: dict, bkey: str, *, via_view: bool):
    """Shared win/lose/continue renderer for View + /hm_guess."""
    key = _hm_key(interaction)
    view = game.get("view")
    b = game["boards"][bkey]
    uid = interaction.user.id

    async def _send(embed, v):
        if via_view:
            await interaction.response.edit_message(embed=embed, view=v)
        else:
            await interaction.response.send_message(embed=embed)
            try:
                if v and v.message:
                    await v.message.edit(embed=embed, view=v)
            except discord.HTTPException:
                pass

    def _lock(v):
        if v:
            for child in v.children:
                child.disabled = True
            v.stop()

    if _hm_solved(game["word"], b):
        _hangman_games.pop(key, None)
        _lock(view)
        await _send(_hm_embed(game, over="win", winner_id=uid if game["mode"] == "race" else None), view)
        return
    if b["lives"] <= 0:
        if game["mode"] == "race":
            alive = [k for k, bb in game["boards"].items()
                     if not _hm_solved(game["word"], bb) and bb["lives"] > 0]
            if not alive:
                _hangman_games.pop(key, None)
                _lock(view)
                await _send(_hm_embed(game, over="lose"), view)
                return
            # This racer is out — the other board plays on.
            embed = _hm_embed(game)
            embed.add_field(name="💀 Eliminated", value=f"<@{uid}> is out of lives!", inline=False)
            if view:
                view.refresh_select(game, uid)
            if via_view:
                await interaction.response.edit_message(embed=embed, view=view)
            else:
                await interaction.response.send_message(embed=embed)
            return
        _hangman_games.pop(key, None)
        _lock(view)
        await _send(_hm_embed(game, over="lose"), view)
        return
    embed = _hm_embed(game)
    if view:
        view.refresh_select(game, uid)
        if via_view:
            await interaction.response.edit_message(embed=embed, view=view)
        else:
            await interaction.response.send_message(embed=embed)
            try:
                if view.message:
                    await view.message.edit(embed=embed, view=view)
            except discord.HTTPException:
                pass
    else:
        await interaction.response.send_message(embed=embed)


async def _hm_apply_guess(interaction: discord.Interaction, raw: str, *, via_view: bool):
    cmd_tick("hm_guess")
    key = _hm_key(interaction)
    game = _hangman_games.get(key)
    if not game:
        if via_view:
            await interaction.response.edit_message(content="Game already ended.", view=None)
        else:
            await interaction.response.send_message("No hangman game in this channel. Start one with `/hangman`.", ephemeral=True)
        return
    letter = (raw or "").strip().upper()
    if len(letter) != 1 or not letter.isalpha():
        await interaction.response.send_message("Guess one letter A–Z.", ephemeral=True)
        return
    keys = _hm_my_boards(game, interaction.user.id)
    if not keys:
        await interaction.response.send_message("Racers only — you're spectating this race! 👀", ephemeral=True)
        return
    bkey = keys[0]
    b = game["boards"][bkey]
    if _hm_solved(game["word"], b) or b["lives"] <= 0:
        await interaction.response.send_message("Your board is already finished!", ephemeral=True)
        return
    if letter in b["guessed"]:
        await interaction.response.send_message(f"`{letter}` was already tried — pick another!", ephemeral=True)
        return
    b["guessed"].add(letter)
    if letter in game["word"]:
        b["scores"][interaction.user.id] = b["scores"].get(interaction.user.id, 0) + game["word"].count(letter)
    else:
        b["wrong"].append(letter)
        b["lives"] -= 1
    await _hm_after_move(interaction, game, bkey, via_view=via_view)


@bot.tree.command(name="hangman", description="Hangman: solo, together, race, or custom word.")
@app_commands.describe(opponent="Friend to race / play with (race needs one)",
                       mode="How to play: solo, together (co-op), or race (head-to-head)")
@app_commands.choices(mode=[app_commands.Choice(name="🧍 Solo vs Quaestio", value="solo"),
                            app_commands.Choice(name="🤝 Together (co-op)", value="together"),
                            app_commands.Choice(name="🏁 Race (head-to-head)", value="race")])
async def hangman(interaction: discord.Interaction, opponent: discord.Member | None = None,
                  mode: str = "solo"):
    cmd_tick("hangman")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/hangman` inside a server.", ephemeral=True)
        return
    if mode == "race" and (not opponent or opponent.bot or opponent == interaction.user):
        await interaction.response.send_message("Race mode needs a real friend — tag an opponent!", ephemeral=True)
        return
    if opponent and (opponent.bot or opponent == interaction.user):
        await interaction.response.send_message("Tag a real friend (not yourself or a bot) — or leave it empty for solo.", ephemeral=True)
        return
    key = _hm_key(interaction)
    if key in _hangman_games or key in _games:
        await interaction.response.send_message("A game is already running in this channel. Finish it first!", ephemeral=True)
        return
    word, hint = random.choice(_HM_WORDS)
    view = HangmanView(key, timeout=300)
    if mode == "race":
        boards = {str(interaction.user.id): _hm_new_board(), str(opponent.id): _hm_new_board()}
        title = f"🏁 **Hangman race:** {interaction.user.mention} vs {opponent.mention} — fastest solver wins!"
    elif mode == "together":
        boards = {"shared": _hm_new_board()}
        title = f"🤝 **Hangman together:** {interaction.user.mention} + everyone — shared board, shared lives!"
    else:
        mode = "solo"
        boards = {"shared": _hm_new_board()}
        title = (f"🎯 **Hangman:** {interaction.user.mention} vs {opponent.mention}"
                 if opponent else f"🎯 **Hangman:** {interaction.user.mention} vs Quaestio 🤖")
    game = {
        "mode": mode, "word": word, "hint": hint, "hint_revealed": False, "hint_used": False,
        "host_id": interaction.user.id,
        "opponent_id": opponent.id if opponent else None,
        "boards": boards, "view": view,
    }
    _hangman_games[key] = game
    view.refresh_select(game, interaction.user.id)
    await interaction.response.send_message(title, embed=_hm_embed(game), view=view)
    view.message = await interaction.original_response()


@bot.tree.command(name="hm_guess", description="Guess a letter in the running hangman game.")
@app_commands.describe(letter="One letter A–Z")
async def hm_guess(interaction: discord.Interaction, letter: str):
    await _hm_apply_guess(interaction, letter, via_view=False)



# ---------------------------------------------------------------------------
# Would-You-Rather — open vote, 60s, early majority close
# ---------------------------------------------------------------------------
_WYR_QUESTIONS = [
    ("🛸 Live on Mars", "🌊 Live under the ocean"),
    ("🧠 Read minds", "👻 Be invisible"),
    ("🐉 Own a dragon", "🦄 Own a unicorn"),
    ("🍕 Eat only pizza forever", "🍔 Eat only burgers forever"),
    ("✈️ Teleport anywhere", "⏳ Time-travel once"),
    ("🎤 Be famous singer", "🎬 Be famous actor"),
    ("❄️ Always be cold", "🔥 Always be hot"),
    ("🐱 Talk to cats", "🐶 Talk to dogs"),
    ("🌙 Never sleep", "🍩 Never eat"),
    ("🏝️ Desert island $1M", "🏙️ City life $100k"),
    ("🧙 Have magic powers", "🤖 Have robot army"),
    ("📚 Know every book", "🗣️ Speak every language"),
    ("🦸 Super strength", "⚡ Super speed"),
    ("🎮 Pro gamer", "🏆 Pro athlete"),
    ("👽 Meet aliens", "🦕 Meet dinosaurs"),
    ("🍫 Chocolate rain", "🧀 Cheese snow"),
    ("🚀 Go to space", "🤿 Explore deep sea"),
    ("🎨 Be art genius", "🎵 Be music genius"),
    ("🐼 Panda sidekick", "🦊 Fox sidekick"),
    ("💸 Win lottery, no friends know", "📣 Win half, everyone celebrates"),
]

_WYR_AI_THEMES = ["sci-fi", "fantasy", "food", "travel", "superpowers", "animals", "school", "gaming"]

_wyr_games = {}
_wyr_last = {}

WYR_DURATION = 60
WYR_COOLDOWN = 10
WYR_EARLY_WIN = 6


def _wyr_bar(pct: int, width: int = 12) -> str:
    filled = max(0, min(width, round(pct / 100 * width)))
    return "▰" * filled + "▱" * (width - filled)


def _wyr_embed(state) -> discord.Embed:
    a, b = state["a"], state["b"]
    va, vb = len(state["votes_a"]), len(state["votes_b"])
    total = va + vb
    pa = round(va / total * 100) if total else 50
    pb = 100 - pa if total else 50
    lead = "⚖️ Tied!" if va == vb else (f"🔵 A leads!" if va > vb else "🔴 B leads!")
    desc = (
        f"🔵 **A:** {a}\n`{_wyr_bar(pa)}` **{pa}%** ({va})\n\n"
        f"🔴 **B:** {b}\n`{_wyr_bar(pb)}` **{pb}%** ({vb})\n\n"
        f"{lead}\n👥 **{total} vote(s)** — tap a button! Change vote anytime."
    )
    if state.get("opponent_id"):
        desc += f"\n⚔️ Duel: <@{state['host_id']}> vs <@{state['opponent_id']}> — everyone votes!"
    embed = discord.Embed(title="🤔 Would You Rather…?", description=desc, color=0xA78BFA)
    embed.set_footer(text=f"Ends in {max(0, int(state['ends_at'] - time.time()))}s · first to {WYR_EARLY_WIN} wins early")
    return embed


class WYRButton(discord.ui.Button):
    def __init__(self, side: str):
        self.side = side
        emoji = "🔵" if side == "a" else "🔴"
        label = "A" if side == "a" else "B"
        super().__init__(style=discord.ButtonStyle.primary if side == "a" else discord.ButtonStyle.danger,
                         label=label, emoji=emoji)

    async def callback(self, interaction: discord.Interaction):
        cmd_tick("wyr_vote")
        view: WYRView = self.view
        state = view.state
        uid = interaction.user.id
        state["votes_a"].discard(uid)
        state["votes_b"].discard(uid)
        (state["votes_a"] if self.side == "a" else state["votes_b"]).add(uid)
        if len(state["votes_a"]) >= WYR_EARLY_WIN or len(state["votes_b"]) >= WYR_EARLY_WIN:
            await view.finish(interaction, reason="majority")
            return
        await view.render(interaction)


class WYRView(discord.ui.View):
    def __init__(self, state, timeout: int = WYR_DURATION):
        super().__init__(timeout=timeout)
        self.state = state
        self.message = None
        self.add_item(WYRButton("a"))
        self.add_item(WYRButton("b"))

    async def render(self, interaction: discord.Interaction):
        embed = _wyr_embed(self.state)
        if self.message:
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            await interaction.response.send_message(embed=embed, view=self)
            self.message = await interaction.original_response()

    async def finish(self, interaction_or_none, reason="time"):
        key = self.state["key"]
        _wyr_games.pop(key, None)
        for c in self.children:
            c.disabled = True
        va, vb = len(self.state["votes_a"]), len(self.state["votes_b"])
        winner = "🤝 Tie!" if va == vb else ("🔵 **A wins!** 🎉" if va > vb else "🔴 **B wins!** 🎉")
        embed = _wyr_embed(self.state)
        embed.add_field(name="🏁 Final", value=f"{winner}\nA: {va} · B: {vb}", inline=False)
        embed.set_footer(text=f"Closed ({reason}) · thanks for voting!")
        embed.color = 0x22C55E if va != vb else 0xA78BFA
        try:
            if interaction_or_none is not None and hasattr(interaction_or_none, "response"):
                try:
                    await interaction_or_none.response.edit_message(embed=embed, view=self)
                except discord.InteractionResponded:
                    await self.message.edit(embed=embed, view=self)
            elif self.message:
                await self.message.edit(embed=embed, view=self)
        except (discord.HTTPException, discord.NotFound):
            pass
        self.stop()

    async def on_timeout(self):
        await self.finish(None, reason="60s up")


async def _wyr_ai_pair(guild_id) -> tuple | None:
    """Optional AI-generated pair. Returns None on any failure -> fallback."""
    try:
        cfg = guild_ai_config(guild_id) if guild_id else None
        if not cfg:
            return None
        theme = random.choice(_WYR_AI_THEMES)
        prompt = (f"Write one family-friendly would-you-rather question, theme {theme}. "
                  "Reply with exactly two short options separated by ' | ', each under 8 words, PG, no explanation.")
        raw = await asyncio.wait_for(
            ask_ollama_any(cfg, prompt, temperature=0.8, max_tokens=60,
                           asker="wyr", system="You write short PG party-game prompts. No NSFW, no politics."),
            timeout=25)
        parts = [p.strip(" .\"'") for p in raw.replace("\n", " ").split("|")]
        if len(parts) >= 2 and all(2 <= len(p) <= 80 for p in parts[:2]):
            return parts[0][:80], parts[1][:80]
    except Exception:
        pass
    return None


@bot.tree.command(name="wouldyou", description="Would-You-Rather vote: A vs B with live bars.")
@app_commands.describe(opponent="Optional rival for a duel (everyone still votes)",
                       use_ai="Let AI invent the question (falls back to built-ins)")
async def wouldyou(interaction: discord.Interaction, opponent: discord.Member | None = None,
                   use_ai: bool = False):
    cmd_tick("wouldyou")
    now = time.time()
    if now - _wyr_last.get(interaction.user.id, 0) < WYR_COOLDOWN:
        await interaction.response.send_message(f"⏳ Slow down — try again in {WYR_COOLDOWN}s.", ephemeral=True)
        return
    _wyr_last[interaction.user.id] = now
    if interaction.guild is None:
        await interaction.response.send_message("Use `/wouldyou` inside a server.", ephemeral=True)
        return
    key = (interaction.guild.id, interaction.channel.id)
    if key in _wyr_games:
        await interaction.response.send_message("⚔️ A Would-You-Rather is already running here — vote on it!", ephemeral=True)
        return
    if opponent and (opponent.bot or opponent == interaction.user):
        await interaction.response.send_message("Pick another human as rival (or leave empty for open vote).", ephemeral=True)
        return

    await interaction.response.defer(thinking=False)
    a, b = random.choice(_WYR_QUESTIONS)
    if use_ai:
        pair = await _wyr_ai_pair(interaction.guild.id)
        if pair:
            a, b = pair

    state = {"key": key, "a": a, "b": b, "votes_a": set(), "votes_b": set(),
             "host_id": interaction.user.id,
             "opponent_id": opponent.id if opponent else None,
             "ends_at": time.time() + WYR_DURATION}
    _wyr_games[key] = state
    view = WYRView(state)
    embed = _wyr_embed(state)
    msg = await interaction.followup.send(embed=embed, view=view)
    view.message = msg


# ---------------------------------------------------------------------------
# Truth or Dare — rotation, AI w/ fallback lists, pass/skip
# ---------------------------------------------------------------------------
_TOD_TRUTHS = [
    "What's the funniest text you've ever sent by accident?",
    "What's a talent you wish you had?",
    "What's your most-used emoji and why?",
    "What's the weirdest food combo you love?",
    "What's a movie you can quote from memory?",
    "What's your earliest childhood memory?",
    "What's the best gift you've ever received?",
    "What's a skill you learned from YouTube?",
    "What's your dream vacation?",
    "What's the silliest fear you had as a kid?",
    "What's your favorite family tradition?",
    "What's a song you sing in the shower?",
    "What's the nicest thing a friend did for you?",
    "What's your favorite game right now?",
    "What would you do with $1,000 today?",
    "What's the best advice you've ever gotten?",
    "What's a hobby you want to try?",
    "What's your favorite breakfast food?",
    "What's the coolest place you've visited?",
    "What's something you're proud of this year?",
    "What's your go-to comfort snack?",
    "What's a book or show you recommend to everyone?",
    "What's the funniest nickname you've had?",
    "What's your dream pet?",
    "What's one thing on your bucket list?",
    "What's your favorite season and why?",
    "What's the bravest thing you've done?",
    "What's a small thing that always makes you smile?",
    "What's your hidden talent?",
    "What's the kindest thing you've done this month?",
]
_TOD_DARES = [
    "Send a compliment to the last person who messaged here.",
    "Do 10 jumping jacks and report back.",
    "Type with your elbows for your next 3 messages.",
    "Draw a cat with your eyes closed and describe it.",
    "Speak in pirate voice for your next 3 messages. 🏴‍☠️",
    "Rank the 3 people above you from funniest to most serious.",
    "Tell a 2-sentence spooky story. 👻",
    "Do your best robot dance — describe it in 3 emojis.",
    "Say the alphabet backwards as far as you can.",
    "Give the bot a new nickname for today.",
    "Post your best knock-knock joke.",
    "Invent a superhero name for the player above you.",
    "Talk in ALL CAPS for your next 2 messages.",
    "Describe your day as a movie trailer. 🎬",
    "Send a voice-note-style message using only emojis (5+).",
    "Compliment everyone's avatar in one message each.",
    "Do 5 push-ups (or 5 silly stretches) and confirm.",
    "Write a haiku about pizza. 🍕",
    "Pretend you're a sports commentator for 2 messages.",
    "Name 5 countries in 10 seconds — go!",
    "Do your best villain laugh in text. 😈",
    "Invent a handshake for you + the host (describe it).",
    "Say something nice about each voter so far.",
    "Draw ASCII art of your mood right now.",
    "Tell us your best dad joke.",
    "Act out (in text) opening a treasure chest. What's inside?",
    "Give a 10-second pep talk to the whole channel. 📣",
    "Swap your nickname to something silly for 10 min (if allowed).",
    "Teach the chat one word in another language.",
    "Predict the next Would-You-Rather winner. 🔮",
]

_tod_games = {}
_tod_last = {}

TOD_COOLDOWN = 8
TOD_PASS_LIMIT = 2


def _tod_embed(player_mention: str, kind: str, text: str, round_n: int, passes_left: int) -> discord.Embed:
    color = 0x38BDF8 if kind == "truth" else 0xF472B6
    emoji = "💭" if kind == "truth" else "🔥"
    title = f"{emoji} {kind.upper()} for {player_mention}"
    embed = discord.Embed(title=title, description=f"**{text}**", color=color)
    embed.add_field(name="Round", value=f"#{round_n}", inline=True)
    embed.add_field(name="Passes left", value=f"{passes_left} ⏭️", inline=True)
    embed.set_footer(text="Truth = answer honestly · Dare = do it or pass · Buttons below!")
    return embed


async def _tod_ai_prompt(guild_id, kind: str):
    try:
        cfg = guild_ai_config(guild_id)
        if not cfg:
            return None
        prompt = (f"Write one short family-friendly {kind} prompt for a Discord party game. "
                  "PG-13 max, no romance/risqué, no personal data, no dangerous stunts, under 20 words. Prompt only.")
        raw = await asyncio.wait_for(
            ask_ollama_any(cfg, prompt, temperature=0.8, max_tokens=60,
                           asker="tod", system="You write safe PG party prompts. Never NSFW, never mean, never unsafe."),
            timeout=25)
        clean = raw.strip().strip("\"'").split("\n")[0][:200]
        banned = ("sex", "kiss", "drink", "alcohol", "naked", "suicide", "self-harm")
        if len(clean) < 8 or any(w in clean.lower() for w in banned):
            return None
        return clean
    except Exception:
        return None


async def _tod_pick(guild_id, kind: str, use_ai: bool):
    if kind == "random":
        kind = random.choice(["truth", "dare"])
    if use_ai:
        ai = await _tod_ai_prompt(guild_id, kind)
        if ai:
            return kind, ai
    pool = _TOD_TRUTHS if kind == "truth" else _TOD_DARES
    return kind, random.choice(pool)


class TODView(discord.ui.View):
    def __init__(self, key, timeout: int = 120):
        super().__init__(timeout=timeout)
        self.key = key
        self.message = None

    def _game(self):
        return _tod_games.get(self.key)

    async def _deal(self, interaction: discord.Interaction, kind: str):
        game = self._game()
        if not game:
            await interaction.response.send_message("No ToD game here — start with `/tod`.", ephemeral=True)
            return
        if interaction.user.id not in (game["queue"][game["idx"]], game["host_id"]):
            await interaction.response.send_message("It's not your turn — wait for your round! 👀", ephemeral=True)
            return
        # Defer FIRST: AI prompt generation can take 25s, interactions die in 3s.
        try:
            if not interaction.response.is_done():
                await interaction.response.defer()
            deferred = True
        except (discord.NotFound, discord.HTTPException):
            deferred = False
        use_ai = game.get("use_ai", True)
        real_kind, text = await _tod_pick(interaction.guild.id if interaction.guild else None, kind, use_ai)
        pid = game["queue"][game["idx"]]
        game["current"] = {"type": real_kind, "text": text, "player": pid}
        game["round"] += 1
        embed = _tod_embed(f"<@{pid}>", real_kind, text, game["round"], TOD_PASS_LIMIT - game.get("passes_used", 0))
        try:
            if deferred:
                await interaction.edit_original_response(embed=embed, view=self)
            else:
                await interaction.response.edit_message(embed=embed, view=self)
        except (discord.NotFound, discord.HTTPException):
            try:
                if self.message:
                    await self.message.edit(embed=embed, view=self)
            except (discord.HTTPException, discord.NotFound):
                pass
        if self.message is None:
            try:
                self.message = await interaction.original_response()
            except discord.HTTPException:
                pass

    async def _advance(self, interaction: discord.Interaction):
        game = self._game()
        if not game:
            return
        game["idx"] = (game["idx"] + 1) % len(game["queue"])
        game["passes_used"] = 0
        nxt = game["queue"][game["idx"]]
        embed = discord.Embed(title="🎲 Truth or Dare",
                              description=f"<@{nxt}>'s turn! Pick **Truth**, **Dare**, or **Random** below. 👇",
                              color=0xA78BFA)
        embed.set_footer(text=f"Players: {len(game['queue'])} · Round #{game['round'] + 1}")
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(label="Truth", emoji="💭", style=discord.ButtonStyle.primary)
    async def truth(self, interaction: discord.Interaction, button: discord.ui.Button):
        cmd_tick("tod_truth")
        await self._deal(interaction, "truth")

    @discord.ui.button(label="Dare", emoji="🔥", style=discord.ButtonStyle.danger)
    async def dare(self, interaction: discord.Interaction, button: discord.ui.Button):
        cmd_tick("tod_dare")
        await self._deal(interaction, "dare")

    @discord.ui.button(label="Random", emoji="🎲", style=discord.ButtonStyle.secondary)
    async def random_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        cmd_tick("tod_random")
        await self._deal(interaction, "random")

    @discord.ui.button(label="Pass", emoji="⏭️", style=discord.ButtonStyle.secondary)
    async def pass_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        game = self._game()
        if not game or not game.get("current"):
            await interaction.response.send_message("Nothing to pass yet — draw first!", ephemeral=True)
            return
        if interaction.user.id != game["current"]["player"] and interaction.user.id != game["host_id"]:
            await interaction.response.send_message("Only the current player can pass.", ephemeral=True)
            return
        game["passes_used"] = game.get("passes_used", 0) + 1
        if game["passes_used"] > TOD_PASS_LIMIT:
            await interaction.response.send_message("❌ No passes left — do it or Skip to next player!", ephemeral=True)
            return
        await self._deal(interaction, game["current"]["type"])

    @discord.ui.button(label="Next player", emoji="⏩", style=discord.ButtonStyle.success)
    async def next_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        cmd_tick("tod_next")
        await self._advance(interaction)

    async def on_timeout(self):
        for c in self.children:
            c.disabled = True
        if self.message:
            try:
                await self.message.edit(view=self)
            except (discord.HTTPException, discord.NotFound):
                pass
        _tod_games.pop(self.key, None)


async def _tod_start(interaction: discord.Interaction, player: discord.Member | None = None,
                     choice: str = "none", use_ai: bool = True):
    """Shared implementation for /truthordare + /tod (NOT a command itself)."""
    cmd_tick("truthordare")
    now = time.time()
    if now - _tod_last.get(interaction.user.id, 0) < TOD_COOLDOWN:
        await interaction.response.send_message("⏳ Slow down — ToD has a short cooldown.", ephemeral=True)
        return
    _tod_last[interaction.user.id] = now
    if interaction.guild is None:
        await interaction.response.send_message("Use `/tod` inside a server.", ephemeral=True)
        return
    key = (interaction.guild.id, interaction.channel.id)
    starter = player.id if (player and not player.bot) else interaction.user.id
    game = _tod_games.get(key)
    if game is None:
        game = {"queue": [starter], "idx": 0, "host_id": interaction.user.id,
                "round": 0, "current": None, "passes_used": 0, "use_ai": use_ai}
        _tod_games[key] = game
    elif starter not in game["queue"]:
        game["queue"].append(starter)
    view = TODView(key, timeout=180)
    if choice == "none":
        embed = discord.Embed(title="🎲 Truth or Dare",
                              description=f"<@{starter}> starts! Pick **Truth**, **Dare**, or **Random**. 👇",
                              color=0xA78BFA)
        embed.set_footer(text=f"Players: {len(game['queue'])} · AI prompts {'on 🤖' if use_ai else 'off 📜'}")
        await interaction.response.send_message(embed=embed, view=view)
    else:
        # Defer FIRST: AI generation can outlive the 3s interaction window.
        await interaction.response.defer(thinking=False)
        real_kind, text = await _tod_pick(interaction.guild.id, choice, use_ai)
        game["current"] = {"type": real_kind, "text": text, "player": starter}
        game["round"] += 1
        embed = _tod_embed(f"<@{starter}>", real_kind, text, game["round"], TOD_PASS_LIMIT)
        await interaction.followup.send(embed=embed, view=view)
    try:
        view.message = await interaction.original_response()
    except discord.HTTPException:
        pass


@bot.tree.command(name="truthordare", description="Truth-or-Dare: multiplayer rotation + AI prompts.")
@app_commands.describe(player="First player (default: you). Others join by pressing buttons on their turn.",
                       choice="Draw immediately, or let buttons decide",
                       use_ai="Use AI for prompts when available (default True)")
@app_commands.choices(choice=[app_commands.Choice(name="Pick with buttons", value="none"),
                              app_commands.Choice(name="💭 Truth", value="truth"),
                              app_commands.Choice(name="🔥 Dare", value="dare"),
                              app_commands.Choice(name="🎲 Random", value="random")])
async def truthordare(interaction: discord.Interaction, player: discord.Member | None = None,
                      choice: str = "none", use_ai: bool = True):
    """Full name — the one normal users will actually try."""
    await _tod_start(interaction, player=player, choice=choice, use_ai=use_ai)


@bot.tree.command(name="tod", description="Short for /truthordare.")
@app_commands.describe(player="First player (default: you).",
                       choice="Draw immediately, or let buttons decide",
                       use_ai="Use AI for prompts when available (default True)")
@app_commands.choices(choice=[app_commands.Choice(name="Pick with buttons", value="none"),
                              app_commands.Choice(name="💭 Truth", value="truth"),
                              app_commands.Choice(name="🔥 Dare", value="dare"),
                              app_commands.Choice(name="🎲 Random", value="random")])
async def tod(interaction: discord.Interaction, player: discord.Member | None = None,
              choice: str = "none", use_ai: bool = True):
    """Short alias for /truthordare."""
    await _tod_start(interaction, player=player, choice=choice, use_ai=use_ai)


# ---------------------------------------------------------------------------
# Majority Rules — guess what the crowd picks (donate/vote-style questions)
# ---------------------------------------------------------------------------
_MAJORITY_QUESTIONS = [
    ("Do more people donate to animal shelters or food banks?", "🐾 Animal shelters", "🍽️ Food banks"),
    ("Do more people Google 'weather' or 'news' every morning?", "🌤️ Weather", "📰 News"),
    ("Do more people prefer working from home or the office?", "🏠 Home", "🏢 Office"),
    ("Do more people donate clothes or throw them away?", "👕 Donate", "🗑️ Toss"),
    ("Do more people Google symptoms or call a doctor first?", "🔍 Google it", "📞 Doctor"),
    ("Do more people tip 20%+ or under 15%?", "💰 20%+", "🪙 Under 15%"),
    ("Do more people vote in local elections or skip them?", "🗳️ Vote local", "😴 Skip"),
    ("Do more people donate to disaster relief or local schools?", "🌊 Disaster relief", "🏫 Local schools"),
    ("Do more people Google a recipe or wing it?", "📖 Recipe", "👨‍🍳 Wing it"),
    ("Do more people keep old phones or recycle them?", "📱 Keep", "♻️ Recycle"),
    ("Do more people prefer cats or dogs?", "🐱 Cats", "🐶 Dogs"),
    ("Do more people text or call?", "💬 Text", "📞 Call"),
    ("Do more people donate blood or say they will 'someday'?", "🩸 Donate", "📅 Someday"),
    ("Do more people Google the ending or watch it through?", "🔍 Spoil it", "🎬 Watch through"),
    ("Do more people tip street performers or walk past?", "🎺 Tip", "🚶 Walk past"),
    ("Do more people wake up to an alarm or naturally?", "⏰ Alarm", "🌅 Naturally"),
    ("Do more people Google 'how to' or ask a friend first?", "🔍 Google how-to", "🙋 Ask a friend"),
    ("Do more people donate to ocean cleanup or tree planting?", "🌊 Ocean cleanup", "🌳 Plant trees"),
    ("Do more people stream movies or go to the cinema?", "📺 Stream", "🎬 Cinema"),
    ("Do more people Google a word's spelling or guess it?", "🔍 Check spelling", "✍️ Guess it"),
    ("Do more people volunteer monthly or once a year?", "📅 Monthly", "🎉 Yearly"),
    ("Do more people use dark mode or light mode?", "🌙 Dark", "☀️ Light"),
    ("Do more people donate old books or keep them forever?", "📚 Donate books", "📖 Keep forever"),
    ("Do more people Google directions or trust memory?", "🗺️ Google maps", "🧠 Memory"),
    ("Do more people tip delivery drivers extra or the default?", "💰 Extra tip", "🧾 Default"),
]

# Side art (dicebear bot avatars — generated, not stored, family-friendly).
_MAJ_IMG = {
    "a": "https://api.dicebear.com/9.x/bottts-neutral/png?seed=MajA&backgroundColor=3b82f6",
    "b": "https://api.dicebear.com/9.x/bottts-neutral/png?seed=MajB&backgroundColor=ef4444",
}

_majority_games = {}
_majority_last = {}
MAJ_DURATION = 15
MAJ_COOLDOWN = 10


def _maj_bar(pct: int, width: int = 12) -> str:
    filled = max(0, min(width, round(pct / 100 * width)))
    return "▰" * filled + "▱" * (width - filled)


def _maj_embed(state) -> discord.Embed:
    a, b = state["a"], state["b"]
    va, vb = len(state["votes_a"]), len(state["votes_b"])
    total = va + vb
    pa = round(va / total * 100) if total else 50
    pb = 100 - pa if total else 50
    lead = "⚖️ Tied!" if va == vb else (f"🔵 A leads!" if va > vb else "🔴 B leads!")
    desc = (
        f"❓ **{state['question']}**\n\n"
        f"🔵 **A:** {a}\n`{_maj_bar(pa)}` **{pa}%** ({va})\n\n"
        f"🔴 **B:** {b}\n`{_maj_bar(pb)}` **{pb}%** ({vb})\n\n"
        f"{lead}\n👥 **{total} vote(s)** — vote, then the majority scores!"
    )
    embed = discord.Embed(title="📊 Majority Rules — guess the crowd!", description=desc, color=0x22C55E)
    embed.set_thumbnail(url=_MAJ_IMG["a"])
    embed.set_image(url=_MAJ_IMG["b"])
    embed.set_footer(text=f"Ends in {max(0, int(state['ends_at'] - time.time()))}s · majority voters score 🏆")
    return embed


class MajButton(discord.ui.Button):
    def __init__(self, side: str):
        self.side = side
        super().__init__(style=discord.ButtonStyle.primary if side == "a" else discord.ButtonStyle.danger,
                         label="A" if side == "a" else "B",
                         emoji="🔵" if side == "a" else "🔴")

    async def callback(self, interaction: discord.Interaction):
        cmd_tick("majority_vote")
        view: MajView = self.view
        state = view.state
        uid = interaction.user.id
        state["votes_a"].discard(uid)
        state["votes_b"].discard(uid)
        (state["votes_a"] if self.side == "a" else state["votes_b"]).add(uid)
        await view.render(interaction)


class MajView(discord.ui.View):
    def __init__(self, state, timeout: int = MAJ_DURATION):
        super().__init__(timeout=timeout)
        self.state = state
        self.message = None
        self.add_item(MajButton("a"))
        self.add_item(MajButton("b"))

    async def render(self, interaction: discord.Interaction):
        embed = _maj_embed(self.state)
        try:
            if self.message:
                await interaction.response.edit_message(embed=embed, view=self)
            else:
                await interaction.response.send_message(embed=embed, view=self)
                self.message = await interaction.original_response()
        except discord.NotFound:
            # Interaction expired (3s window) but the game is live — update in place.
            try:
                if self.message:
                    await self.message.edit(embed=embed, view=self)
            except (discord.HTTPException, discord.NotFound):
                pass
        except discord.HTTPException:
            pass
        # Multi mode: everyone has voted → end right away.
        try:
            await self._maybe_auto_end(interaction)
        except Exception:
            pass

    async def _maybe_auto_end(self, interaction=None):
        """Multi: all present members voted → finish. Single: one vote ends it."""
        if self.state.get("mode") == "single":
            total = len(self.state["votes_a"]) + len(self.state["votes_b"])
            if total >= 1:
                await self.finish(interaction, reason="vote in")
            return
        try:
            members = sum(1 for m in self.state.get("guild_members", []) if not m.get("bot"))
        except Exception:
            members = 0
        if members <= 0:
            return
        voted = len(self.state["votes_a"] | self.state["votes_b"])
        if voted >= members:
            await self.finish(interaction, reason="everyone voted")

    async def finish(self, interaction_or_none, reason="time"):
        key = self.state["key"]
        _majority_games.pop(key, None)
        for c in self.children:
            c.disabled = True
        va, vb = len(self.state["votes_a"]), len(self.state["votes_b"])
        t0 = self.state.get("started_at", time.time())
        elapsed = max(0.1, time.time() - t0)
        if va == vb:
            result = "🤝 Tie — no points!"
            color = 0xA78BFA
        else:
            winners = self.state["votes_a"] if va > vb else self.state["votes_b"]
            win_side = "A" if va > vb else "B"
            # Faster majority = bigger bonus (15s rounds reward speed).
            speed_bonus = max(0, round((MAJ_DURATION - elapsed) / MAJ_DURATION * 5))
            pts = 1 + speed_bonus
            mentions = ", ".join(f"<@{u}>" for u in list(winners)[:10])
            result = f"🏆 **Side {win_side} takes it!** {va}–{vb}\n{mentions} guessed the crowd! +{pts} 🏅 ({elapsed:.0f}s, speed bonus +{speed_bonus})"
            color = 0x22C55E
        embed = _maj_embed(self.state)
        embed.add_field(name="🏁 Final", value=result, inline=False)
        embed.set_footer(text=f"Closed ({reason}) · thanks for voting!")
        embed.color = color
        try:
            if interaction_or_none is not None and hasattr(interaction_or_none, "response"):
                try:
                    await interaction_or_none.response.edit_message(embed=embed, view=self)
                except discord.InteractionResponded:
                    await self.message.edit(embed=embed, view=self)
            elif self.message:
                await self.message.edit(embed=embed, view=self)
        except (discord.HTTPException, discord.NotFound):
            pass
        self.stop()

    async def on_timeout(self):
        await self.finish(None, reason="15s up")


@bot.tree.command(name="majority", description="Majority Rules: vote, then the crowd majority scores!")
@app_commands.describe(mode="Single (your vote ends it) or multi (waits for everyone, 15s)")
@app_commands.choices(mode=[app_commands.Choice(name="👤 Single (default)", value="single"),
                            app_commands.Choice(name="👥 Multi (everyone votes)", value="multi")])
async def majority(interaction: discord.Interaction, mode: str = "single"):
    cmd_tick("majority")
    now = time.time()
    if now - _majority_last.get(interaction.user.id, 0) < MAJ_COOLDOWN:
        await interaction.response.send_message(f"⏳ Slow down — try again in {MAJ_COOLDOWN}s.", ephemeral=True)
        return
    _majority_last[interaction.user.id] = now
    if interaction.guild is None:
        await interaction.response.send_message("Use `/majority` inside a server.", ephemeral=True)
        return
    key = (interaction.guild.id, interaction.channel.id)
    if key in _majority_games:
        await interaction.response.send_message("📊 A Majority Rules round is already running here — vote on it!", ephemeral=True)
        return
    q, a, b = random.choice(_MAJORITY_QUESTIONS)
    members = []
    try:
        if mode == "multi" and interaction.guild:
            members = [{"id": m.id, "bot": m.bot} for m in interaction.guild.members if not m.bot][:50]
    except Exception:
        members = []
    state = {"key": key, "question": q, "a": a, "b": b,
             "votes_a": set(), "votes_b": set(), "mode": mode,
             "guild_members": members, "started_at": time.time(),
             "host_id": interaction.user.id, "ends_at": time.time() + MAJ_DURATION}
    _majority_games[key] = state
    view = MajView(state)
    embed = _maj_embed(state)
    tag = "👥 Multi — everyone votes, auto-ends when all in!" if mode == "multi" else "👤 Single — your vote ends it!"
    await interaction.response.send_message(tag, embed=embed, view=view)
    try:
        view.message = await interaction.original_response()
    except (discord.HTTPException, discord.NotFound):
        view.message = None


# ---------------------------------------------------------------------------
# Word Scramble — race to unscramble, fastest wins
# ---------------------------------------------------------------------------
_scramble_games = {}

_SCRAMBLE_WORDS = ["PYTHON", "DISCORD", "ROBOT", "DRAGON", "PIZZA", "GUITAR", "CASTLE",
                   "ROCKET", "PLANET", "OCEAN", "TIGER", "PANDA", "WIZARD", "PIRATE",
                   "COMPUTER", "BOOK", "GARDEN", "MOUNTAIN", "RAINBOW", "TREASURE"]


def _scramble_word(word: str) -> str:
    letters = list(word)
    for _ in range(20):
        random.shuffle(letters)
        if "".join(letters) != word:
            break
    return " ".join(letters)


@bot.tree.command(name="scramble", description="Word Scramble race — fastest to unscramble wins!")
async def scramble(interaction: discord.Interaction):
    cmd_tick("scramble")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/scramble` inside a server.", ephemeral=True)
        return
    key = (interaction.guild.id, interaction.channel.id)
    if key in _scramble_games:
        await interaction.response.send_message("🔀 A scramble is already running here — solve it with `/unscramble`!", ephemeral=True)
        return
    word = random.choice(_SCRAMBLE_WORDS)
    _scramble_games[key] = {"word": word, "at": time.time(), "host_id": interaction.user.id}
    embed = discord.Embed(title="🔀 Word Scramble — race!",
                          description=f"Unscramble this:\n\n# `{_scramble_word(word)}`\n\nFirst to `/unscramble <word>` wins! 🏁",
                          color=0xF59E0B)
    embed.set_footer(text="60s on the clock · anyone can answer")
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="unscramble", description="Answer the running word scramble.")
@app_commands.describe(word="Your unscrambled guess")
async def unscramble(interaction: discord.Interaction, word: str):
    cmd_tick("unscramble")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/unscramble` inside a server.", ephemeral=True)
        return
    key = (interaction.guild.id, interaction.channel.id)
    game = _scramble_games.get(key)
    if not game:
        await interaction.response.send_message("No scramble running here. Start one with `/scramble`!", ephemeral=True)
        return
    if time.time() - game["at"] > 60:
        _scramble_games.pop(key, None)
        await interaction.response.send_message(f"⏰ Time! The word was `{game['word']}`.", ephemeral=True)
        return
    if word.strip().upper() == game["word"]:
        dt = round(time.time() - game["at"], 1)
        _scramble_games.pop(key, None)
        embed = discord.Embed(title="🏆 Correct!",
                              description=f"{interaction.user.mention} unscrambled `{game['word']}` in **{dt}s**! ⚡",
                              color=0x22C55E)
        await interaction.response.send_message(embed=embed)
    else:
        await interaction.response.send_message(f"`{word.strip().upper()}` — nope, keep trying! 🔀", ephemeral=True)


# ---------------------------------------------------------------------------
# AI
# ---------------------------------------------------------------------------

AI_GROUP = app_commands.Group(name="ai", description="AI settings (admins only)")


_model_cache = {"at": 0.0, "endpoint": "", "models": []}


async def ai_model_autocomplete(interaction: discord.Interaction, current: str):
    if interaction.guild is None:
        return [app_commands.Choice(name=f"default ({OLLAMA_MODEL})", value="default")]
    try:
        endpoint = guild_ai_config(interaction.guild.id)["endpoint"]
    except Exception:
        return [app_commands.Choice(name=f"default ({OLLAMA_MODEL})", value="default")]
    now = time.time()
    if endpoint == _model_cache["endpoint"] and now - _model_cache["at"] < 60:
        models = _model_cache["models"]
    else:
        try:
            models = await asyncio.wait_for(asyncio.to_thread(list_ollama_models, endpoint), timeout=2.5)
        except Exception:
            models = _model_cache.get("models", [])
            if not models:
                return [app_commands.Choice(name=f"default ({OLLAMA_MODEL})", value="default")]
        _model_cache.update({"at": now, "endpoint": endpoint, "models": models})
    models = [m for m in models if current.lower() in m.lower()][:20]
    if not models:
        return [app_commands.Choice(name=f"default ({OLLAMA_MODEL})", value="default")]
    return [app_commands.Choice(name=m, value=m) for m in models]


@AI_GROUP.command(name="model", description="Choose an AI model (default = host's model).")
@app_commands.describe(model="Pick from models on your configured AI host, or 'default'")
@app_commands.autocomplete(model=ai_model_autocomplete)
async def ai_model_selector(interaction: discord.Interaction, model: str):
    cmd_tick("ai_model_selector")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/ai model` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    if model.lower() in ("default", "reset", "off"):
        set_cfg(interaction.guild.id, "ai_model", "")
        await interaction.response.send_message("AI model reset to the host default.")
        return
    set_cfg(interaction.guild.id, "ai_model", model)
    await interaction.response.send_message(f"AI model → **{model}** on this server.")


@AI_GROUP.command(name="toggle", description="Enable or disable AI chat on this server.")
@app_commands.describe(enabled="true to enable, false to disable")
async def ai_toggle(interaction: discord.Interaction, enabled: bool):
    cmd_tick("ai_toggle")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/ai toggle` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    set_cfg(interaction.guild.id, "ai_enabled", "1" if enabled else "0")
    await interaction.response.send_message(f"AI chat {'enabled' if enabled else 'disabled'}.")


def preset_choice_items(guild_id, kind):
    """Autocomplete choices: 'none' + built-in + custom preset names."""
    found = {"none"}
    for name in PERSONALITIES if kind == "personality" else CHARACTERS:
        found.add(name)
    for name in guild_presets(guild_id, kind):
        found.add(name)
    return [app_commands.Choice(name=n, value=n) for n in sorted(found)]


async def _personality_ac(interaction: discord.Interaction, current: str):
    """Autocomplete callback (must be a coroutine function)."""
    return preset_choice_items(interaction.guild_id or 0, "personality")


async def _character_ac(interaction: discord.Interaction, current: str):
    """Autocomplete callback (must be a coroutine function)."""
    return preset_choice_items(interaction.guild_id or 0, "character")


@AI_GROUP.command(name="personality", description="Set the bot's personality tone (or 'none').")
@app_commands.describe(name="Personality name, or 'none'")
@app_commands.autocomplete(name=_personality_ac)
async def ai_personality(interaction: discord.Interaction, name: str):
    cmd_tick("ai_personality")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/ai personality` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    key = name.strip()
    if key == "none":
        set_cfg(interaction.guild.id, "ai_personality", "none")
        ai_queue.drop(interaction.guild.id)
        await interaction.response.send_message("Personality → **none** (default buddy). Applies to your next reply! ⚡")
        return
    if key not in PERSONALITIES and key not in guild_presets(interaction.guild.id, "personality"):
        await interaction.response.send_message(f"Unknown personality `{key}`.", ephemeral=True)
        return
    set_cfg(interaction.guild.id, "ai_personality", key)
    ai_queue.drop(interaction.guild.id)
    await interaction.response.send_message(f"Personality → **{key}**. Applies to your next reply! ⚡")


@AI_GROUP.command(name="character", description="Set how the bot pretends to be (or 'none').")
@app_commands.describe(name="Character name, or 'none'")
@app_commands.autocomplete(name=_character_ac)
async def ai_character(interaction: discord.Interaction, name: str):
    cmd_tick("ai_character")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/ai character` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    key = name.strip()
    if key == "none":
        set_cfg(interaction.guild.id, "ai_character", "")
        ai_queue.drop(interaction.guild.id)
        await interaction.response.send_message("Character → **none**. Applies to your next reply! ⚡")
        return
    if key not in CHARACTERS and key not in guild_presets(interaction.guild.id, "character"):
        await interaction.response.send_message(f"Unknown character `{key}`.", ephemeral=True)
        return
    set_cfg(interaction.guild.id, "ai_character", key)
    ai_queue.drop(interaction.guild.id)
    await interaction.response.send_message(f"Character → **{key}**. Applies to your next reply! ⚡")


@AI_GROUP.command(name="status", description="Show this server's AI settings.")
async def ai_status(interaction: discord.Interaction):
    cmd_tick("ai_status")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/ai status` inside a server.", ephemeral=True)
        return
    cfg = guild_ai_config(interaction.guild.id)
    allowed_note = "everywhere" if not (cfg["ai_channels"] or "").strip() else "picked channels only"
    source_note = "shared Quaestio box" if cfg["source"] == "shared" else (f"your own box{f' (in community pool)' if cfg['contribute'] else ''}")
    personality = get_cfg(interaction.guild.id, "ai_personality", "none")
    character = get_cfg(interaction.guild.id, "ai_character", "")
    admin = is_admin(interaction.user)
    contributor = bool(cfg.get("contributor_perks"))
    embed = discord.Embed(
        title="🌟 Quaestio AI — Pool Contributor" if contributor else "Quaestio AI settings",
        color=0xF5C518 if contributor else 0xA78BFA,
    )
    embed.add_field(name="Enabled", value="✅" if cfg["enabled"] else "❌", inline=True)
    embed.add_field(name="Source", value=source_note, inline=True)
    embed.add_field(name="Model", value=f"`{cfg['model']}`", inline=True)
    if admin:
        embed.add_field(name="Endpoint", value=f"`{cfg['endpoint']}`", inline=False)
    embed.add_field(name="Personality",
                    value=f"`{personality or 'none'}` · {cfg['memory']} turns/channel",
                    inline=False)
    embed.add_field(name="Quota",
                    value=f"{cfg['quota']} calls/{cfg['window']}h {'(unlimited)' if not cfg['quota'] else ''}",
                    inline=True)
    embed.add_field(name="Creativity",
                    value=f"{cfg['temperature']} · {cfg['max_tokens']} tokens", inline=True)
    embed.add_field(
        name="Chat",
        value=(f"Replies on mention: {'✅' if cfg['ai_mention'] else '❌'}\n"
               f"Conversation mode: {'✅ stays ' + str(cfg['conv_minutes']) + ' min' if cfg['conv'] else '❌'}\n"
               f"Allowed channels: {allowed_note}"),
        inline=False,
    )
    if cfg.get("ai_character"):
        embed.add_field(name="Character", value=f"`{cfg['ai_character']}`", inline=False)
    elif character:
        embed.add_field(name="Character", value=f"`{character}`", inline=False)
    if contributor:
        mult = contributor_mult(interaction.guild.id)
        embed.set_footer(text=f"🌟 Pool contributor badge — priority routing + {mult}x request limits")
    else:
        embed.set_footer(text="Lend compute with quaestio pool-serve to earn a contributor badge")
    await interaction.response.send_message(embed=embed, ephemeral=True)


@AI_GROUP.command(name="clear", description="Forget this channel's conversation memory.")
async def ai_clear(interaction: discord.Interaction):
    cmd_tick("ai_clear")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/ai clear` inside a server channel.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    memory.clear(interaction.guild.id, interaction.channel.id)
    await interaction.response.send_message("🧹 Memory cleared for this channel.", ephemeral=True)


@bot.tree.command(name="pool", description="See the community pool + what contributors earn.")
async def pool_info(interaction: discord.Interaction):
    cmd_tick("pool")
    """Anonymous pool stats and the contributor perk pitch (no identities)."""
    await interaction.response.defer(thinking=True, ephemeral=True)
    try:
        nodes = pool_candidates("", limit=8)
        total = pool_total_share()
    except Exception:
        nodes, total = [], 0
    total_nodes = 0
    try:
        total_nodes = len(pool_hosters())
    except Exception:
        total_nodes = len(nodes)
    try:
        n_active = len(pool_active_nodes(limit=25))
    except Exception:
        n_active = 0
    lines = ["**⚡ Community pool**",
             f"Registered nodes: **{total_nodes}** (enabled, incl. offline/cooling) · shared capacity: **{total}%**",
             f"Active now: **{n_active}** responding/listening"]
    try:
        active = pool_active_nodes(limit=8)
        if active:
            names = ", ".join(f"`{a['name']}`" for a in active[:8])
            n_listen = sum(1 for a in active if a["pull"])
            lines.append(f"🟢 Active now ({len(active)} listening/responding): {names}")
            if n_listen:
                lines.append(f"📡 {n_listen} pull worker(s) listening for jobs")
        else:
            lines.append("⚪ No pool nodes active right now — serving from the host box")
    except Exception:
        pass
    try:
        leaders = pool_leaders(limit=3)
        medals = ["🥇", "🥈", "🥉"]
        for i, l in enumerate(leaders):
            lines.append(f"{medals[i]} `{l['name']}` — {l['served']} served")
    except Exception:
        pass
    if interaction.guild is not None:
        try:
            cfg = guild_ai_config(interaction.guild.id)
            if cfg.get("contributor_perks"):
                lines.append(f"🌟 This server contributes ✓ (priority routing, "
                             f"{contributor_mult(interaction.guild.id)}x request limits)")
            else:
                lines.append(f"Lend your box (`quaestio pool-serve`) and earn "
                             f"priority routing, higher request limits "
                             f"and a contributor badge. Anonymous — random node ID only.")
        except Exception:
            pass
    await interaction.followup.send("\n".join(lines), ephemeral=True)


def _remember_reply(interaction, answer, prompt):
    cfg = guild_ai_config(interaction.guild.id)
    question = (prompt or "")[:400]
    memory.push(interaction.guild.id, interaction.channel.id, "user", question, cfg["memory"],
                user_id=interaction.user.id, name=interaction.user.display_name)
    memory.push(interaction.guild.id, interaction.channel.id, "bot", answer[:400], cfg["memory"],
                user_id=interaction.guild.me.id, name=interaction.guild.me.display_name)


@bot.tree.command(name="ask", description="Chat with Quaestio's local AI.")
@app_commands.describe(prompt="What you want to say or ask")
async def ask(interaction: discord.Interaction, prompt: str):
    cmd_tick("ask")
    if interaction.guild is None:
        await interaction.response.defer(thinking=False)
        try:
            await dm_chat(interaction.channel, prompt[:400], host_cfg(),
                          mention=interaction.user.mention,
                          user_id=interaction.user.id, name=interaction.user.display_name)
            try:
                await interaction.followup.send("Answered above 👆", ephemeral=True)
            except discord.HTTPException:
                pass
        except Exception:  # noqa: BLE001 — never leave an interaction stuck
            try:
                await interaction.followup.send("```⚠️ Something went wrong. Try again in a moment.```", ephemeral=True)
            except discord.HTTPException:
                pass
        return
    cfg = guild_ai_config(interaction.guild.id)
    # /ask is an explicit, one-shot question: it never wakes conversation mode
    # (only @Quaestio does).
    _conv_until.pop((interaction.guild.id, interaction.channel.id), None)
    if not cfg["enabled"]:
        await interaction.response.send_message("AI chat is disabled here.", ephemeral=True)
        return
    if not quota_ok(interaction.guild.id, cfg["quota"], cfg["window"]):
        await interaction.response.send_message(
            "⚠️ This server has hit its AI quota for this hour (set in the dashboard).",
            ephemeral=True,
        )
        return
    if not flood_ok(interaction.guild.id, interaction.user.id, mult=contributor_mult(interaction.guild.id)):
        await interaction.response.send_message(
            "⏳ Slow down — too many AI requests at once. Try again in a minute.",
            ephemeral=True,
        )
        return
    if not channel_allowed(interaction.guild.id, interaction.channel.id, cfg):
        await interaction.response.send_message(
            "AI chat is switched off in this channel — it's only allowed in the channels picked in the dashboard.",
            ephemeral=True,
        )
        return
    await interaction.response.defer(thinking=False)
    # NOTE: no ephemeral "thinking…" followup — defer already shows thinking
    # in the client and the channel typing indicator covers the wait. The old
    # double indicator (message + typing) was just noise.

    try:
        context = memory.context(interaction.guild.id, interaction.channel.id, cfg["memory"])
        profiles = profile_lines(interaction.guild.id, [m.get("user_id") for m in context] + [interaction.user.id])
        persona_system, full_prompt = build_prompt(cfg["persona"], context, prompt, cfg["instructions"], profiles, asker_name=interaction.user.display_name)
    except Exception:
        persona_system, full_prompt = "", prompt[:400]

    async def factory():
        return await ask_ollama_any(cfg, full_prompt, asker=interaction.user.display_name, system=persona_system)

    fut = ai_queue.submit(interaction.guild.id, factory)
    try:
        async with interaction.channel.typing():
            answer = await _wait_ai_answer(fut, OLLAMA_TIMEOUT + 30)
            await asyncio.sleep(0)
    except BusyError as exc:
        await interaction.followup.send(str(exc), ephemeral=True)
        return
    except asyncio.TimeoutError:
        _log_ai_failure("timeout", interaction.guild.id, cfg["model"], TimeoutError("waiter budget spent"))
        await interaction.followup.send("```The AI took too long. Try again in a moment.```", ephemeral=True)
        return
    except ConnectionError as exc:
        _log_ai_failure("connection", interaction.guild.id, cfg["model"], exc)
        await interaction.followup.send(
            f"```{NO_COMPUTE_NOTICE}```" if _is_no_compute(exc) else f"```⚠️ {exc}```",
            ephemeral=True)
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — never leave an interaction stuck
        try:
            await interaction.followup.send(f"```⚠️ Something went wrong: {exc}```", ephemeral=True)
        except discord.HTTPException:
            pass
        return

    try:
        quota_tick(interaction.guild.id, cfg["window"])
        _remember_reply(interaction, answer, prompt)
        await human_type(interaction.channel, answer, mention=interaction.user.mention)
    except Exception:  # noqa: BLE001 — DB or send hiccup must not strand the user
        try:
            await interaction.followup.send(f"{interaction.user.mention} {answer[:1900]}", ephemeral=True)
        except discord.HTTPException:
            pass


PANEL_URL = os.environ.get("PANEL_URL", "https://admin.quaestio.online")


@bot.tree.command(name="panel", description="Open this server's web settings panel.")
async def panel(interaction: discord.Interaction):
    cmd_tick("panel")
    await interaction.response.send_message(
        "🛠️ **Quaestio web panel**\n"
        f"⚙️ Settings: {PANEL_URL}\n"
        "Sign in with Discord **as an admin of this server** to change AI settings, "
        "custom instructions, welcomes, and more.\n"
        "Settings you see in Discord stay here too — the panel is just easier.",
        ephemeral=True,
    )


@bot.tree.command(name="site", description="Quaestio on the web: docs, pool, and source.")
async def site_cmd(interaction: discord.Interaction):
    cmd_tick("site")
    await interaction.response.send_message(
        "🌐 **Quaestio on the web**\n"
        "📖 Main site: https://quaestio.online\n"
        "⚙️ Admin panel: https://admin.quaestio.online\n"
        "⚡ Pool: https://pool.quaestio.online\n"
        "💻 Source: https://github.com/fishesarethings/discord-ai-companion-bot-pool",
        ephemeral=True,
    )


@bot.tree.command(name="contribute", description="How to lend compute to the community pool.")
async def contribute_cmd(interaction: discord.Interaction):
    cmd_tick("contribute")
    await interaction.response.send_message(
        "⚡ **Lend spare AI compute**\n"
        "Run `quaestio pool-serve` (or host in your browser at "
        "https://pool.quaestio.online) — no port forwards, anonymous node ID.\n"
        "Earn priority routing, higher limits, and a 🌟 badge. Details on the pool page.",
        ephemeral=True,
    )


@bot.tree.command(name="summarize", description="Summarize the last N messages in this channel.")
@app_commands.describe(limit="How many messages to summarize (default 20, max 60)")
async def summarize(interaction: discord.Interaction, limit: int = 20):
    cmd_tick("summarize")
    if interaction.guild is None:
        await interaction.response.send_message("Summarize works in a server's channel, not DMs.", ephemeral=True)
        return
    cfg = guild_ai_config(interaction.guild.id)
    if not cfg["enabled"]:
        await interaction.response.send_message("AI chat is disabled here.", ephemeral=True)
        return
    if not channel_allowed(interaction.guild.id, interaction.channel.id, cfg):
        await interaction.response.send_message(
            "AI chat is switched off in this channel — it's only allowed in the channels picked in the dashboard.",
            ephemeral=True,
        )
        return
    if not flood_ok(interaction.guild.id, interaction.user.id, mult=contributor_mult(interaction.guild.id)):
        await interaction.response.send_message(
            "⏳ Slow down — too many AI requests at once. Try again in a minute.",
            ephemeral=True,
        )
        return
    limit = max(1, min(limit, 60))
    if not quota_ok(interaction.guild.id, cfg["quota"], cfg["window"]):
        await interaction.response.send_message(
            "⚠️ This server has hit its AI quota for this hour (set in the dashboard).",
            ephemeral=True,
        )
        return
    await interaction.response.defer(thinking=False)
    texts = []
    async for msg in interaction.channel.history(limit=limit):
        if msg.author.bot:
            continue
        texts.append(f"{msg.author.display_name}: {msg.content[:200]}")
        if len(texts) >= 60:
            break
    if not texts:
        await interaction.followup.send("Nothing to summarize.")
        return
    prompt = (
        "Summarize these Discord messages in a few short bullets. Keep it neutral "
        "and concise.\n\n" + "\n".join(reversed(texts))
    )
    if cfg["instructions"]:
        prompt += "\n\nFollow these server instructions where relevant:\n" + cfg["instructions"]

    async def factory():
        return await ask_ollama_any(cfg, prompt, asker="summary",
                                    system="Summarize chat messages in a few short neutral bullets.")

    fut = ai_queue.submit(interaction.guild.id, factory)
    try:
        async with interaction.channel.typing():
            answer = await _wait_ai_answer(fut, OLLAMA_TIMEOUT + 30)
    except BusyError as exc:
        await interaction.followup.send(str(exc))
        return
    except asyncio.TimeoutError:
        _log_ai_failure("timeout", interaction.guild.id, cfg["model"], TimeoutError("waiter budget spent"))
        await interaction.followup.send("The AI took too long. Try again in a moment.")
        return
    except ConnectionError as exc:
        _log_ai_failure("connection", interaction.guild.id, cfg["model"], exc)
        await interaction.followup.send(
            NO_COMPUTE_NOTICE if _is_no_compute(exc) else f"⚠️ {exc}")
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — never leave an interaction stuck
        try:
            await interaction.followup.send(f"```⚠️ Something went wrong: {exc}```")
        except discord.HTTPException:
            pass
        return

    await interaction.followup.send(
        f"📄 **Summary of last {len(texts)} messages**\n{answer[:1900]}"
    )


bot.tree.add_command(AI_GROUP)


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------

@bot.tree.command(name="rank", description="Check your XP and level.")
@app_commands.describe(member="Member to check (defaults to you)")
async def rank(interaction: discord.Interaction, member: discord.Member = None):
    cmd_tick("rank")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/rank` inside a server.", ephemeral=True)
        return
    member = member or interaction.user
    conn = db()
    row = conn.execute(
        "SELECT messages FROM xp WHERE guild_id=? AND user_id=?",
        (str(interaction.guild.id), str(member.id)),
    ).fetchone()
    conn.close()
    messages = row["messages"] if row else 0
    level = level_for_messages(messages)
    nxt = xp_for_level(level + 1)
    pct = min(1.0, messages / nxt) if nxt else 1.0
    filled = round(pct * 10)
    bar = "▰" * filled + "▱" * (10 - filled)
    await interaction.response.send_message(
        f"🎉 {member.mention} — **Level {level}**\n{bar} {messages}/{nxt} XP",
    )


@bot.tree.command(name="profile", description="What Quaestio has learned about a member.")
@app_commands.describe(member="Member to check (defaults to you)")
async def profile(interaction: discord.Interaction, member: discord.Member = None):
    cmd_tick("profile")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/profile` inside a server.", ephemeral=True)
        return
    member = member or interaction.user
    facts = profile_facts(interaction.guild.id, member.id)
    if not facts:
        await interaction.response.send_message(
            f"{member.display_name} — I haven't learned anything about them yet. "
            "Say things like *'I like fish'* and I'll start remembering.",
            ephemeral=True,
        )
        return
    await interaction.response.send_message(
        f"🧠 **What I know about {member.display_name}**\n{facts}",
        ephemeral=True,
    )


@bot.tree.command(name="leaderboard", description="Top chatters by XP in this server.")
@app_commands.describe(top="How many to show (default 10, max 25)")
async def leaderboard(interaction: discord.Interaction, top: int = 10):
    cmd_tick("leaderboard")
    if interaction.guild is None:
        await interaction.response.send_message("Leaderboard works inside a server.", ephemeral=True)
        return
    top = max(3, min(top, 25))
    conn = db()
    rows = conn.execute(
        "SELECT user_id, messages FROM xp WHERE guild_id=? ORDER BY messages DESC LIMIT ?",
        (str(interaction.guild.id), top),
    ).fetchall()
    conn.close()
    if not rows:
        await interaction.response.send_message("No XP yet — get people chatting first!", ephemeral=True)
        return
    medals = ["🥇", "🥈", "🥉"]
    max_msgs = max(r["messages"] for r in rows)
    bar_w = 10
    lines = ["**🏆 Top chatters**"]
    for i, r in enumerate(rows):
        member = interaction.guild.get_member(int(r["user_id"]))
        name = member.display_name if member else f"<@{r['user_id']}>"
        filled = round((r["messages"] / max_msgs) * bar_w) if max_msgs else 0
        bar = "▰" * filled + "▱" * (bar_w - filled)
        rank = medals[i] if i < 3 else f"**{i + 1}.**"
        tier = "· 🔥" if i == 0 else ""
        lines.append(f"{rank} **{name}** — Lv {level_for_messages(r['messages'])} · {bar} {r['messages']} msgs{tier}")
    await interaction.response.send_message("\n".join(lines))


# ---------------------------------------------------------------------------
# Moderation
# ---------------------------------------------------------------------------

def _warn_db(guild_id, user_id, reason):
    conn = db()
    conn.execute(
        "INSERT INTO warned (guild_id, user_id, reason, at) VALUES (?, ?, ?, ?)",
        (str(guild_id), str(user_id), reason, datetime.datetime.now().isoformat()),
    )
    conn.commit()
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM warned WHERE guild_id=? AND user_id=?",
        (str(guild_id), str(user_id)),
    ).fetchone()["n"]
    conn.close()
    return count


@bot.tree.command(name="warn", description="Warn a member.")
@app_commands.describe(member="Member to warn", reason="Reason")
async def warn(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason"):
    cmd_tick("warn")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/warn` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    count = _warn_db(interaction.guild.id, member.id, reason)
    limit = _safe_int(get_cfg(interaction.guild.id, "warnlimit"), WARN_LIMIT_DEFAULT)
    msg = f"⚠️ {member.mention} warned — **{count}/{limit}**\n> {reason}"
    await interaction.response.send_message(msg)
    if count >= limit:
        try:
            await member.kick(reason=f"Reached warn limit ({count}).")
        except discord.Forbidden:
            await interaction.followup.send("Auto-kick failed — I lack permission.", ephemeral=True)
            return
        except discord.HTTPException:
            await interaction.followup.send("Auto-kick failed — check my role position.", ephemeral=True)
            return
        await interaction.followup.send(f"{member.display_name} auto-kicked (warn limit).")


@bot.tree.command(name="warns", description="List a member's warnings.")
@app_commands.describe(member="Member to check")
async def warns(interaction: discord.Interaction, member: discord.Member):
    cmd_tick("warns")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/warns` inside a server.", ephemeral=True)
        return
    conn = db()
    rows = conn.execute(
        "SELECT reason, at FROM warned WHERE guild_id=? AND user_id=?",
        (str(interaction.guild.id), str(member.id)),
    ).fetchall()
    conn.close()
    if not rows:
        await interaction.response.send_message(f"{member.mention} has no warnings. ✅")
        return
    lines = [f"**{member.display_name}** — {len(rows)} warning(s):"]
    for r in rows:
        lines.append(f"- {r['at'][:10]}: {r['reason']}")
    await interaction.response.send_message("\n".join(lines))


@bot.tree.command(name="delwarns", description="Clear all warnings for a member.")
@app_commands.describe(member="Member to clear")
async def delwarns(interaction: discord.Interaction, member: discord.Member):
    cmd_tick("delwarns")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/delwarns` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    conn = db()
    conn.execute(
        "DELETE FROM warned WHERE guild_id=? AND user_id=?",
        (str(interaction.guild.id), str(member.id)),
    )
    conn.commit()
    conn.close()
    await interaction.response.send_message(f"Cleared warnings for {member.mention}.")


@bot.tree.command(name="kick", description="Kick a member.")
@app_commands.describe(member="Member to kick", reason="Reason")
async def kick(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason"):
    cmd_tick("kick")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/kick` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    try:
        await member.kick(reason=reason)
    except discord.Forbidden:
        await interaction.response.send_message("I can't kick that member.", ephemeral=True)
        return
    except discord.HTTPException:
        await interaction.response.send_message("Kick failed — check my role position.", ephemeral=True)
        return
    await interaction.response.send_message(f"👢 Kicked {member.display_name} — {reason}")


@bot.tree.command(name="ban", description="Ban a member.")
@app_commands.describe(member="Member to ban", reason="Reason")
async def ban(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason"):
    cmd_tick("ban")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/ban` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    try:
        await member.ban(reason=reason)
    except discord.Forbidden:
        await interaction.response.send_message("I can't ban that member.", ephemeral=True)
        return
    except discord.HTTPException:
        await interaction.response.send_message("Ban failed — check my role position.", ephemeral=True)
        return
    await interaction.response.send_message(f"🔨 Banned {member.display_name} — {reason}")


@bot.tree.command(name="unban", description="Unban a user by name.")
@app_commands.describe(user="Name of the banned user")
async def unban(interaction: discord.Interaction, user: str):
    cmd_tick("unban")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/unban` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    await interaction.response.defer(thinking=True, ephemeral=True)
    try:
        banned = [entry async for entry in interaction.guild.bans()]
    except (discord.Forbidden, discord.HTTPException):
        await interaction.followup.send("I can't see the ban list.", ephemeral=True)
        return
    q = user.strip().lower()
    target = next((e for e in banned
                   if q == str(e.user).lower() or q == str(e.user.id)
                   or q == getattr(e.user, "name", "").lower()), None)
    if target is None:
        await interaction.followup.send(f"No banned user matching `{user}`.", ephemeral=True)
        return
    try:
        await interaction.guild.unban(target.user, reason="Quaestio unban")
    except (discord.Forbidden, discord.HTTPException, discord.NotFound):
        await interaction.followup.send("Unban failed.", ephemeral=True)
        return
    await interaction.followup.send(f"🔓 Unbanned {target.user}.", ephemeral=True)


@bot.tree.command(name="purge", description="Bulk-delete recent messages.")
@app_commands.describe(count="How many to delete (max 100)")
async def purge(interaction: discord.Interaction, count: int = 20):
    cmd_tick("purge")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/purge` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    count = max(1, min(count, 100))
    await interaction.response.defer(thinking=True, ephemeral=True)
    try:
        deleted = await interaction.channel.purge(limit=count)
    except (discord.Forbidden, discord.HTTPException):
        await interaction.followup.send("I can't delete messages here.", ephemeral=True)
        return
    await interaction.followup.send(
        f"🧹 Purged {len(deleted)} messages.", ephemeral=True
    )


@bot.tree.command(name="mute", description="Timeout a member.")
@app_commands.describe(member="Member to mute", minutes="How many minutes", reason="Reason")
async def mute(interaction: discord.Interaction, member: discord.Member, minutes: int = 10, reason: str = "No reason"):
    cmd_tick("mute")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/mute` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    duration = datetime.timedelta(minutes=max(1, min(minutes, 10080)))
    until = discord.utils.utcnow() + duration
    try:
        await member.timeout(until, reason=reason)
    except discord.Forbidden:
        await interaction.response.send_message("I can't mute that member.", ephemeral=True)
        return
    except discord.HTTPException:
        await interaction.response.send_message("Mute failed — check my role position.", ephemeral=True)
        return
    await interaction.response.send_message(
        f"🔇 Timed out {member.display_name} for {max(1, int(duration.total_seconds() // 60))}m ({reason})"
    )


@bot.tree.command(name="unmute", description="Remove a timeout.")
@app_commands.describe(member="Member to unmute")
async def unmute(interaction: discord.Interaction, member: discord.Member):
    cmd_tick("unmute")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/unmute` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    try:
        await member.timeout(None)
    except (discord.Forbidden, discord.HTTPException, discord.NotFound):
        await interaction.response.send_message("Unmute failed.", ephemeral=True)
        return
    await interaction.response.send_message(f"🔊 Unmuted {member.display_name}.")


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

@bot.tree.command(name="tag", description="Show a saved tag.")
@app_commands.describe(name="Tag name")
async def tag(interaction: discord.Interaction, name: str):
    cmd_tick("tag")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/tag` inside a server.", ephemeral=True)
        return
    conn = db()
    row = conn.execute(
        "SELECT content FROM tags WHERE guild_id=? AND name=lower(?)",
        (str(interaction.guild.id), name),
    ).fetchone()
    conn.close()
    if not row:
        await interaction.response.send_message(f"Tag `{name}` not found.", ephemeral=True)
        return
    await interaction.response.send_message(row["content"])


@bot.tree.command(name="tagcreate", description="Create a tag.")
@app_commands.describe(name="Tag name", content="Tag content")
async def tagcreate(interaction: discord.Interaction, name: str, content: str):
    cmd_tick("tagcreate")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/tagcreate` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    conn = db()
    try:
        conn.execute(
            """INSERT INTO tags (guild_id, name, content, author, at) VALUES (?, lower(?), ?, ?, ?)""",
            (str(interaction.guild.id), name, content, str(interaction.user.id),
             datetime.datetime.now().isoformat()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        await interaction.response.send_message(f"Tag `{name}` already exists.", ephemeral=True)
        return
    conn.close()
    await interaction.response.send_message(f"📌 Tag `{name}` created.")


@bot.tree.command(name="tagdelete", description="Delete a tag.")
@app_commands.describe(name="Tag name")
async def tagdelete(interaction: discord.Interaction, name: str):
    cmd_tick("tagdelete")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/tagdelete` inside a server.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message("Needs Administrator.", ephemeral=True)
        return
    conn = db()
    cur = conn.execute(
        "DELETE FROM tags WHERE guild_id=? AND name=lower(?)",
        (str(interaction.guild.id), name),
    )
    conn.commit()
    conn.close()
    if cur.rowcount:
        await interaction.response.send_message(f"🗑️ Tag `{name}` deleted.")
    else:
        await interaction.response.send_message(f"Tag `{name}` not found.", ephemeral=True)


@bot.tree.command(name="tags", description="List all tags in this server.")
async def tags(interaction: discord.Interaction):
    cmd_tick("tags")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/tags` inside a server.", ephemeral=True)
        return
    conn = db()
    rows = conn.execute(
        "SELECT name FROM tags WHERE guild_id=? ORDER BY name", (str(interaction.guild.id),)
    ).fetchall()
    conn.close()
    if not rows:
        await interaction.response.send_message("No tags yet. Create one with /tagcreate.")
    else:
        names = ", ".join(r["name"] for r in rows)
        await interaction.response.send_message(f"📚 **Tags:** {names}")


# ---------------------------------------------------------------------------
# Birthdays
# ---------------------------------------------------------------------------

BDAY_GROUP = app_commands.Group(name="birthday", description="Birthday reminders")

bot.tree.add_command(BDAY_GROUP)


@BDAY_GROUP.command(name="set", description="Save your birthday (month/day).")
@app_commands.describe(month="Birth month (1-12)", day="Birth day (1-31)")
async def bday_set(interaction: discord.Interaction, month: int, day: int):
    cmd_tick("birthday")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/birthday set` inside a server.", ephemeral=True)
        return
    if not (1 <= month <= 12 and 1 <= day <= 31):
        await interaction.response.send_message("Pick a valid month (1-12) and day (1-31).", ephemeral=True)
        return
    conn = db()
    conn.execute(
        """INSERT INTO birthdays (guild_id, user_id, month, day) VALUES (?, ?, ?, ?)
           ON CONFLICT(guild_id, user_id)
           DO UPDATE SET month=excluded.month, day=excluded.day""",
        (str(interaction.guild.id), str(interaction.user.id), f"{month:02d}", f"{day:02d}"),
    )
    conn.commit()
    conn.close()
    await interaction.response.send_message(
        f"🎂 Birthday saved as **{month}/{day}**. I'll wish you a note in the server's birthday channel.",
        ephemeral=True,
    )


@BDAY_GROUP.command(name="remove", description="Remove your birthday.")
async def bday_remove(interaction: discord.Interaction):
    cmd_tick("bday_remove")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/birthday remove` inside a server.", ephemeral=True)
        return
    conn = db()
    conn.execute(
        "DELETE FROM birthdays WHERE guild_id=? AND user_id=?",
        (str(interaction.guild.id), str(interaction.user.id)),
    )
    conn.commit()
    conn.close()
    await interaction.response.send_message("Birthday removed.", ephemeral=True)


@BDAY_GROUP.command(name="list", description="Everyone's saved birthdays.")
async def bday_list(interaction: discord.Interaction):
    cmd_tick("birthday")
    if interaction.guild is None:
        await interaction.response.send_message("Use `/birthday list` inside a server.", ephemeral=True)
        return
    conn = db()
    rows = conn.execute(
        "SELECT user_id, month, day FROM birthdays WHERE guild_id=? ORDER BY month, day",
        (str(interaction.guild.id),),
    ).fetchall()
    conn.close()
    if not rows:
        await interaction.response.send_message("No birthdays saved yet. Members can save theirs with /birthday set.", ephemeral=True)
        return
    lines = ["**🎂 Birthdays**"]
    for r in rows:
        lines.append(f"**{r['month']}/{r['day']}** — <@{r['user_id']}>")
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


async def announce_birthdays(month_day: str):
    """Post in every guild's birthday channel for the people with a birthday today."""
    month, day = month_day.split("-")
    # Match both zero-padded ("09") and legacy bare ("9") stored values.
    conn = db()
    rows = conn.execute("SELECT guild_id, user_id, month, day FROM birthdays").fetchall()
    conn.close()
    for r in rows:
        try:
            if int(r["month"]) != int(month) or int(r["day"]) != int(day):
                continue
        except (ValueError, TypeError):
            continue
        try:
            gid = r["guild_id"]
            if not flag_on(gid, "birthday_enabled", "0"):
                continue
            channel_id = get_cfg(gid, "birthday_channel", "")
            if not channel_id or not str(channel_id).strip().isdigit():
                continue
            if not str(gid).strip().isdigit():
                continue
            guild = bot.get_guild(int(gid))
            channel = guild.get_channel(int(channel_id)) if guild else None
            if not channel:
                continue
            member = guild.get_member(int(r["user_id"])) if guild else None
            try:
                await channel.send(
                    f"🎂🎉 Happy birthday <@{r['user_id']}>!"
                    + (f" Have an amazing day, **{member.display_name}**! 🎈" if member else "")
                )
            except (discord.Forbidden, discord.HTTPException):
                pass
        except Exception:
            continue


async def birthday_loop():
    await bot.wait_until_ready()
    last = None
    while not bot.is_closed():
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%m-%d")
        if today != last:
            last = today
            try:
                await announce_birthdays(today)
            except Exception:
                pass
        await asyncio.sleep(60)


# ---------------------------------------------------------------------------
# Optional localhost settings page (LOCAL_WEB=1 + LOCAL_WEB_PORT, off by default)
# ---------------------------------------------------------------------------

def _local_web() -> bool:
    return os.environ.get("LOCAL_WEB", "0").strip() in ("1", "true", "yes", "on")


def _serve_local_web():
    """Tiny settings page bound to 127.0.0.1 only — the same settings the CLI
    edits, but in a browser. Toggled on/off from the CLI (quaestio localweb).
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import urllib.parse

    LOCAL_KEYS = [
        ("OLLAMA_BASE_URL", "AI endpoint (Ollama URL)"),
        ("OLLAMA_MODEL", "Model"),
        ("OLLAMA_TIMEOUT", "AI timeout (seconds)"),
        ("PREFIX", "Command prefix"),
        ("WARN_LIMIT", "Auto-kick after warns"),
    ]
    conffile = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    def _read_conf():
        vals = {}
        if os.path.isfile(conffile):
            with open(conffile) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        vals[k] = v
        return vals

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _html(self, vals, msg=""):
            rows = "".join(
                f"""<label>{label}
                      <input name="{k}" value="{vals.get(k, '').replace(chr(34), '&quot;')}" autocomplete="off"></label>"""
                for k, label in LOCAL_KEYS
            )
            return f"""<!doctype html><meta charset="utf-8">
            <title>Quaestio — local settings</title>
            <style>
              body{{font-family:system-ui,sans-serif;background:#0e0e16;color:#eee;display:grid;place-items:center;min-height:100vh;margin:0}}
              form{{background:#16161f;border:1px solid #2a2a3a;border-radius:14px;padding:28px;width:min(420px,90vw);display:flex;flex-direction:column;gap:12px}}
              h1{{font-size:18px;margin:0 0 4px}}
              p{{color:#888;margin:0 0 8px}}
              label{{display:flex;flex-direction:column;gap:4px;font-size:13px;color:#aaa}}
              input{{padding:9px 10px;border-radius:8px;border:1px solid #333;background:#0e0e16;color:#eee;font-size:14px}}
              button{{padding:11px;border-radius:8px;border:0;background:#6366f1;color:#fff;font-weight:600;font-size:14px;cursor:pointer}}
              .msg{{color:#7ee787;font-size:13px}}
              .hint{{font-size:12px;color:#666}}
            </style>
            <form method="post">
              <h1>Quaestio · local settings</h1>
              <p>Only reachable from this machine (127.0.0.1). Read + write the same settings file as the CLI.</p>
              {"<p class='msg'>Saved.</p>" if msg else ""}
              {rows}
              <button>Save</button>
              <span class="hint">Token is not shown here for safety — edit it with the CLI (quaestio settings).</span>
            </form>"""

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self._html(_read_conf()).encode())

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode()
            data = urllib.parse.parse_qs(body)
            vals = _read_conf()
            for k, _label in LOCAL_KEYS:
                if k in data:
                    vals[k] = data[k][0].strip()
            with open(conffile, "w") as f:
                for k, v in vals.items():
                    f.write(f"{k}={v}\n")
            os.chmod(conffile, 0o600)
            self.send_response(303)
            self.send_header("Location", "/?saved=1")
            self.end_headers()

    port = int(os.environ.get("LOCAL_WEB_PORT", "8123"))
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except OSError:
        try:
            ThreadingHTTPServer(("localhost", port), Handler).serve_forever()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not BOT_TOKEN:
        print("Quaestio: no BOT_TOKEN set — bot idle. Add one with `quaestio settings`,")
        print("then restart (Linux: sudo systemctl restart quaestio). Pool hosting still works via Ollama.")
        # Exit 0 (not 1) so systemd Restart=on-failure does NOT crash-loop
        # a token-less pool-only box. The service stays installed but idle.
        sys.exit(0)
    MIN_PY = (3, 10)
    if sys.version_info < MIN_PY:
        print(f"ERROR: Python {'.'.join(map(str, MIN_PY))}+ required (got {sys.version_info[0]}.{sys.version_info[1]}).")
        sys.exit(1)
    try:
        bot.run(BOT_TOKEN)
    except discord.LoginFailure:
        print("ERROR: Invalid BOT_TOKEN.")
        sys.exit(1)