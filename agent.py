#!/usr/bin/env python3
"""
Sustainability Regulation Digest Agent
---------------------------------------
Two-step pipeline:
  1. Fetch RSS feeds, filter relevant articles via OpenAI
  2. Synthesise a structured monthly digest grouped by topic
  3. Post to Slack

All config via environment variables (see config.example.env).
"""

import argparse
import json
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import feedparser
import requests
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

FEED_URLS = [u.strip() for u in os.environ.get("FEED_URLS", "").split(",") if u.strip()]
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
RELEVANCE_CRITERIA = os.environ.get("RELEVANCE_CRITERIA", "EU sustainability legislation.")
STATE_FILE = Path(os.environ.get("STATE_FILE", "state.json"))
MAX_ITEMS_PER_RUN = int(os.environ.get("MAX_ITEMS_PER_RUN", "50"))
POST_ON_FIRST_RUN = os.environ.get("POST_ON_FIRST_RUN", "false").lower() == "true"
EMAIL_FROM = os.environ.get("EMAIL_FROM", "")
EMAIL_TO = [a.strip() for a in os.environ.get("EMAIL_TO", "").split(",") if a.strip()]
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD", "")

TOPICS = [
    "ESPR & Ecodesign",
    "Green Claims",
    "Packaging & Plastics",
    "CSRD & Reporting",
    "Supply Chain (LkSG / CSDDD)",
    "Materials & Latex",
    "Cyto- & Ecotoxicity",
    "REACH & Chemicals",
    "Certifications (GOTS, OEKO-TEX)",
    "Social & Labour",
    "EU Ecolabel",
    "Medical Devices (MDR)",
]

BATCH_SIZE = 15

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print(f"WARN: {STATE_FILE} unreadable, starting fresh.", file=sys.stderr)
    return {"seen_ids": [], "last_run": None}


def save_state(state: dict) -> None:
    state["seen_ids"] = state["seen_ids"][-5000:]
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Feed fetching
# --------------------------------------------------------------------------- #

def entry_id(entry) -> str:
    return getattr(entry, "id", None) or getattr(entry, "link", None) or getattr(entry, "title", "")


def resolve_url(url: str) -> str:
    """Follow Google News redirects to get the real article URL."""
    if "news.google.com" not in url:
        return url
    try:
        resp = requests.get(url, allow_redirects=True, timeout=8, stream=True)
        resp.close()
        return resp.url
    except Exception:
        return url


def resolve_digest_links(digest: dict) -> dict:
    """Replace Google News redirect URLs with real article URLs in the digest."""
    for topic in digest.get("topics", []):
        for link in topic.get("links", []):
            link["url"] = resolve_url(link["url"])
    return digest


def fetch_new_entries(seen_ids: set) -> list[dict]:
    new = []
    for url in FEED_URLS:
        print(f"-> loading feed: {url}")
        parsed = feedparser.parse(url)
        if parsed.bozo:
            print(f"   WARN: feed issue ({parsed.bozo_exception})", file=sys.stderr)
        for entry in parsed.entries:
            eid = entry_id(entry)
            if not eid or eid in seen_ids:
                continue
            new.append({
                "id": eid,
                "title": getattr(entry, "title", "(no title)"),
                "link": getattr(entry, "link", ""),
                "summary": getattr(entry, "summary", ""),
                "published": getattr(entry, "published", getattr(entry, "updated", "")),
            })
            seen_ids.add(eid)
    return new[:MAX_ITEMS_PER_RUN]


# --------------------------------------------------------------------------- #
# Step 1: Filter relevant articles
# --------------------------------------------------------------------------- #

FILTER_SYSTEM = """Du bist ein Compliance-Assistent, der Regulierungsnews für einen Nachhaltigkeitsmanager vorsortiert.
Du erhältst Artikel und ein Relevanzkriterium. Beurteile für JEDEN Artikel kurz die Relevanz.
Schreibe Schlagzeilen und Zusammenfassungen auf Deutsch.
Antworte NUR mit gültigem JSON, kein Markdown, keine Einleitung."""


def build_filter_prompt(items: list[dict]) -> str:
    docs = "\n\n".join(
        f"[{i}] Title: {it['title']}\nDescription: {it['summary'][:600]}"
        for i, it in enumerate(items)
    )
    return f"""Relevance criteria:
\"\"\"{RELEVANCE_CRITERIA}\"\"\"

Articles:
{docs}

Return JSON with key "results", list in article order. Each element:
{{"index": <int>, "relevant": <true|false>, "confidence": "<high|medium|low>",
  "headline": "<short English headline>",
  "summary": "<2 sentences: what changed and who is affected>"}}
Only this JSON, nothing else."""


def _filter_batch(client: OpenAI, batch: list[dict]) -> list[dict]:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        max_tokens=6000,
        messages=[
            {"role": "system", "content": FILTER_SYSTEM},
            {"role": "user", "content": build_filter_prompt(batch)},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        results = data["results"] if isinstance(data, dict) else data
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"ERROR: filter response not parseable ({e}). Raw:\n{raw[:400]}", file=sys.stderr)
        return []
    merged = []
    for r in results:
        idx = r.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(batch)):
            continue
        merged.append({**batch[idx], **r})
    return merged


def filter_articles(client: OpenAI, items: list[dict]) -> list[dict]:
    relevant = []
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i: i + BATCH_SIZE]
        print(f"-> filtering items {i+1}-{i+len(batch)} of {len(items)}...")
        assessed = _filter_batch(client, batch)
        relevant.extend(a for a in assessed if a.get("relevant"))
    return relevant


# --------------------------------------------------------------------------- #
# Step 2: Synthesise structured digest
# --------------------------------------------------------------------------- #

DIGEST_SYSTEM = """Du schreibst einen monatlichen Regulierungs-Digest für ein kleines deutsches Startup
(17 Mitarbeitende) das Kondome und Periodenprodukte herstellt – mit GOTS-zertifizierter Bio-Baumwolle
aus Tansania und Latex aus regenerativer Agroforstwirtschaft in Thailand.

Du erhältst eine Liste vorgefilterter relevanter Artikel. Gruppiere sie nach Themen und schreibe
einen prägnanten, praxisorientierten Digest auf Deutsch. Fokussiere dich auf das, was für dieses
Unternehmen konkret relevant ist.
Antworte NUR mit gültigem JSON, kein Markdown, keine Einleitung."""


def build_digest_prompt(relevant: list[dict]) -> str:
    articles = "\n\n".join(
        f"[{i}] {it.get('headline') or it['title']}\n"
        f"URL: {it['link']}\n"
        f"Summary: {it.get('summary', '')}"
        for i, it in enumerate(relevant)
    )
    topics_str = "\n".join(f"- {t}" for t in TOPICS)
    return f"""Articles this month:
{articles}

Organise these into the following 12 topics. For each topic:
- Schreibe eine 2-3-Satz-Zusammenfassung auf Deutsch, die KONKRET und PRAXISORIENTIERT ist:
  * Nenne die genaue Verordnung/Richtlinie (nicht nur "neue Regeln")
  * Nenne konkrete Deadlines oder Zeitpläne wenn bekannt
  * Sag genau was das Startup TUN oder BEOBACHTEN muss (keine generischen Ratschläge)
  * Vermeide Phrasen wie "das Unternehmen muss Compliance sicherstellen" – sei spezifisch was das bedeutet
- Bewerte die Dringlichkeit: "high" (Deadline in 12 Monaten oder sofortiger Handlungsbedarf), "medium" (beobachten, 1-2 Jahre), "low" (Frühphase, zur Info)
- Wähle bis zu 3 der relevantesten Artikel-Links mit kurzen deutschen Titeln
- Falls keine Artikel zu einem Thema passen: summary auf null setzen und urgency auf "none"

Topics:
{topics_str}

Return JSON:
{{
  "topics": [
    {{
      "name": "<topic name>",
      "urgency": "<high|medium|low|none>",
      "summary": "<2-3 sentences or null>",
      "links": [{{"title": "<short title>", "url": "<url>"}}]
    }}
  ]
}}
Only this JSON, nothing else."""


def synthesise_digest(client: OpenAI, relevant: list[dict]) -> dict | None:
    print(f"-> synthesising digest from {len(relevant)} relevant articles...")
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        max_tokens=4000,
        messages=[
            {"role": "system", "content": DIGEST_SYSTEM},
            {"role": "user", "content": build_digest_prompt(relevant)},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"ERROR: digest response not parseable ({e}). Raw:\n{raw[:400]}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# Slack output
# --------------------------------------------------------------------------- #

URGENCY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢", "none": "⚪"}


def build_slack_blocks(digest: dict, n_articles: int) -> list[dict]:
    month = datetime.now(timezone.utc).strftime("%B %Y")
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Sustainability Digest – {month}"},
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn",
                          "text": f"Based on {n_articles} relevant articles · Powered by OpenAI"}],
        },
        {"type": "divider"},
    ]

    topics = digest.get("topics", [])
    has_news = [t for t in topics if t.get("urgency") != "none" and t.get("summary")]
    no_news = [t for t in topics if t.get("urgency") == "none" or not t.get("summary")]

    for topic in has_news:
        emoji = URGENCY_EMOJI.get(topic.get("urgency", "low"), "⚪")
        urgency_label = topic.get("urgency", "").upper()
        header_line = f"{emoji} *{topic['name']}*  _{urgency_label}_"
        summary = topic.get("summary", "")
        links = topic.get("links", [])
        link_line = "  ·  ".join(f"<{l['url']}|{l['title']}>" for l in links[:3]) if links else ""
        text = f"{header_line}\n{summary}"
        if link_line:
            text += f"\n{link_line}"
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})
        blocks.append({"type": "divider"})

    if no_news:
        quiet = ", ".join(t["name"] for t in no_news)
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"⚪ No significant developments: {quiet}"}],
        })

    return blocks[:50]


def post_to_slack(digest: dict, n_articles: int) -> None:
    month = datetime.now(timezone.utc).strftime("%B %Y")
    payload = {
        "text": f"Sustainability Regulation Digest – {month}",
        "blocks": build_slack_blocks(digest, n_articles),
    }
    resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=20)
    resp.raise_for_status()
    print("-> Digest posted to Slack.")


URGENCY_COLOR = {"high": "#d93025", "medium": "#f5a623", "low": "#34a853", "none": "#9e9e9e"}
URGENCY_LABEL = {"high": "DRINGEND", "medium": "BEOBACHTEN", "low": "ZUR INFO", "none": ""}


def build_email_html(digest: dict, n_articles: int) -> str:
    month = datetime.now(timezone.utc).strftime("%B %Y")
    topics = digest.get("topics", [])
    has_news = [t for t in topics if t.get("urgency") != "none" and t.get("summary")]
    no_news = [t["name"] for t in topics if t.get("urgency") == "none" or not t.get("summary")]

    active_topics = ", ".join(t["name"] for t in has_news)

    topic_blocks = ""
    for topic in has_news:
        urgency = topic.get("urgency", "low")
        color = URGENCY_COLOR.get(urgency, "#9e9e9e")
        label = URGENCY_LABEL.get(urgency, "")
        links_html = "".join(
            f'<a href="{l["url"]}" style="color:#1a73e8;margin-right:16px;">{l["title"]}</a>'
            for l in topic.get("links", [])[:3]
        )
        topic_blocks += f"""
        <div style="margin-bottom:24px;padding:16px 20px;border-left:4px solid {color};background:#fafafa;border-radius:4px;">
          <div style="font-size:13px;font-weight:700;color:{color};letter-spacing:.5px;margin-bottom:4px;">{label}</div>
          <div style="font-size:16px;font-weight:600;color:#202124;margin-bottom:8px;">{topic["name"]}</div>
          <div style="font-size:14px;color:#3c4043;line-height:1.6;margin-bottom:12px;">{topic.get("summary","")}</div>
          <div style="font-size:13px;">{links_html}</div>
        </div>"""

    no_news_html = ""
    if no_news:
        no_news_html = f"""
        <div style="font-size:13px;color:#9e9e9e;margin-top:8px;">
          Keine wesentlichen Entwicklungen diesen Monat: {", ".join(no_news)}
        </div>"""

    return f"""
    <!DOCTYPE html><html><body style="margin:0;padding:0;background:#f1f3f4;font-family:Arial,sans-serif;">
    <div style="max-width:620px;margin:32px auto;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.15);">

      <!-- Header -->
      <div style="background:#1a73e8;padding:28px 32px;">
        <div style="color:#fff;font-size:22px;font-weight:700;">Regulierungs-Digest Nachhaltigkeit</div>
        <div style="color:#c5d8f6;font-size:14px;margin-top:4px;">{month} &nbsp;·&nbsp; {n_articles} relevante Artikel</div>
      </div>

      <!-- Greeting -->
      <div style="padding:24px 32px 0 32px;">
        <p style="font-size:15px;color:#202124;line-height:1.7;margin:0;">
          Lieber Max, liebe Einhornler,<br><br>
          das ist euer monatliches Regulierungs-Update zu den Themen
          <strong>{active_topics}</strong>.
          Hier sind die wichtigsten Entwicklungen des Monats:
        </p>
      </div>

      <!-- Topics -->
      <div style="padding:20px 32px;">
        {topic_blocks}
        {no_news_html}
      </div>

      <!-- Contact -->
      <div style="padding:16px 32px;background:#e8f0fe;border-top:1px solid #c5d8f6;">
        <p style="font-size:14px;color:#1a73e8;margin:0;">
          Wenn ihr Fragen dazu habt, meldet euch gerne bei
          <a href="mailto:raffiruppert@gmail.com" style="color:#1a73e8;font-weight:600;">raffiruppert@gmail.com</a>
        </p>
      </div>

      <!-- Technical explanation -->
      <div style="padding:20px 32px;background:#f8f9fa;border-top:1px solid #e8eaed;">
        <p style="font-size:12px;color:#5f6368;margin:0 0 8px 0;font-weight:600;letter-spacing:.5px;">WIE FUNKTIONIERT DAS?</p>
        <p style="font-size:12px;color:#80868b;line-height:1.6;margin:0;">
          Diese E-Mail wird automatisch am 1. jedes Monats verschickt – ohne dass jemand etwas tun muss.
          Ein Python-Skript läuft auf GitHub Actions, liest sechs Nachrichtenfeeds zu EU-Regulierungen
          (~50 Artikel) und schickt sie an OpenAI (GPT-4o-mini). Die KI entscheidet welche Artikel für uns
          als Hersteller von Kondomen und Periodenprodukten relevant sind, gruppiert sie nach Themen und
          schreibt die Zusammenfassungen auf Deutsch. Kosten: ca. 0,01 € pro Monat.
        </p>
      </div>

    </div>
    </body></html>"""


def send_email(digest: dict, n_articles: int) -> None:
    month = datetime.now(timezone.utc).strftime("%B %Y")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Sustainability Digest – {month}"
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(EMAIL_TO)
    msg.attach(MIMEText(build_email_html(digest, n_articles), "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    print(f"-> Email sent to {', '.join(EMAIL_TO)}.")


def print_digest(digest: dict) -> None:
    for topic in digest.get("topics", []):
        if topic.get("urgency") == "none" or not topic.get("summary"):
            continue
        emoji = URGENCY_EMOJI.get(topic.get("urgency", "low"), "⚪")
        print(f"\n{emoji} {topic['name']} [{topic.get('urgency','').upper()}]")
        print(f"  {topic.get('summary','')}")
        for link in topic.get("links", [])[:3]:
            print(f"  -> {link['title']}: {link['url']}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Sustainability Regulation Digest Agent")
    ap.add_argument("--dry-run", action="store_true", help="Print digest to console, don't post to Slack")
    ap.add_argument("--seed", action="store_true", help="Mark current entries as seen without posting")
    args = ap.parse_args()

    if not FEED_URLS:
        print("ERROR: FEED_URLS is empty.", file=sys.stderr)
        return 2

    state = load_state()
    seen = set(state["seen_ids"])
    first_run = state["last_run"] is None

    items = fetch_new_entries(seen)
    print(f"-> {len(items)} new entries found.")

    if args.seed or (first_run and not POST_ON_FIRST_RUN):
        state["seen_ids"] = list(seen)
        save_state(state)
        print("-> Seed mode: entries marked as seen, nothing posted.")
        return 0

    if not items:
        print("-> No new entries this month.")
        state["seen_ids"] = list(seen)
        save_state(state)
        return 0

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").strip())

    relevant = filter_articles(client, items)
    print(f"-> {len(relevant)} relevant articles found.")

    if not relevant:
        print("-> Nothing relevant this month, skipping digest.")
        state["seen_ids"] = list(seen)
        save_state(state)
        return 0

    digest = synthesise_digest(client, relevant)
    if not digest:
        print("ERROR: Could not create digest.", file=sys.stderr)
        return 1

    print("-> resolving article links...")
    digest = resolve_digest_links(digest)

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print_digest(digest)
    elif EMAIL_TO and EMAIL_FROM and EMAIL_APP_PASSWORD:
        send_email(digest, len(relevant))
    elif SLACK_WEBHOOK_URL:
        post_to_slack(digest, len(relevant))
    else:
        print("No output configured (EMAIL_TO or SLACK_WEBHOOK_URL missing). Use --dry-run to preview.")
        print_digest(digest)

    state["seen_ids"] = list(seen)
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
