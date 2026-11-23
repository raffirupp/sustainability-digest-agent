# Sustainability Digest Agent

A small automated pipeline that turns a pile of regulatory RSS feeds into a monthly, plain-language sustainability compliance digest — delivered straight to your inbox (or Slack).

Built and running in production for a small German consumer-goods manufacturer. Their sustainability team needed a way to keep track of EU regulation (ESPR, CSRD, REACH, MDR, packaging law, etc.) without reading EUR-Lex by hand every week. Everything client-specific lives in environment variables, so this repository carries no customer data.

## How it works

```
RSS feeds ──▶ OpenAI relevance filter ──▶ OpenAI digest synthesis ──▶ Email / Slack
```

1. **Fetch** — pulls new entries from one or more RSS feeds (EUR-Lex is the default source, but any feed works).
2. **Filter** — sends batches of articles to GPT-4o-mini along with a plain-language relevance description of your business. Each article gets a relevance verdict, confidence, and a short summary.
3. **Synthesize** — groups the relevant articles into fixed topic buckets (see below), each with an urgency rating and a 3–5 sentence, jargon-free explanation of what actually changes operationally.
4. **Deliver** — renders a branded HTML email (or posts formatted Slack blocks) and sends it.
5. **Track state** — remembers which article IDs it has already seen (`state.json`) so nothing gets reported twice, even across months with zero relevant news.

Runs on the 1st of every month via GitHub Actions — no server to maintain.

### Keeping the client out of the repository

Three variables hold everything that would identify the customer. They are read at runtime and
have neutral defaults, so a fork of this repository runs without them:

| Variable | What it does | Default |
|---|---|---|
| `COMPANY_PROFILE` | one sentence describing the business, used in the digest prompt | a generic manufacturer |
| `BRAND_LABEL` | the small label above the headline in the email | `Regulierungs-Digest` |
| `GREETING` | the opening line of the email | `Hallo zusammen,` |

In GitHub Actions they are repository variables, set under Settings, Secrets and variables,
Actions, tab Variables, next to `FEED_URLS` and `RELEVANCE_CRITERIA`.

### Default topics

ESPR & Ecodesign · Green Claims · Packaging & Plastics · CSRD & Reporting · Supply Chain (LkSG / CSDDD) · Materials & Latex · Cyto- & Ecotoxicity · REACH & Chemicals · Certifications (GOTS, OEKO-TEX) · Social & Labour · EU Ecolabel · Medical Devices (MDR)

Topics with nothing new that month are collapsed into a single "no significant developments" line instead of cluttering the digest.

## Setup

### 1. Local run

```bash
git clone https://github.com/raffirupp/sustainability-digest-agent.git
cd sustainability-digest-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp config.example.env .env
# fill in .env with your own values (see Configuration below)
set -a; source .env; set +a

python agent.py --dry-run   # prints the digest to the console instead of sending
```

On the very first run, use `--seed` to mark all currently-existing articles as "already seen" without generating a digest — otherwise the first run will try to process your feed's entire backlog:

```bash
python agent.py --seed
```

### 2. Automated monthly runs (GitHub Actions)

The workflow at [`.github/workflows/monitor.yml`](.github/workflows/monitor.yml) runs on the 1st of every month at 07:00 UTC (`workflow_dispatch` also lets you trigger it manually from the Actions tab).

In your repo settings, add:

**Secrets** (Settings → Secrets and variables → Actions → Secrets)
| Name | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `EMAIL_FROM` | Sender Gmail address |
| `EMAIL_TO` | Comma-separated recipient list |
| `EMAIL_APP_PASSWORD` | [Gmail app password](https://myaccount.google.com/apppasswords) for `EMAIL_FROM` |

**Variables** (Settings → Secrets and variables → Actions → Variables)
| Name | Description |
|---|---|
| `FEED_URLS` | Comma-separated RSS feed URLs |
| `RELEVANCE_CRITERIA` | Plain-language description of what's relevant to your business |
| `OPENAI_MODEL` | Optional, defaults to `gpt-4o-mini` |

The workflow has `contents: write` permission because it commits the updated `state.json` back to the repo after each run (so dedup state persists between scheduled runs).

## Configuration

All configuration is environment variables — see [`config.example.env`](config.example.env) for the full annotated list. The two that matter most:

- **`FEED_URLS`** — comma-separated RSS feeds. [EUR-Lex predefined feeds](https://eur-lex.europa.eu/predefined-rss.html) are a good starting point; you can also build custom ones via EUR-Lex's Expert Search → "Create RSS alert".
- **`RELEVANCE_CRITERIA`** — 2–4 plain-language sentences describing your industry, materials, and obligations. This is what the OpenAI filter step uses to decide what's noise vs. what matters. Be specific about what to *exclude*, too — it cuts down false positives a lot.

Output channel is auto-selected: email if `EMAIL_TO`/`EMAIL_FROM`/`EMAIL_APP_PASSWORD` are set, otherwise Slack if `SLACK_WEBHOOK_URL` is set, otherwise it just prints to console.

## Adapting it to a different company

Everything industry-specific lives in two places:

1. `RELEVANCE_CRITERIA` in your `.env` / repo variables — controls the filtering step.
2. `TOPICS` at the top of [`agent.py`](agent.py) — controls how the digest is grouped. The tone and the business context come from the `COMPANY_PROFILE` variable, not from the code.

Swap both out and you have the same pipeline for a completely different industry — no other code changes needed.

## Cost

Roughly **€0.01/month** running on `gpt-4o-mini` with ~120 articles processed per run (per the digest's own footer). GitHub Actions minutes are free for a monthly cron on a public repo.

## Tech stack

Python · [feedparser](https://pypi.org/project/feedparser/) · [OpenAI API](https://platform.openai.com/docs) (GPT-4o-mini, JSON mode) · GitHub Actions (cron) · Gmail SMTP / Slack Incoming Webhooks
