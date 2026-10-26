# Deploying Prism next to the Aetherion platform

Prism is a small model that decides which sources to search for a question (kb, jira, confluence,
slack, notion, gmail, web). This guide runs it **in shadow mode** next to the Aetherion platform:

- It receives **copies** of real questions, together with the answer the platform's intent prompt gave.
- It makes its own prediction and saves both, so the two can be compared and the model retrained.
- **It never answers clients and is never in the chat's path.** If Prism is slow or down, the
  platform carries on exactly as today.

```
Client → platform → intent prompt → answer to client              (unchanged)
              │
              └─ platform database ──► CRON (every N minutes) ──POST /intake──► Prism (Docker)
                                                                                  saves + predicts
                                                                                  GET /prediction/{id}
```

> Items marked **TO CONFIRM** are still being agreed.

---

## 1. What the server needs

| | |
|---|---|
| Docker | Docker Engine with the `docker compose` command |
| Chip type | **amd64** (Intel/AMD). The image is amd64 only. **TO CONFIRM:** if the server is ARM, an ARM build is one extra step |
| Memory | about 2 GB free (Prism uses ~0.7–1.3 GB with the model loaded) |
| Disk | ~3 GB for the image, plus the questions (small: a few MB per 10,000 questions) |
| Network | the cron must reach Prism on port 8000. **TO CONFIRM:** keep port 8000 reachable only from inside the network |
| Secrets | none. Prism runs the model only, so it needs no API key |

## 2. What the image contains

`ghcr.io/saramamatha19/prism:v1` (public, no login needed to pull) has a small Linux with Python 3.12,
the libraries, the code, the settings and the trained model. No secrets and no data. It stores
questions in `/app/intake/prism.db` inside the container, and that folder is linked to a folder on
the server (below), so the data survives updates.

## 3. Deploy (once)

```bash
# 1. A folder for the data (the questions). TO CONFIRM: the folder to use
sudo mkdir -p /data/prism

# 2. Get docker-compose.yml from the repo (the only file needed on the server)
curl -O https://raw.githubusercontent.com/saramamatha19/AetherionPrism/main/docker-compose.yml

# 3. Pull and start
docker compose pull
docker compose up -d

# 4. Check (wait ~1 minute while the model loads)
curl http://localhost:8000/health
#   {"status":"ok","model":"tfidf_ftv2_bge_logreg_v2","default_mode":"model_only"}
docker ps        # STATUS shows "healthy"
```

Another data folder: `PRISM_DATA_DIR=/my/folder docker compose up -d`.

A test page with every endpoint is at `http://<server>:8000/docs`.

## 4. The cron job (built on the platform side)

This is the only code the platform needs. Nothing in the live chat changes.

**Every N minutes** (**TO CONFIRM:** suggested 15):

1. Read the questions added since the **bookmark** (the last question already sent) from the platform database.
2. Send them to Prism in batches of **at most 500**.
3. Move the bookmark **only after Prism replies 200**. On no reply or an error, leave the bookmark
   where it is: the same questions are sent again next time.

```
every N minutes:
    rows = questions in the platform database after the bookmark, oldest first
    for each batch of up to 500 rows:
        reply = POST http://<prism-server>:8000/intake  with the batch
        if reply is 200:
            bookmark = last row of this batch
        else:
            stop; try again next run          (nothing is lost)
```

**Sending the same question twice is safe.** Prism saves each `question_id` only once and skips
repeats, so retries never create duplicates.

### What to send

```http
POST /intake
Content-Type: application/json

{
  "questions": [
    {"question_id": "108", "question": "status of ticket ABC-12?", "intent_answer": ["jira"]},
    {"question_id": "109", "question": "where is the leave policy?", "intent_answer": ["kb"]}
  ]
}
```

| Field | Meaning |
|---|---|
| `question_id` | the platform's own id for this message. **Must be unique and never change** (TO CONFIRM) |
| `question` | the client's question, as asked. No length limit |
| `intent_answer` | the sources the intent prompt chose, as a **list**: `kb`, `jira`, `confluence`, `slack`, `notion`, `gmail`, `web` (the same names Prism uses, confirmed). `[]` = no source. **TO CONFIRM:** that it's sent as a list, and that "no source" is `[]` |

Try it from the server:

```bash
curl -X POST http://localhost:8000/intake -H 'Content-Type: application/json' \
  -d '{"questions":[{"question_id":"test-1","question":"status of ticket ABC-12?","intent_answer":["jira"]}]}'
```

### What Prism replies

`200` with counts:

```json
{"saved": 2, "skipped_duplicates": 0, "skipped_empty": 0, "predicted": 2}
```

| Count | Meaning |
|---|---|
| `saved` | new questions stored |
| `skipped_duplicates` | `question_id`s Prism already had (from a retry). Fine |
| `skipped_empty` | questions with no text. Not stored |
| `predicted` | how many questions Prism's model answered in this call (it also retries any it couldn't answer before) |

| Error | Meaning | What the cron should do |
|---|---|---|
| `422` | the message has the wrong shape (e.g. a missing `question_id`) | log it and fix the cron: retrying won't help |
| `400` "Too many questions" | more than 500 in one send | send smaller batches |
| no reply / `5xx` | Prism is down or restarting | keep the bookmark, retry next run |

### Reading a prediction (optional)

```bash
curl http://localhost:8000/prediction/108
```

```json
{"question_id": "108", "status": "predicted", "prism_answer": ["jira"], "model_sure": true,
 "confidence": 0.79, "model_version": "tfidf_ftv2_bge_logreg_v2", "predicted_at": "2026-10-24T16:58:06"}
```

`"status": "waiting"` = saved but not predicted yet. `404` = no question with that id was sent.
Times are in UTC.

## 5. Updates and rollback

When a new model version is ready (e.g. `v2`), the only change is the version in `docker-compose.yml`:

```bash
# image: ghcr.io/saramamatha19/prism:v1  →  ghcr.io/saramamatha19/prism:v2
docker compose pull
docker compose up -d
```

The questions are kept (they're in the data folder, not in the container). To go back, put `v1` back
and run the same two commands.

## 6. Day to day

| Task | Command |
|---|---|
| Is it running and healthy? | `docker ps` |
| What is it doing, any errors? | `docker compose logs -f` |
| Restart | `docker compose restart` |
| Stop (data kept) | `docker compose down` |
| Server rebooted | nothing to do: it starts by itself |
| Where the questions are | `/data/prism/prism.db` (one SQLite file) |

## 7. Getting the collected questions for review

The questions in `prism.db` are reviewed and used to retrain the model.
**TO CONFIRM:** how the file reaches the Prism team (a copy on request, server access, or a download
link), and the rules for handling client questions.

## 8. Data and privacy

- `prism.db` holds real client questions. It stays on the server and is never put in git.
- The image and the code contain no client data and no secrets.
- Before any model is retrained on client questions, the repository and image will be made private
  (a retrained model contains words from the questions it learned from).
