# Dataset

The data used to train and test the Source Router.

---

## 1. In one paragraph

Each row is a **question someone might ask at work** and the **sources where
its answer lives** (for example `jira`, or `slack|gmail`). There are **4,419
questions**: 3,219 for training and 1,200 for evaluation. The data comes from
three parts: **real-world questions** from 10 public datasets, **hard cases**
that try to trick the model, and **same-topic questions** that stop the model
from guessing the source from the subject. Every row can be traced back to
where it came from.

---

## 2. Quick facts

| | |
|---|---|
| Rows | 4,419 |
| Train / eval | 3,219 / 1,200 |
| Labels | 7 sources: `kb`, `jira`, `confluence`, `slack`, `notion`, `gmail`, `web` (or none) |
| Question types | 7 (see §5) |
| Questions with 2+ sources | 426 |
| Questions needing no search | 311 |
| Built in | `Projects_Sara/Ml_Aetherion/` (scripts and sources; see §12) |
| Eval set | **Frozen**: never add, remove or change these 1,200 rows |

---

## 3. Files and columns

| File | What it contains |
|---|---|
| `data/train.csv` | 3,219 rows to train on |
| `data/eval.csv` | 1,200 rows to test on (frozen) |
| `data/provenance.csv` | Where every row came from (original ID, link, who wrote it), matched by `id` |
| `data/probes/` | Extra test questions for topic bias (§8) |

Each file has 6 columns:

| Column | Meaning | Example |
|---|---|---|
| `id` | Row number | `q00042` |
| `question` | The question | `is the login bug fixed yet` |
| `labels` | Where the answer is. Several are joined with `\|`; empty = no search needed | `jira` |
| `question_type` | What kind of question it is (§5) | `Implied source` |
| `origin` | Which dataset it came from | `Apache Kafka Jira` |
| `split` | `train` or `eval` | `train` |


---

## 4. Labels: the 7 sources

| Label | Where the answer lives | Example question | Rows |
|---|---|---|---|
| `kb` | Official help, how-to or policy articles | "how do I set up direct deposit" | 726 |
| `jira` | Tickets, bugs, work status | "is the fix for the consumer hang released?" | 803 |
| `confluence` | Team docs: specs, designs, runbooks, process docs | "what does the design doc say about failover?" | 644 |
| `slack` | Chat conversations | "what did people suggest in chat for the wifi issue?" | 742 |
| `notion` | Notes: meeting notes, personal or team notes | "what did we decide in Monday's meeting?" | 656 |
| `gmail` | Email | "did the client reply about the invoice?" | 602 |
| `web` | Public knowledge outside the company | "what is a KV cache?" | 525 |
| *(none)* | No search needed | "good morning!" | 311 |

**How a label is decided:** it is **where the answer is actually found**, not a
guess from keywords. A question written from a Jira ticket is labelled `jira`
because its answer is in that ticket.

**Labelling rules** (used for every check and re-labelling):
1. **A named tool counts:** "check Slack for…" includes `slack`.
2. **Negation removes:** "not in Slack" means `slack` is **out**.
3. **Misleading words don't count:** "parking ticket", "slack off", "the
   notion of" are not Jira, Slack or Notion.
4. **Specs, designs, runbooks, process docs** → `confluence`. **Meeting notes
   and personal notes** → `notion`. **Official policy or how-to** → `kb`.
5. **Public knowledge** → `web`. Company topics are never `web`, even though
   Confluence is a website.
6. **No information need** (greetings, thanks, statements) → no label.
7. **Several labels only when clearly needed.**

---

## 5. Question types

Every question has **exactly one** type. If more than one fits, the type
**higher in the list wins**. The list runs from hardest for a model to easiest.

| # | Type | Meaning | Example → label | Rows |
|---|---|---|---|---|
| 1 | Small talk | No information needed | "thanks a lot!" → none | 311 |
| 2 | Misleading keyword | A tool's name used with another meaning | "I got a parking **ticket**, do we reimburse that?" → kb | 79 |
| 3 | Excludes a source | Rules a source out | "not the ticket, the design doc for the API" → confluence | 232 |
| 4 | Multiple sources | Needs 2 or more sources | "what did the client email and what did we decide in standup?" → gmail, notion | 426 |
| 5 | Unclear source | Several sources fit, nothing points to one | "anything on feature flags?" | 167 |
| 6 | Names the source | Names the tool or its obvious item | "any emails from the vendor about the outage?" → gmail | 576 |
| 7 | Implied source | Clear need, tool not named | "is the login bug fixed yet?" → jira | 2,628 |

**Why this matters:** results per type show *where* the model fails. Types 2–5
are where the model should hand the question to the LLM.

---

## 6. Where the data comes from

### The three parts

| Part | Rows (train / eval) | What it is | Why it's there |
|---|---|---|---|
| **A. Real-world** | 2,500 (2,000 / 500) | Questions from 10 published public datasets | Realistic questions with traceable sources |
| **B. Hard cases** | 1,160 (600 / 560) | Earlier project dataset: keyword traps, negation, vague and multi-source questions | Real data rarely contains these |
| **C. Same-topic** | 759 (619 / 140) | 22 topics written under every source, from one company's documents | Stops the model from guessing the source from the topic |

### Part A: real-world sources

| Dataset | Rows | Label(s) | What the questions are | Licence |
|---|---|---|---|---|
| EnterpriseRAG-Bench (Onyx, 2026) | 437 | jira, confluence, slack, notion, gmail; some multi-source | Benchmark questions tagged with the sources that answer them | MIT |
| WixQA | 230 | kb | Real customer help-center questions | MIT |
| CLINC150 | 350 | kb, web, none | People-written HR questions, definitions and small talk | CC-BY-3.0 |
| MASSIVE (Amazon) | 310 | gmail, web | People-written assistant requests ("any emails from jo?") | CC-BY-4.0 |
| Natural Questions (Google) | 210 | web | Real Google searches | CC BY-SA 3.0 |
| ConcurrentQA | 70 | web | People-written questions answered by Wikipedia | MIT |
| QMSum | 273 | notion | People-written questions about real meetings | MIT |
| Apache Kafka Jira | 200 | jira | Written from real Kafka Jira tickets (one per ticket) | Public ASF tracker |
| Apache Kafka Confluence | 220 | confluence | Written from real Kafka design pages (one per page) | ASF contributor grant |
| Ubuntu help chat | 200 | slack | Written from real help-chat conversations | CC-BY-4.0 |

### Who wrote the questions

| Written by | Rows |
|---|---|
| Real people (searches, help-center customers, crowd workers) | 1,443 |
| A published benchmark (EnterpriseRAG-Bench) | 437 |
| AI, from a real document (Kafka tickets, design pages, chat threads) | 620 |
| Hand-written hard cases | 263 |
| AI-generated hard cases | 897 |
| AI, from one company's documents (Part C) | 759 |

`provenance.csv` shows this for every single row.

---

## 7. How it was built: three versions

Each version fixed a problem found in the one before.

### v1: real-world data (2,500 rows)
- Downloaded 10 public datasets and mapped their categories to our 7 labels.
- Jira, Confluence and Slack had few real "ask an assistant" questions, so one
  question was written per real document (a Kafka ticket, design page or chat
  thread).
- Company names (Wix, Enron, Kafka, Redwood) were removed from questions, so
  the model can't learn "Wix → kb".
- Split 2,000 / 500 so that questions from the same document or meeting never
  appear in both train and eval.

**Problems found:**
1. A blind re-check showed 180 email questions read like public trivia. Their
   label was true but impossible to guess from the question.
2. A topic-swap test showed the model guessing the source from the **topic**.

### v2: + hard cases, fixes (3,660 rows)
- Removed the 180 unclear email questions and replaced them with real email
  requests (MASSIVE) and real Google searches.
- Added the 1,160 hard cases from the earlier dataset, converted from 5 labels
  to 7 (257 labels changed, mainly notion → confluence for specs and runbooks).
- Reduced the columns to 6 and added `question_type`.

**Problem still there:** topic bias. Each real source still came from a
different organisation (Slack = Linux chat, kb = website help, and so on).

### v3: + same-topic questions (4,419 rows) ← current
- Added 759 questions on 22 shared topics (KV cache, SSO, billing, postmortems
  and more). Each topic is written under **every** source, all from one
  company's documents.
- 4 whole topics are kept for eval only, so eval tests **topics the model has
  never seen**.
- v2 rows are unchanged. The eval set is now frozen.

---

## 8. Is it correct? Checks we ran

| Check | Result |
|---|---|
| Rebuilding gives the identical file | ✅ |
| Same document / meeting in both train and eval | ✅ Never |
| Company names left in questions | ✅ None |
| **Blind label check:** 200 rows re-labelled without seeing the original labels | 84% agree, 93% without the email rows later removed in v2 |
| **Topic bias:** model trained on v2 vs v3, tested on the same questions | Unseen topics **36% → 74%**; topic-swap test **43% → 77%**; real-world unchanged (73% → 71%) |

The topic-swap test uses questions whose topic belongs to one source but whose
intent points to another. Example: *"how do I install the VPN on Linux?"* is a
how-to (**kb**), but all the Linux questions in the real data came from Slack.

---

## 9. Baseline results

A simple model (TF-IDF + logistic regression, untuned) trained on `train` and
scored on the frozen `eval` set. Score = **exact match** (all labels right).

| Slice | Rows | Score |
|---|---|---|
| **Real-world questions** | 500 | **71%** |
| Same-topic questions (unseen topics) | 140 | 74% |
| Hard cases | 560 | 45% |
| All eval *(never quote this without the rows above)* | 1,200 | 59% |

| Question type | Rows | Score |
|---|---|---|
| Small talk | 87 | 85% |
| Names the source | 179 | 71% |
| Implied source | 520 | 71% |
| Excludes a source | 105 | 38% |
| Multiple sources | 194 | 35% |
| Unclear source | 67 | 25% |
| Misleading keyword | 48 | 25% |

**What this shows:** easy questions already work. Hard types are where the
project's confidence gate should send questions to the LLM. These are the
numbers to beat.

---

## 10. Limitations

1. **Not company data.** No row comes from our own company's employees. A set
   of 150–300 real colleague questions would be the best final test.
2. **Some questions were written by AI:** 620 from real documents, 759 from a
   fictional company's documents, and 897 generated hard cases. All are labelled
   as such in `provenance.csv`.
3. **Each real dataset has its own writing style.** A model can tell which
   dataset a question came from about 73% of the time, so always report results
   per part and per type, not only the total.
4. **Labels were checked by AI and spot-checked**, not double-labelled by humans.
5. **Few "Misleading keyword" examples** (79 in total).

---

## 11. How to talk about it

✅ **Say:**
> "4,419 questions: 2,500 real-world questions from 10 public datasets, each
> traceable to its source; 1,160 hard cases for traps and negation; and 759
> same-topic questions that stop the model guessing the source from the
> subject. Labels are where the answer actually lives, and I checked them
> blind."

❌ **Don't say:**
- "Real company data" or "our employees' questions": none of it is.
- "Real-time data": the word is **real-world**.
- "All real questions": part of it is AI-written, and the table in §6 shows exactly how much.
- The 59% total score on its own: always give the breakdown.

---

## 12. Where it was built

The dataset was built in a separate folder, **`Projects_Sara/Ml_Aetherion/`**,
so this project folder stays clean. That folder has everything needed to show
or repeat how it was made:

| In `Ml_Aetherion/` | What it is |
|---|---|
| `scripts/build_dataset.py`, `_v2.py`, `_v3.py` | The build scripts, one per version. Rebuilding gives the identical file. |
| `data/v1/`, `data/v2/`, `data/v3/` | Each version's output and detailed notes (`DATASET_CARD.md`) |
| `data/v1/sources/` | The real Kafka tickets, design pages and chat threads questions were written from |
| `data/v2/label_changes.csv` | All 257 relabelled hard-case rows, with reasons |
| `data/v1/audit/` | The blind label-check results |

To rebuild there: run the three scripts in order (v1 → v2 → v3). They need
`pandas`, `pyarrow` and `openpyxl`, and download the public datasets
automatically.
