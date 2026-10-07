# Research

What already exists, what research says, and what this project borrows.
Researched October 2026. Links are at the end.

---

## 1. Summary

- **The problem is well known.** Deciding "which source should I search?" has
  been studied since the 1990s. Search researchers call it **resource
  selection**.
- **Today's tools ask an LLM every time.** LlamaIndex, LangChain, Onyx and
  Glean all use an LLM (or search everything) to pick sources.
- **This project puts a cheap model in front of the LLM.** Research calls this
  a **cascade**: cheap first, expensive only when needed. A 2024 paper (*Online
  Cascade Learning*) studies almost exactly this design and reports large cost
  savings.
- **There's no public dataset for this exact task.** We built our own from
  public data (see DATASET.md).

---

## 2. Existing tools

| Tool | How it picks sources | What we take from it |
|---|---|---|
| **Onyx** (open source) | Asks an LLM which connected sources to search | Its prompt style for our LLM fallback. Its known bug: it picks "web" for Confluence questions, which we test for. Its public benchmark (EnterpriseRAG-Bench) is part of our data. |
| **LlamaIndex / LangChain** | An LLM reads source descriptions and picks | This is the "LLM every time" approach we compare against |
| **semantic-router** | Compares the question with examples per route; each route has its own threshold | One threshold per source, and "no source" as a valid answer |
| **Rasa** | Falls back when the top score is low **or** the top two are too close | Our gate's idea: unsure **or** torn between two answers → LLM |
| **RouteLLM** | Chooses a cheap vs expensive LLM | Show results as a **cost vs accuracy curve**, not one number |
| **Haystack zero-shot router** | Uses label names only, no training | A free baseline to beat |
| **Glean, Microsoft Copilot** | Commercial; internals not public | Shows the problem matters commercially |

**How we're different:**
1. We route to **data sources**, and a question can need several or none.
   Most routers pick one LLM.
2. We **don't call the LLM every time**.
3. The model **improves from the LLM's answers**, with safeguards.
4. The gate is chosen **with a safety margin**, not by eye.

---

## 3. What research says, and what we do about it

| Topic | Key finding | What we do |
|---|---|---|
| **Choosing sources** (federated search, 1995–today) | One yes/no decision per source works; source-specific clues help (ticket IDs, `#channels`, email words) | One score per source; add source clues as model features |
| **Cascades** (FrugalGPT 2023, RouteLLM 2025) | Cheap-first routing cuts cost a lot | Our whole design |
| **When cascades work** (Jitkrittum et al. 2023) | Only if low confidence really predicts mistakes | We check this before trusting the gate (Phase 4) |
| **Learning from the LLM** (Online Cascade Learning 2024) | The small model can learn from the LLM over time | Our learning loop (Phase 6) |
| **Risks of self-learning** (Nature 2024, and others) | Training on your own answers locks in mistakes; you only see the LLM's view on hard questions | Never train on the model's own answers; send 2–5% of confident questions to the LLM anyway |
| **Choosing the threshold** (selective classification, 2017) | Picking it on the same data you report on is too optimistic | Pick it on separate data, with a 95% safety margin |
| **LLMs choosing tools** (MetaTool 2024) | LLMs often pick too many tools | Don't treat LLM answers as automatically correct |
| **Generated data** (AttrPrompt 2023, EMNLP 2023) | Generated questions are too clean, and models learn shortcuts from them | Real-world data first; check for shortcuts (topic bias) |
| **Testing models** (CheckList, ACL 2020) | Small "must pass" tests catch specific failures | Behaviour tests, e.g. "search Slack, not email" → slack only |

---

## 4. Datasets we looked at

| Dataset | Used? | Why |
|---|---|---|
| EnterpriseRAG-Bench (Onyx) | ✅ | Only public benchmark with questions tagged by workplace source |
| WixQA, CLINC150, MASSIVE, Natural Questions, QMSum | ✅ | People-written questions for kb, gmail, web, notion and small talk |
| Apache Kafka Jira and Confluence | ✅ | Real tickets and design pages, used to write questions |
| Ubuntu help chat | ✅ | Real chat threads, used to write questions |
| ConcurrentQA | Partly | Only its web questions; its email questions read like trivia |
| HERB (Salesforce) | ❌ | Non-commercial licence |
| MS MARCO, ORCAS | ❌ | Research-only licence |
| WildChat (real chatbot logs) | Not yet | Could add real phrasing later |
| Enron emails | ❌ | Licence unclear, contains personal data |

**Main lesson:** each dataset has its own topic and style. Mixing them lets a
model guess the source from the topic. We measured this and fixed it with
same-topic questions (DATASET.md §7–8).

---

## 5. Engineering practices we adopted

| Practice | Prevents |
|---|---|
| `uv` with a lock file | Missing libraries (a prototype bug) |
| One config file for all settings | Hidden defaults that differ (a prototype bug) |
| Save settings, data version and scores with every run | "Which settings gave this number?" |
| Save the best model in the repo with a short description | A fresh copy that can't run (a prototype bug) |
| `gitleaks` before each commit | Committing API keys |
| Remove personal data from logs | Privacy problems |

---

## 6. Links

**Tools**
- Onyx: https://github.com/onyx-dot-app/onyx
- EnterpriseRAG-Bench: https://github.com/onyx-dot-app/EnterpriseRAG-Bench
- LlamaIndex routers: https://developers.llamaindex.ai/python/framework/module_guides/querying/router/
- LangChain router: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base
- semantic-router: https://github.com/aurelio-labs/semantic-router
- RouteLLM: https://github.com/lm-sys/RouteLLM
- Rasa fallback: https://rasa.com/docs/rasa/2.x/reference/rasa/core/policies/fallback
- Haystack zero-shot router: https://docs.haystack.deepset.ai/docs/transformerszeroshottextrouter

**Papers**
- FrugalGPT: https://arxiv.org/abs/2305.05176
- RouteLLM: https://arxiv.org/abs/2406.18665
- When does confidence-based deferral work: https://arxiv.org/abs/2307.02764
- Online Cascade Learning: https://arxiv.org/abs/2402.04513
- Selective classification: https://arxiv.org/abs/1705.08500
- Calibration: https://arxiv.org/abs/1706.04599
- Model collapse: https://www.nature.com/articles/s41586-024-07566-y
- MetaTool: https://arxiv.org/abs/2310.03128
- RAGRoute (choosing sources for RAG): https://arxiv.org/abs/2502.19280
- AttrPrompt (generated data): https://arxiv.org/abs/2306.15895
- CheckList (behaviour tests): https://aclanthology.org/2020.acl-main.442/

**Datasets**
- WixQA: https://huggingface.co/datasets/Wix/WixQA
- CLINC150: https://huggingface.co/datasets/clinc/clinc_oos
- MASSIVE: https://github.com/alexa/massive
- Natural Questions: https://huggingface.co/datasets/google-research-datasets/nq_open
- ConcurrentQA: https://huggingface.co/datasets/stanfordnlp/concurrentqa
- QMSum: https://github.com/Yale-LILY/QMSum
- Apache Kafka Jira: https://issues.apache.org/jira/projects/KAFKA
- Apache Kafka Confluence: https://cwiki.apache.org/confluence/display/KAFKA
- Ubuntu chat (IRC disentanglement): https://huggingface.co/datasets/jkkummerfeld/irc_disentangle

**Tools for engineering**
- uv: https://docs.astral.sh/uv/ · ruff: https://docs.astral.sh/ruff/ · pre-commit: https://pre-commit.com/ · gitleaks: https://github.com/gitleaks/gitleaks
