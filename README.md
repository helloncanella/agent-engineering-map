# Agent engineering map

A browsable map of every talk on the [AI Engineer](https://www.youtube.com/@aiDotEngineer) YouTube channel, sorted into the components of agent engineering (context engineering, tools and MCP, evaluation, observability, improvement loops, and so on) and ranked by fit, traction and recency. The page is in Portuguese.

**Live page:** https://helloncanella.github.io/agent-engineering-map/

It updates itself every day with no API key: a small classifier, distilled from Claude Sonnet's labels, scores the new uploads on a GitHub Actions runner.

## How it works

```
 YouTube channel ──InnerTube (anonymous)──▶ new uploads since the last run
                                                   │
                       title + description + chapters
                                                   ▼
        ┌──────────────── teacher (once, local) ────────────────┐
        │ Claude Sonnet scores every talk 0-10 per component     │
        └──────────────────────────┬─────────────────────────────┘
                                   │ 1,427 labelled talks
                                   ▼
        ┌──────────────── student (daily, CI) ──────────────────┐
        │ TF-IDF + MiniLM embedding → one logistic regression    │
        │ per component (3.7 MB, CPU, seconds)                   │
        └──────────────────────────┬─────────────────────────────┘
                                   ▼
            data/mapa.json + data/vectors.i8 ──▶ index.html (GitHub Pages)
```

- **`data/videos.json`**: the source of truth. Components (each with its own classifier rule), sources, ranking parameters, and every talk with its scores. `scored_by` says who scored it: `sonnet` (teacher) or `model` (student).
- **`data/mapa.json`**: the slim copy the page reads. The page has nothing hard-coded: tabs, ranking parameters, sources and the search model all come from this file.
- **`data/vectors.i8`**: one 384-dim MiniLM vector per talk (int8), used by the page's "ask" box.
- **`model/`**: the trained classifier and its cross-validation metrics.
- **`pipeline/`**: `update.py` (daily update), `train.py` (train and measure), `sonnet.py` (optional teacher), `innertube.py` (channel reader).

## Ranking

```
score   = (w_fit · fit + w_traction · traction + w_recency · recency) / (w_fit + w_traction + w_recency)
fit      = component score 0-10 ÷ 10   (only talks scoring 5 or more are listed)
traction = percentile of views per day since publication
recency  = 0.5 ^ (age in days / 365)
default weights 50 · 25 · 25, adjustable on the page
```

## Asking a question

The "ask" box embeds your question in the browser with the same MiniLM model (via transformers.js, about 23 MB, downloaded on the first question only) and returns the 30 talks whose title, description and chapters are closest to it. It points you to talks, it does not write answers. The model is English-only, so ask in English.

## Does the distilled classifier hold up?

5-fold cross-validation over the 1,427 teacher-labelled talks: each fold trains on 80% and predicts the held-out 20%. "Fit" means a teacher score of 5 or more.

| Setup (macro average over 14 components) | Precision | Recall | F1 | Top 20 |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.83 | 0.34 | 0.47 | 0.92 |
| MiniLM embedding + logistic regression | 0.46 | 0.71 | 0.55 | 0.89 |
| bge-small / bge-base embedding | 0.46–0.49 | 0.71–0.72 | 0.55–0.58 | 0.91 |
| bge-small / bge-base + TF-IDF | 0.73–0.74 | 0.54–0.56 | 0.62–0.63 | 0.95–0.96 |
| mxbai-embed-large (335M) + TF-IDF | 0.62 | 0.67 | 0.64 | 0.97 |
| **MiniLM + TF-IDF (shipped)** | **0.65** | **0.61** | **0.63** | **0.93** |
| Sonnet re-scoring 280 talks against its own first pass | | | 0.87 | |

"Top 20" is the share of the 20 talks the student ranks highest, per component, that the teacher also rates as a fit.

What this means:

- **The top of each list is reliable.** 93% of the student's top 20 per component are talks the teacher agrees with, and that is the part of the page people read.
- **The boundary is fuzzy.** At F1 0.63 against a teacher ceiling of 0.87, the student misses or adds borderline talks. That is why its scores show on the page as "nota estimada" (estimated score).
- **Larger embedding models barely move it.** A 335M-parameter model gained 0.01 F1 at about 15 times MiniLM's size and run time, and tuning the decision threshold per component gained nothing. The student only sees a few sentences of metadata per talk, and that limits it more than model size. MiniLM also runs in the browser for the ask box, so the page and the classifier share one model.

Per component (shipped setup):

| Component | Positives | Precision | Recall | F1 | Top 20 |
|---|---|---|---|---|---|
| O modelo como componente | 210 | 0.66 | 0.72 | 0.69 | 0.95 |
| Engenharia de contexto | 308 | 0.74 | 0.68 | 0.71 | 1.00 |
| Ferramentas e MCP | 200 | 0.69 | 0.61 | 0.65 | 1.00 |
| Controle de fluxo | 240 | 0.55 | 0.57 | 0.56 | 0.90 |
| Multi-agente | 91 | 0.59 | 0.40 | 0.47 | 0.70 |
| Humano e interface | 128 | 0.60 | 0.53 | 0.56 | 0.95 |
| Segurança e guardrails | 153 | 0.71 | 0.71 | 0.71 | 1.00 |
| Custo, latência e confiabilidade | 193 | 0.66 | 0.61 | 0.63 | 0.90 |
| Estado e execução durável | 95 | 0.55 | 0.51 | 0.53 | 0.90 |
| Avaliação | 267 | 0.72 | 0.70 | 0.71 | 1.00 |
| Observabilidade | 77 | 0.61 | 0.52 | 0.56 | 0.95 |
| Loops de melhoria | 239 | 0.60 | 0.62 | 0.61 | 1.00 |
| Software factory | 102 | 0.59 | 0.64 | 0.61 | 0.80 |
| Ontologias | 41 | 0.83 | 0.73 | 0.78 | 0.95 |

`train.py` rewrites `model/metrics.json` with these numbers each time it runs.

## Running it

```
pip install -r requirements.txt
cd pipeline
python update.py                    # new uploads, scored by the student (no API key)
python update.py --refresh-views    # also refresh every talk's view count (about 1,400 requests)
python train.py                     # retrain the student on the teacher's labels and re-measure
```

Every run also refreshes the views of talks from the last 30 days, and the scheduled job refreshes all of them on Sundays. A talk YouTube refuses is kept in `pending` and retried on the next run (a removed or private talk is dropped); while any talk is pending, or if the listing never reaches a stored talk, `update.py` exits with status 1 after saving, so the scheduled job turns red instead of silently freezing.

To view the page locally, serve the folder (`python -m http.server`) and open `index.html`.

**Adding a topic.** Append a component to `components` in `data/videos.json`, with its classifier rule:

```json
{"id": "VOICE", "layer": "Tema especial", "name": "Agentes de voz", "desc": "...",
 "classifier": {"mode": "candidates", "candidates": "voice|speech|realtime audio",
                "criterion": "voice agents (speech in/out, realtime audio, telephony)",
                "title_max": "voice agent"}}
```

`mode: "all"` scores every talk; `mode: "candidates"` scores only talks matching the `candidates` regex, and `title_max` gives 10 to talks with the term in the title. A new topic has no labels yet, so the teacher has to score it once, locally (needs the `claude` CLI). Then retrain:

```
python update.py --labeler sonnet && python train.py
```

## Data collection

- **Channel listing and details:** YouTube's internal API (InnerTube, `youtubei/v1`), anonymous, no cookies or account. Description and chapters come from the `player` endpoint.
- **Teacher labels:** `claude -p --model sonnet`, in batches of 40 talks, reading title, description and chapters.
- **Taxonomy:** checked against Anthropic, 12-Factor Agents, Cognition, Manus, LangChain, Simon Willison, Lilian Weng, Chip Huyen, Hamel Husain, Eugene Yan and OWASP. The links are in the page footer.
