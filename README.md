# Dental OSCE Agent

An OSCE-style dental trainer. It invents a patient case, plays the patient in free
text, walks the encounter through clinical phases at its own pace, then grades the
whole thing against a rubric written before the learner spoke.

For clinical and final-year dental students, or a new dentist. You are practising
judgement, not recalling facts.

## How the encounter works

The patient will not volunteer anything you do not ask for, and will not correct you
while you work. Both are deliberate: an OSCE does not hand out findings, and being
told you were wrong mid-consultation is not how the real thing goes.

Seven phases run in order — complaint, history, examination, investigations,
diagnosis, management, closing. The agent decides when each is complete based on what
you covered, so a thorough history takes as many turns as it needs. A phase is capped
at 8 turns regardless, so a stuck model cannot hang the encounter.

Errors are not corrected live. You get the whole debrief at the end.

## How grading works

The case and its rubric are generated together in one call, before you speak. That
rubric is the examiner's answer key — it names *this* patient's drug history and *this*
patient's likely investigation, not generic dentistry. Generating it up front is what
stops the grader being generous one run and harsh the next.

Two rules do most of the work:

- **Completeness is scored separately from accuracy.** "You never asked whether she
  takes warfarin" is more useful than any comment about how you handled what you did
  ask.
- **No credit for unasked findings.** If the patient mentioned it and you did not ask,
  the patient uncovered it, not you.

Score is the percentage of rubric points earned across all phases. Bands: 80+
excellent, 65-79 competent, 50-64 borderline, below 50 needs another session. The band
is computed in code from the points, not taken from the model's own arithmetic.

Re-grade from the debrief panel without replaying the encounter.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # or: source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env          # Windows; macOS/Linux: cp
```

Put at least one provider key in `.env`. Providers are tried in this order, first one
with a key wins:

| Provider | Key | Default model |
|---|---|---|
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Gemini | `GOOGLE_API_KEY` | `gemini-2.0-flash` |
| OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-3.5-sonnet` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |

```bash
streamlit run app.py
```

Difficulty (`junior`, `mid`, `senior`) shapes case complexity and how much is hidden.
Default `mid`.

## LangGraph Studio

```bash
langgraph dev
```

The same graph is exported through `langgraph.json`, with eight nodes — seven phases
plus the debrief.

## Tests

```bash
python -m pytest -v
```

Tests run against a scripted fake model. No network, no API key needed.

The suite covers the ways this fails quietly rather than loudly: a learner who asks
nothing scoring well, a patient handing over a finding that gets credited, a model
that never finishes, and a phase whose output cannot be parsed.

## Layout

```
src/dental/
  state.py         Phase enum, DentalState, the turn ceiling
  llm.py           provider cascade
  schemas.py       structured output shapes
  case_gen.py      case + rubric, one call
  phase_runner.py  one encounter turn: reply, pacing, ceiling
  grader.py        grading, as its own subgraph
  graph.py         StateGraph assembly
  nodes/           one file per clinical phase
```

Adding a phase means adding a node file. `graph.py` wires `PHASE_ORDER` and needs no
edit.

## Not for clinical use

This is a teaching tool. It generates fictional patients and is not a source of
diagnosis or treatment advice.
