# jev — jpt-4b as a Jev-like decision service (llama.cpp)

A local, open implementation of the **Jev typed-decision API** (the `/v1/systemone` wire format
made popular by TypeSafe's closed "Jev" model), served by **jpt-4b** — an open
calibrated-decision fine-tune of Qwen3.5-4B (base `kirp/jpt-4b`, GGUF by `prithivMLmods`).

Given a `state` and typed questions, the model returns a **calibrated probability for every
option in a single prefill** — no generated reasoning tokens.

## Layout

| Path | What it is |
|---|---|
| `jpt-4b.Q4_K_M.gguf` | model weights, 2.7 GB, sha256 `35233cef…b3c39` — **not in this repo**; download from `prithivMLmods/jpt-4b-GGUF` (hf-mirror.com mirror works) and drop it next to the scripts |
| *(engine-agnostic)* | the stack runs any chat GGUF llama.cpp supports: `jev_config.bat` points `MODEL`/`MODEL_NAME`/`MODEL_TEMP` at the weights. Current local deployment: **Spark-X2.5-4B** (`spark2_5` arch, temperature 1.0, `-ngl all` on an RTX 4050 6 GB — fits weights + full 32k×4 KV) |
| `jev_adapter.py` | `/v1/systemone` HTTP service on top of llama-server (the llama.cpp backend llm2jev doesn't ship); also serves the UI at `/` |
| `jev_ui.html` | 决策台 single-file UI (served by the adapter at `http://127.0.0.1:8090/`) |
| `start_jev.bat` / `stop_jev.bat` | start/stop llama-server (port 8091) + the adapter (port 8090) |
| `test_jev.py` | smoke test with the example from the model card |
| `jev_config.bat.example` | demo template for the local config — copy to `jev_config.bat` (gitignored) and fill in your real python / llama-server paths |
| `junqi/` | **军棋对抗模块** — full rules-refereed 陆战棋 environment: JEV as a player vs an α-β tree-search AI, with auto-generated HTML replays. See [junqi/README.md](junqi/README.md) |

Engine: any llama.cpp build that supports the model's arch (`qwen35` for jpt-4b,
`spark2_5` for Spark-X2.5-4B; CPU-only is fine, `-ngl all` if it fits in VRAM).
Machine-specific paths (python, llama-server, model weights) go in `jev_config.bat` —
copy `jev_config.bat.example` and fill in real values; that file is gitignored so they
never reach the repo. Adapter: pure stdlib + `requests`.

## Run

```bat
start_jev.bat
```

Wait until the console prints `llm2jev-llamacpp on http://127.0.0.1:8090 ...` (model load
takes ~a minute), then open the UIs:

- **决策台（本部署的 UI）**：<http://127.0.0.1:8090/> — 编辑状态和问题（是否/选择/评分/自由决策），
  一键发送并以概率条可视化每个选项的分布、期望分值与置信度；自带三个示例。
- **llama.cpp 内置聊天 WebUI**：<http://127.0.0.1:8091/> — 原始对话界面（注意这是决策
  微调模型，自由聊天表现不是它的设计用途）。

Or drive the API directly:

```bash
curl http://127.0.0.1:8090/v1/systemone -H "Content-Type: application/json" -d '{
  "state": [{"role":"system","content":"You are a support assistant."},
            {"role":"user","content":"I was charged twice. Please refund the duplicate."}],
  "questions": {
    "refund":    {"type":"noul","instructions":"Does the user request a refund?"},
    "department":{"type":"choice","instructions":"Which department should handle this?",
                  "criteria":{"billing":"Payments and refunds","technical":"Software bugs"}},
    "urgency":   {"type":"score","instructions":"How urgent is the request?",
                  "criteria":["Routine","Urgent","Emergency"]}}}'
```

Or `python test_jev.py`. Endpoints mirror llm2jev 0.6.x: `POST /v1/systemone`,
`POST /v1/systemone/auto`, `GET /health`, `GET /v1/models`.

## 自由决策 (auto) — options without presets

The real Jev cannot do this either: per [TypeSafe's System One docs](https://docs.typesafe.ai/concepts/system-one),
System One models "do not write replies, produce code, or generate explanations" and every
answer is drawn from a **caller-defined** answer space (the official guidance is to add an
OTHER/ESCALATE option since probability is forced onto declared labels). The documented
pattern is: *another system proposes candidates first, the decision model chooses*.

This deployment implements exactly that pattern as an optional extra endpoint, leaving
`/v1/systemone` strictly compatible:

```bash
curl http://127.0.0.1:8090/v1/systemone/auto -H "Content-Type: application/json" -d '{
  "state": "I received two identical items in my order, but I only paid for one.",
  "questions": {"action": {"type": "auto",
                           "instructions": "How should the customer handle this duplicate delivery?",
                           "num_options": 4}}}'
```

1. A **generator** LLM proposes the candidate options (OpenAI-compatible chat endpoint;
   default is the local engine's own generative mode — a decision-tuned 4B, so quality is
   modest; point it at a stronger model with `--generator-url` / `--generator-model`).
2. jpt-4b scores the generated options as a normal calibrated choice; the options it chose
   from are returned in `answer.generated_options`. The UI shows the same as the
   「自由决策 (自动生成选项)」 question type.

In the UI this is the 示例：自由决策 button.

## How it works

1. The state plus one user turn (instruction + marker) is rendered with the model chat
   template, thinking off, via llama-server's `/apply-template`; the marker splits into a
   shared prefix (KV-cache friendly) and the template ending.
2. Each question becomes `prefix + "Question: …/Options: A. … B. …" + ending + "Answer:"`,
   one prefill, and the distribution over the next token's option labels (A, B, … single
   tokens) is read.
3. That distribution is obtained **exactly**: the request samples under a GBNF grammar that
   only allows the label tokens, with `post_sampling_probs: true` — the reported post-grammar
   distribution is the raw softmax renormalized over the labels, identical to what llm2jev's
   reference HF backend computes. A global `--temperature 1.036` (per the model card) is
   applied on top.
4. Labels are discovered at adapter startup with the engine's own tokenizer
   (`/tokenize`), checked to be single tokens right after the real prompt ending.

Differences from stock llm2jev: text-only (vision would need the repo's
`jpt-4b.mmproj-bf16.gguf` and adapter support), `--prompt jevlm` style not implemented.

## 军棋模块 (junqi)

`junqi/` turns the decision service into a board-game player: a complete 二人陆战棋
(open-hand military chess) environment where **JEV plays against a rule-evaluation +
iterative-deepening α-β tree-search AI**, every move validated by a referee engine
(railway slides, sapper turns, camp immunity, HQ freeze, mines/bombs, flag capture and
stalemate — 8 rule selftest assertions). Each game auto-generates a self-contained HTML
replay (board rendering, animated moves, move list, casualty trays).

```bash
python junqi/junqi.py --selftest          # referee rule assertions
python junqi/junqi.py --games 4           # match: JEV vs RULE-AI, alternating colors
# -> junqi/games/game_<ts>_gN.html replays
```

Full usage, parameters, rules list and match results: [junqi/README.md](junqi/README.md).
Measured over two 4-game matches: RULE-AI 8:0 — expected for a 4B decision model reading
text options against a 2-second tree search; honest combat-outcome option labels still
raised average game length by 29%.

## Ports

| Port | Service |
|---|---|
| 8090 | `/v1/systemone` adapter (Jev-like API) + UI |
| 8091 | llama-server (llama.cpp engine) |

Engine flags: `-c 32768 -np 4` → four parallel slots of 8192 context; `-np` can be raised
with more questions in flight, at half context per doubling.

## git

Model weights (`.gguf`), local config (`jev_config.bat`) and runtime artifacts are gitignored.

```bash
git remote add origin git@github.com:Pawnnwap/jev.git   # once
git push -u origin main
```
