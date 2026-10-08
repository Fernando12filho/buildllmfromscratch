# Analysis — `small` model, run 1 (Câmara, legislature 57)

Date: 2026-10-08 · Checkpoint: `checkpoints/best.pt` (step 20,000) · Config: `configs/small.toml`

## TL;DR

| | |
|---|---|
| Final val loss | **3.377** (perplexity **29.3**) |
| Train loss (same checkpoint, 3,360 random windows) | 3.228 |
| Train/val gap | 0.15: mild overfitting has started |
| Unigram baseline (no context, token frequencies only) | 7.61 (perplexity ≈ 2,000) |
| Best step | 20,000 = the last step: val was still going down |

The model writes fluent, on-style Brazilian parliamentary Portuguese: correct forms of address,
typical openings and closings, plausible sentences. It has **no reliable facts or argument
structure** and **loops under greedy decoding**. That's normal for 23M parameters.

**The main limit is data, not code.** We have ~21M unique tokens for a 23M-parameter model, so
training went over the corpus ~7.8 times. The biggest gain will come from adding more
legislatures (section 4.1). Architecture changes come after that.

---

## 1. What we built (architecture and choices)

### Data pipeline

| Stage | Choice | Why |
|---|---|---|
| Source | Câmara open-data API, legislature 57 (Feb 2023 – Oct 2026), all 648 deputies | Public, clean JSON, full transcripts |
| Scrape | stdlib `urllib`, 0.3 s between requests, `.done` file per deputy | Polite to a government server; Ctrl+C-safe |
| Bad API pages | If a 100-item page returns HTTP 500 every time, re-fetch it 1 item at a time and skip the failing item | Deputy 204572 had 2 broken records that crashed the whole page |
| Clean | Strip the speaker header, stage notes (`(Palmas.)`…), fix cp1252 junk, NFC, drop < 50 words, exact dedupe | 51,723 → 49,909 speeches, 16.9M words |
| Tokenizer | Pretrained **TucanoBR/Tucano-160m** BPE, 32k vocab | 1.27 tokens/word vs 2.21 for GPT-2's tokenizer, so ~42% more text fits in each context window |
| Storage | One flat `uint16` array per split, `</s>` (id 2) between speeches, memory-mapped | The corpus never has to fit in RAM; the files copy easily between machines |
| Split | 1% of **speeches** to val (random, seed 1337) | Val text never appears in train |
| Batching | Non-overlapping 256-token windows, shuffled each epoch | Each token seen once per epoch |

Result: **20.9M train tokens / 215k val tokens.**

### Model (`llmbr/model/gpt.py`): GPT-2-style decoder

```
ids (B,256) -> tok_emb + learned pos_emb -> dropout
            -> 6 x [ x + Attn(LN(x)) ; x + MLP(LN(x)) ]     (pre-LayerNorm)
            -> LN -> out_head (tied to tok_emb) -> logits (B,256,32000)
```

| Hyperparameter | Value | Note |
|---|---|---|
| `emb_dim` / `n_heads` / `head_dim` | 384 / 6 / 64 | 64 per head is the GPT-2 standard |
| `n_layers` | 6 | |
| `context_length` | 256 | ≈ 200 words |
| MLP | 4× expansion, GELU | as in GPT-2 |
| Attention | `F.scaled_dot_product_attention(is_causal=True)` | fused FlashAttention-style kernel; no mask buffer |
| Positional encoding | learned absolute embedding | as in GPT-2 / `study/gpt2.py` |
| Weight tying | `out_head.weight = tok_emb.weight` | saves 12.3M params |
| Init | N(0, 0.02); residual output projections scaled by 1/√(2·n_layers) | GPT-2 trick for stable early training |
| Dropout | 0.1 (embeddings, attention, residual, MLP) | |
| QKV bias | off | |
| **Parameters** | **23.0M**, of which **12.4M (54%) are the embedding tables** | only **10.6M** are transformer blocks |

### Training (`llmbr/train.py`)

| Setting | Value |
|---|---|
| Optimizer | AdamW, β = (0.9, 0.95), fused on CUDA; weight decay 0.1 on matrices only |
| LR schedule | linear warmup 200 steps → cosine decay from 3e-4 to 3e-5 |
| Batch | 32 × 256 = 8,192 tokens/step, no accumulation |
| Steps | 20,000 → **164M tokens seen ≈ 7.8 epochs** |
| Precision | bf16 autocast + TF32 matmuls (RTX 4060 Ti) |
| Stability | gradient clipping at 1.0 |
| Throughput | ~95–100k tokens/s, so ~30 min for the whole run |
| Eval | every 500 steps over the full val set (26 batches); `best.pt` saved whenever val improves |

---

## 2. What the numbers say

### 2.1 Overall loss
- **3.38 val vs 7.61 for the unigram baseline.** The model gets most of its predictive power
  from context; it isn't just copying word frequencies.
- Perplexity 29 means that at each step the model is about as unsure as if it were choosing
  among ~29 equally likely tokens.
- **Train 3.23 vs val 3.38.** At 7.8 epochs the model has started to memorise
  training speeches. Val was still improving at the last step only because the cosine LR was
  getting very small. More of the same data would mostly make this gap wider.

### 2.2 How much context helps (val loss by position in the 256-token window)

| Positions | 0 | 1–7 | 8–31 | 32–63 | 64–127 | 128–191 | 192–255 |
|---|---|---|---|---|---|---|---|
| Loss | 5.07 | 3.78 | 3.47 | 3.36 | 3.35 | 3.35 | 3.34 |

Almost all of the benefit comes from the **first ~64 tokens**. After that the model barely
uses more context. So **raising `context_length` to 512 won't help this model much**; it first
needs more capacity and more data to use long-range information. (Bigger models get a much
steeper curve past 64.)

### 2.3 Where the loss comes from (val loss by how often the target token appears in train)

| Target token frequency in train | Share of val tokens | Loss |
|---|---|---|
| < 100 | 2% | **9.24** |
| 100 – 1k | 17% | 5.14 |
| 1k – 10k | 34% | 3.49 |
| ≥ 10k | 47% | 2.37 |

Common tokens ("de", "que", "Presidente", punctuation) are mostly solved. **Rare tokens are
where the model is weakest**, and those carry the content: names, places, numbers, technical
vocabulary. Of the 32k vocabulary, 9,323 entries appear < 10 times in train (5,240 of them
never), so their embeddings are basically untrained. That's half the parameter budget doing
little. More data fixes this directly.

### 2.4 Samples (`best.pt`, top-k 50, temperature 0.8, seed 42)

> **Senhor Presidente,** Senhoras e Senhores Deputados, é com profundo pesar, que uso esta
> tribuna hoje para celebrar o Dia Internacional da Pessoa com Deficiência, […] é essencial
> que esta Casa legislativa continue a reafirmar o compromisso de garantir que nenhuma pessoa
> com deficiência seja tratada, para que nenhum cidadão seja idoso seja beneficiado com a
> saúde mental […]

> **Sr. Presidente, Sras. e Srs. Deputados,** a Bahia tem a marca da retomada da gestão […]
> A Bahia hoje é uma das maiores economias do Brasil, que tem mais de 3 milhões de
> habitantes. A Bahia está de parabéns para o povo baiano e para o povo baiano […]

> **A reforma tributária** é, infelizmente, só a reforma da Previdência. […] O nosso voto é
> "sim" à emenda.

> **Eu quero aqui registrar** a presença de meu amigo da minha cidade, […] Vereador de
> Campina Grande, […] Que Deus o abençoe! Um forte abraço e um forte abraço. Muito obrigado.

Greedy decoding (temperature 0) falls into a loop:

> Senhor Presidente, Senhoras e Senhores Deputados, Subo hoje a esta tribuna para tratar de um
> tema que afeta diretamente a vida de milhares de brasileiros e brasileiras. O Brasil é um dos
> países que mais se encontra em situação de vulnerabilidade social. *(this sentence ×4)*

**What it does well:** register and style, forms of address, the speech genres it learned (homage,
regional praise, vote declaration, "registrar a presença"), ending speeches with `</s>` at a
natural point, near-zero repetition when sampling (distinct-3 ≥ 0.96).
**What it does badly:** facts (Bahia "3 million inhabitants"), contradictions ("profundo
pesar… para celebrar"), loss of topic after ~2 sentences, filler repetition ("para o povo
baiano e para o povo baiano"), and greedy loops.

### 2.5 The corpus

- Speech length (words): p10 107 · median 279 · p90 627 · p99 1,375. The median speech
  (~350 tokens) is longer than the 256-token window.
- Speech types: **PELA ORDEM 46%**, Breves Comunicações 28%, Discussão 6%, Discurso
  Encaminhado 5%, Como Líder 4%. Short procedural interventions dominate.
- 601 speakers; **the top 20 produce 31% of the speeches**, so the model's "voice" leans
  toward a few very active deputies.
- Years: 2023 15.3k · 2024 11.4k · 2025 16.2k · 2026 7.0k (partial year).

---

## 3. Diagnosis

1. **Data-limited.** 0.9 unique tokens per parameter (Chinchilla-optimal is ~20 tokens/param,
   all ideally unique). We made up for it by repeating the data 7.8×, and the train/val gap
   shows that's running out.
2. **Capacity-limited, second.** Only 10.6M parameters are transformer blocks, so knowledge
   and long-range coherence are limited. But growing the model without more data would just
   overfit faster.
3. **The vocabulary is too big for this corpus.** Half the parameters sit in embeddings, and
   1/3 of the vocabulary is barely trained.
4. **The training recipe is fine.** No instability, sensible LR schedule, fast kernels. No
   code bugs that cost quality.

---

## 4. Where to improve, ranked by expected impact

### 4.1 More data (biggest lever)
- **Older legislatures:** `python -m llmbr.scrape.camara --legislatura 52 53 54 55 56`. Each
  full legislature should be roughly the size of 57 (≈ 20–25M tokens; 57 isn't over yet), so
  that's **~5–6× more data, ~120–150M tokens**. At the current scrape speed, about 1 h per
  legislature.
- **Senado and Presidência:** `scrape/senado.py` and `scrape/planalto.py` are still templates.
  They add data and different registers (more formal speeches, presidential addresses).
- Keep the test split honest as data grows: also report val on a **held-out time period**
  (e.g. the most recent 3 months). The random speech split shares speakers and topics with train,
  so it overestimates how well the model generalizes.

### 4.2 Scale the model together with the data
Once there are ~120M+ tokens, a suggested `configs/medium.toml` for the 8 GB card:

```toml
[model]
context_length = 512     # worth it once the model can use it (see 2.2)
emb_dim        = 512
n_heads        = 8
n_layers       = 8
dropout        = 0.05    # lower: fewer epochs over more data

[train]
batch_size    = 16
grad_accum    = 4        # 16 x 512 x 4 = 32k tokens/step (4x today)
lr            = 6e-4
warmup_steps  = 500
max_steps     = 12000    # ~390M tokens, ~3 epochs of 130M
```
≈ 25M block params + 16M embedding ≈ **41M params**. Estimated time: 2–3 h on the 4060 Ti.
Measure memory with `--max-steps 30` first. Rule of thumb: **≤ 3–4 epochs**, and grow the
model only when the train/val gap stays small.

### 4.3 Training recipe (cheap, small but real gains)
- **Log metrics to a file** (`checkpoints/<run>/log.csv`: step, train loss, val loss, lr).
  Right now the curve only goes to stdout, so it's lost when you train in a terminal. That's
  why this report couldn't include the loss curve.
- **`torch.compile(model)`**: typically 1.3–2× faster on CUDA; Windows needs Triton
  (`triton-windows`), so make it a config flag.
- **Bigger effective batch** (`grad_accum` 4) plus a higher peak LR (6e-4): smoother gradients
  for small models.
- **Run `best.pt` selection on the full val set every eval** (already true: 26 batches < 50
  `eval_iters`; keep it true as val grows by raising `eval_iters`).

### 4.4 Architecture upgrades (the "modern GPT" set, each a few % in loss)
Good learning exercises; in rough order of value for the effort:
1. **RoPE** (rotary position embeddings) instead of learned `pos_emb`: better use of
   position and better behaviour at lengths not seen in training.
2. **SwiGLU MLP** instead of GELU (hidden size ≈ 8/3·emb_dim to keep params equal).
3. **RMSNorm** instead of LayerNorm: simpler and slightly faster.
4. Drop all biases (Linear and norms).

Change one at a time, using the same data and the same number of steps, so each val-loss change
can be credited to one thing.

### 4.5 Tokenizer
The Tucano tokenizer is a good fit for Portuguese, but 32k is a lot of vocabulary for 21M tokens.
- With ~150M tokens (4.1) the problem mostly goes away, so **keep Tucano for now**.
- Option to try later: train our own 16k BPE on the corpus, which halves embedding params and
  gives parliamentary words single tokens. Compare **bits per byte** (not loss), since loss per
  token isn't comparable across tokenizers.

### 4.6 Data quality and control
- **Metadata tags:** prepend `<tipo> <partido> <uf>` (e.g. `[BREVES COMUNICAÇÕES] [PT-BA]`)
  to each speech. The model learns to condition on them, so you can generate a speech in a
  given genre or party style. Cheap to do in `clean.py` / `prepare.py`.
- **Rebalance genres:** PELA ORDEM (46%) is mostly short procedural talk. Down-weight it, or
  give it a tag (above) so it doesn't dilute the long-form speeches.
- **Speaker imbalance:** cap speeches per deputy, or leave as is and rely on tags.
- **Near-duplicates:** exact dedupe dropped 728 speeches; there are likely more templated
  near-copies (homages, repeated notes). MinHash dedupe would catch them.

### 4.7 Generation
- Add a **repetition penalty** and/or **top-p (nucleus) sampling** in `generate.py`. That
  fixes the greedy loop and "povo baiano e para o povo baiano" without retraining.
- Keep temperature 0.7–0.9; avoid greedy decoding with a model this small.

---

## 5. Suggested next steps

1. Add CSV logging to `train.py` (small change, needed for every later comparison).
2. Scrape legislatures 52–56, re-run `clean` + `prepare`.
3. Retrain **the same `small` config** on the bigger corpus, so the data effect is measured on
   its own. Expect val to drop clearly and the train/val gap to shrink.
4. Then `medium.toml` (4.2), then architecture changes one at a time (4.4).
5. Repetition penalty / top-p in `generate.py` at any point.

## How these numbers were produced

Loss measured with `best.pt` in eval mode, bf16, over all 840 val windows and 3,360 random
train windows of 256 tokens. Unigram baseline = add-one-smoothed token frequencies from
`train.bin`. Samples: seed 42, top-k 50, 150 new tokens.
