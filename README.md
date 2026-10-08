# LLM-BR

A small GPT trained from scratch on Brazilian political speeches.

The current model has 23M parameters and was trained on every plenary speech given in the
Câmara dos Deputados from February 2023 to October 2026 (legislature 57, 648 deputies,
about 50,000 speeches). Training took about 30 minutes on an RTX 4060 Ti. Speeches from the
Senado Federal and the Presidência are planned next.

The project is a way to learn how language models work by building one end to end:
scraping and cleaning the data, tokenizing it, writing the model and the training loop,
then sampling from it. The `study/` folder has the exercises I worked through first.

## What it writes

Prompt `Sr. Presidente, Sras. e Srs. Deputados,` (temperature 0.8, top-k 50):

> Sr. Presidente, Sras. e Srs. Deputados, a Bahia tem a marca da retomada da gestão do
> ex-Governador, da gestão do Presidente Lula, na sua gestão de Governador Jerônimo. E não
> era apenas um momento simbólico, mas um momento de grande alegria, como nosso Governador,
> e que também vem fazendo um grande trabalho, um grande trabalho na Bahia e no Estado da
> Bahia. Nós estamos com o Presidente Luiz Inácio Lula da Silva. É um orgulho para todos nós
> baianos e para o povo brasileiro. [...]

Prompt `Eu quero aqui registrar`:

> Eu quero aqui registrar a presença de meu amigo da minha cidade, Edmilson de Oliveira de
> Santos, seu Vereador de Campina Grande, e também de todos os Vereadores de Campina Grande,
> a minha querida cidade de Campina Grande. Que Deus o abençoe! Um forte abraço e um forte
> abraço. Muito obrigado.

The style is convincing: forms of address, the usual openings and closings, regional
homages, vote declarations. The content is not. It gets facts wrong (the same sample later
says Bahia has "mais de 3 milhões de habitantes"), drifts off topic after a couple of
sentences, and loops when decoded greedily.

## Results

| | |
|---|---|
| Parameters | 23.0M (12.4M of them in the embedding table) |
| Training data | 20.9M tokens, 7.8 epochs, 164M tokens seen |
| Validation loss | 3.38 (perplexity 29) |
| Training loss | 3.23 |
| Unigram baseline | 7.61 |

Most of what's holding the model back is the amount of data. There's less than one unique
training token per parameter, and the gap between training and validation loss shows it
has started to memorize. The full analysis, with loss by context position, loss by token
frequency and a ranked list of improvements, is in
[docs/analysis-small-v1.md](docs/analysis-small-v1.md).

## How it works

1. Scrape. `llmbr/scrape/camara.py` downloads every speech from the
   [Câmara open data API](https://dadosabertos.camara.leg.br/swagger/api.html). It uses
   only the standard library, waits between requests, and can be stopped and resumed.
2. Clean. `llmbr/data/clean.py` strips the speaker header and stenographer notes such
   as `(Palmas.)`, fixes encoding damage, drops very short speeches and exact duplicates.
3. Tokenize. `llmbr/data/prepare.py` uses the pretrained
   [TucanoBR/Tucano-160m](https://huggingface.co/TucanoBR/Tucano-160m) tokenizer (32k
   vocabulary, built for Brazilian Portuguese). It needs 1.27 tokens per word on these
   speeches, against 2.21 for GPT-2's tokenizer. The corpus is stored as one flat `uint16`
   array per split and memory-mapped during training, so it never has to fit in RAM.
4. Model. `llmbr/model/gpt.py` is a GPT-2-style decoder: 6 pre-LayerNorm blocks,
   384-dim embeddings, 6 attention heads, 256-token context, learned position embeddings
   and an output layer tied to the token embeddings. Attention runs through PyTorch's
   fused `scaled_dot_product_attention`.
5. Train. `llmbr/train.py` uses AdamW with linear warmup and cosine decay, gradient
   clipping, and bf16 on recent NVIDIA cards. It saves `best.pt` whenever validation loss
   improves.
6. Generate. `llmbr/generate.py` samples with temperature and top-k.

## Repository layout

```
configs/        model and training sizes (tiny.toml, small.toml)
docs/           training-run analyses
llmbr/          the Python package
  scrape/       downloaders (camara.py works; senado.py and planalto.py are templates)
  data/         clean.py, prepare.py, loader.py
  tokenizer/    loads the TucanoBR tokenizer
  model/gpt.py  the transformer
  train.py      training loop
  generate.py   sampling
study/          the exercises I did before this project: tokenizers, self-attention,
                a first GPT-2 implementation
data/           downloaded and processed data (not in git)
checkpoints/    model weights (not in git)
```

## Running it

You need Python 3.11 or newer. Create the virtual environment inside the repository.

macOS (Apple Silicon, uses MPS):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows with an NVIDIA GPU (PowerShell). Install the CUDA build of PyTorch first, because a
plain `pip install torch` on Windows is CPU-only. Get the exact command for your CUDA
version from [pytorch.org](https://pytorch.org/get-started/locally/).

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
$env:PYTHONUTF8=1    # read and write the Portuguese text as UTF-8
```

Then run the pipeline from the repository root:

```bash
python -m llmbr.scrape.camara --legislatura 57          # download (about 1 hour)
python -m llmbr.data.clean
python -m llmbr.data.prepare --config configs/small.toml
python -m llmbr.train --config configs/small.toml       # add --max-steps 30 for a quick test
python -m llmbr.generate --prompt "Senhor Presidente,"
```

`configs/tiny.toml` is an 11M-parameter model that trains in a few minutes on a laptop. The
code picks CUDA, MPS or CPU on its own, and a checkpoint trained on one machine can be
sampled on another.

Earlier legislatures can be added with `--legislatura 52 53 54 55 56` (2003 to 2023):

| id | years |
|----|-----------|
| 57 | 2023-2027 |
| 56 | 2019-2023 |
| 55 | 2015-2019 |
| 54 | 2011-2015 |
| 53 | 2007-2011 |
| 52 | 2003-2007 |

## Next steps

- Scrape legislatures 52 to 56, which should give five to six times more data.
- Retrain the same model on the larger corpus, then try a 40M-parameter config with a
  512-token context.
- Try RoPE, SwiGLU and RMSNorm one at a time.
- Add a repetition penalty and top-p sampling to the generator.
- Finish the Senado and Presidência scrapers.

## Data

All training text is public record, from the Câmara dos Deputados open data service. The
downloaded data and trained weights are not stored in this repository; the pipeline above
recreates them.
