"""Download plenary speeches from the Câmara dos Deputados open data API.

API docs: https://dadosabertos.camara.leg.br/swagger/api.html

How it works:
  1. List every deputy who served in a legislature   (GET /deputados?idLegislatura=N)
  2. For each deputy, page through their speeches     (GET /deputados/{id}/discursos)
  3. Append each speech (with full transcript) as one JSON line to
     data/raw/camara/leg_<N>.jsonl

Resumable: finished deputies are recorded in leg_<N>.done, so Ctrl+C and
re-running continues where it stopped instead of starting over.

Examples (run from the repository root):
  python -m llmbr.scrape.camara --legislatura 57 --max-deputados 3   # quick test
  python -m llmbr.scrape.camara --legislatura 57                     # full 2023–2027
  python -m llmbr.scrape.camara --legislatura 52 53 54 55 56 57      # 2003 → today
"""

import argparse
import json
from pathlib import Path

from llmbr.scrape.common import get_json

API = "https://dadosabertos.camara.leg.br/api/v2"
PAGE_SIZE = 100  # API maximum


def paginate(url: str, params: dict):
    """Yield every item across all pages, following the API's rel="next" links."""
    params = {**params, "itens": PAGE_SIZE, "pagina": 1}
    while True:
        try:
            payload = get_json(url, params)
        except RuntimeError:
            # A single broken record can make the API return HTTP 500 for the whole
            # page, every time (seen on deputy 204572, page 5). Re-fetch that page
            # one item at a time and skip only the item(s) that keep failing.
            yield from page_item_by_item(url, params)
            params["pagina"] += 1
            continue
        yield from payload["dados"]
        if not any(link["rel"] == "next" for link in payload.get("links", [])):
            return
        params["pagina"] += 1


def page_item_by_item(url: str, params: dict):
    """Yield the items of one failed page by requesting pages of size 1.

    Page p of size N holds items (p-1)*N+1 .. p*N, and with itens=1 item k is
    simply page k. An empty page means we ran past the end of the list.
    """
    first = (params["pagina"] - 1) * params["itens"] + 1
    for k in range(first, first + params["itens"]):
        try:
            items = get_json(url, {**params, "itens": 1, "pagina": k}, retries=2)["dados"]
        except RuntimeError:
            print(f"  ! skipping item {k}: the API keeps failing on it", flush=True)
            continue
        if not items:
            return
        yield from items


def legislature_dates(leg_id: int) -> tuple[str, str]:
    """Start/end dates of a legislature, e.g. 57 -> ("2023-02-01", "2027-01-31")."""
    d = get_json(f"{API}/legislaturas/{leg_id}")["dados"]
    return d["dataInicio"], d["dataFim"]


def list_deputies(leg_id: int) -> list[dict]:
    # The same person can appear more than once (e.g. after changing party); dedupe by id.
    seen = {}
    for dep in paginate(f"{API}/deputados", {"idLegislatura": leg_id, "ordem": "ASC",
                                             "ordenarPor": "nome"}):
        seen.setdefault(dep["id"], dep)
    return list(seen.values())


def speeches_for(dep_id: int, start: str, end: str):
    yield from paginate(f"{API}/deputados/{dep_id}/discursos",
                        {"dataInicio": start, "dataFim": end,
                         "ordenarPor": "dataHoraInicio", "ordem": "ASC"})


def scrape_legislature(leg_id: int, out_dir: Path, max_deputados: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"leg_{leg_id}.jsonl"
    done_path = out_dir / f"leg_{leg_id}.done"
    done = set(done_path.read_text().split()) if done_path.exists() else set()

    start, end = legislature_dates(leg_id)
    deputies = list_deputies(leg_id)[:max_deputados]
    print(f"Legislatura {leg_id} ({start} to {end}): {len(deputies)} deputies, "
          f"{len(done)} already done")

    total = 0
    for i, dep in enumerate(deputies, 1):
        if str(dep["id"]) in done:
            continue
        n = 0
        # Write a deputy's speeches to memory first, then append in one go, so an
        # interrupted run never leaves half a deputy in the file.
        lines = []
        for sp in speeches_for(dep["id"], start, end):
            text = (sp.get("transcricao") or "").strip()
            if not text:
                continue  # some entries only have audio/video
            lines.append(json.dumps({
                "source": "camara",
                "legislatura": leg_id,
                "deputado_id": dep["id"],
                "nome": dep["nome"],
                "partido": dep.get("siglaPartido"),
                "uf": dep.get("siglaUf"),
                "data": sp.get("dataHoraInicio"),
                "fase": (sp.get("faseEvento") or {}).get("titulo"),
                "tipo": sp.get("tipoDiscurso"),
                "keywords": sp.get("keywords"),
                "sumario": sp.get("sumario"),
                "texto": text,
            }, ensure_ascii=False))
            n += 1
        with open(out_path, "a", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in lines)
        with open(done_path, "a", encoding="utf-8") as f:
            f.write(f"{dep['id']}\n")
        total += n
        print(f"[{i}/{len(deputies)}] {dep['nome']} ({dep.get('siglaPartido')}-"
              f"{dep.get('siglaUf')}): {n} speeches", flush=True)  # flush: visible in logs

    print(f"Done. +{total} speeches -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legislatura", type=int, nargs="+", default=[57],
                    help="legislature id(s); 57 = 2023–2027")
    ap.add_argument("--out", type=Path, default=Path("data/raw/camara"))
    ap.add_argument("--max-deputados", type=int, default=None,
                    help="only the first N deputies (for testing)")
    args = ap.parse_args()
    for leg in args.legislatura:
        scrape_legislature(leg, args.out, args.max_deputados)


if __name__ == "__main__":
    main()
