"""TEMPLATE — download speeches (pronunciamentos) from the Senado Federal.

API docs: https://legis.senado.leg.br/dadosabertos/docs/

Plan (mirror camara.py so the output format is identical):
  1. List senators           GET /senador/lista/legislatura/{start}/{end}
  2. For each senator        GET /senador/{codigo}/discursos?dataInicio=YYYYMMDD&dataFim=YYYYMMDD
  3. Each speech has a link to its full text (UrlTexto / TextoIntegral) — fetch it.
  4. Write one JSON line per speech to data/raw/senado/leg_<N>.jsonl with the same
     keys as camara.py: source, nome, partido, uf, data, tipo, texto, ...

Notes:
  - The Senado API returns XML by default; add ".json" to the path or send
    "Accept: application/json".
  - Verify the endpoint names against the docs above before running — they have
    changed between API versions.
"""

from llmbr.scrape.common import get_json  # noqa: F401  (used once implemented)

API = "https://legis.senado.leg.br/dadosabertos"


def main() -> None:
    # TODO: implement following the plan in the module docstring.
    raise NotImplementedError("Senado scraper not implemented yet")


if __name__ == "__main__":
    main()
