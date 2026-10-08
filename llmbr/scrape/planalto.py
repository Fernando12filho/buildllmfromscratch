"""TEMPLATE — download presidential speeches.

Sources:
  - Current government: https://www.gov.br/planalto/pt-br/acompanhe-o-planalto/discursos-e-pronunciamentos
    HTML listing pages → each links to a page with the full speech text.
  - Past presidents: Biblioteca da Presidência (biblioteca.presidencia.gov.br).
    Was timing out when last checked; retry later.

Plan:
  1. Walk the listing pages (gov.br uses ?b_start:int=0,30,60,... for pagination).
  2. Collect each speech URL, fetch it, extract the article body text.
     No API here, so this is HTML scraping — add `beautifulsoup4` to
     requirements.txt when implementing.
  3. Write JSON lines to data/raw/planalto/discursos.jsonl with the same keys as
     camara.py (source="planalto", nome=<president>, data, texto, ...).
  4. Keep a .done file of fetched URLs so the run is resumable.
"""


def main() -> None:
    # TODO: implement following the plan in the module docstring.
    raise NotImplementedError("Planalto scraper not implemented yet")


if __name__ == "__main__":
    main()
