"""Carga del grafo Mondial Europe (RDF) y tool SPARQL para el agente."""
import os
import re
from pathlib import Path

from langchain_core.tools import tool
from rdflib import Graph

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
MONDIAL_PREFIX = "http://www.semwebtech.org/mondial/10/meta#"
MAX_ROWS = 50

# Nombres en español / inglés común -> literal exacto de mon:name en Mondial.
# Mondial usa a menudo el exónimo alemán (Donau, Rhein, Weichsel, ...).
_NAME_ALIASES = {
    # países
    "españa": "Spain",
    "alemania": "Germany",
    "países bajos": "Netherlands",
    "paises bajos": "Netherlands",
    "holanda": "Netherlands",
    "república checa": "Czech Republic",
    "republica checa": "Czech Republic",
    "reino unido": "United Kingdom",
    "francia": "France",
    "italia": "Italy",
    "grecia": "Greece",
    "suiza": "Switzerland",
    "austria": "Austria",
    "polonia": "Poland",
    "portugal": "Portugal",
    "suecia": "Sweden",
    "noruega": "Norway",
    "dinamarca": "Denmark",
    "finlandia": "Finland",
    "irlanda": "Ireland",
    "hungría": "Hungary",
    "hungria": "Hungary",
    "rumanía": "Romania",
    "rumania": "Romania",
    "ucrania": "Ukraine",
    "rusia": "Russia",
    "turquía": "Turkey",
    "turquia": "Turkey",
    "bélgica": "Belgium",
    "belgica": "Belgium",
    # ríos (español / inglés -> Mondial)
    "loira": "Loire",
    "sena": "Seine",
    "rin": "Rhein",
    "rhin": "Rhein",
    "rhine": "Rhein",
    "rhein": "Rhein",
    "danubio": "Donau",
    "danube": "Donau",
    "donau": "Donau",
    "támesis": "Thames",
    "tamesis": "Thames",
    "thames": "Thames",
    "ródano": "Rhone",
    "rodano": "Rhone",
    "rhône": "Rhone",
    "rhone": "Rhone",
    "elba": "Elbe",
    "elbe": "Elbe",
    "vístula": "Weichsel",
    "vistula": "Weichsel",
    "weichsel": "Weichsel",
    "tajo": "Tejo",
    "tagus": "Tejo",
    "tejo": "Tejo",
    "duero": "Douro",
    "douro": "Douro",
    "tíber": "Tevere",
    "tiber": "Tevere",
    "tevere": "Tevere",
    "mosa": "Maas",
    "meuse": "Maas",
    "maas": "Maas",
    "mosela": "Mosel",
    "moselle": "Mosel",
    "po": "Po",
    "ebro": "Ebro",
    "guadalquivir": "Guadalquivir",
    "guadiana": "Guadiana",
    "garona": "Garonne",
    "garonne": "Garonne",
    "adigio": "Etsch",
    "adige": "Etsch",
    "etsch": "Etsch",
    "dniéper": "Dnepr",
    "dnieper": "Dnepr",
    "dniepr": "Dnepr",
    "dnepr": "Dnepr",
    "volga": "Volga",
    "oder": "Oder",
    "óder": "Oder",
}

_NAME_PREFIXES = (
    "el río ",
    "el rio ",
    "río ",
    "rio ",
    "the river ",
    "river ",
)

# Hint que se devuelve al agente cuando la consulta falla o no trae filas.
_HINT = (
    "Pista: empezá con 'PREFIX mon: <" + MONDIAL_PREFIX + ">'. "
    "Los mon:name están en el idioma de Mondial: Loira->\"Loire\", Sena->\"Seine\", "
    "Rin->\"Rhein\", Danubio->\"Donau\", España->\"Spain\". "
    "Dónde queda un río (países): "
    "SELECT ?country ?len WHERE { "
    "?r a mon:River ; mon:name \"Loire\" ; mon:length ?len ; mon:locatedIn ?loc . "
    "?loc a mon:Country ; mon:name ?country . } "
    "Ríos más largos (NO busques un nombre 'longest'): "
    "SELECT ?name ?len WHERE { "
    "?r a mon:River ; mon:name ?name ; mon:length ?len . "
    "} ORDER BY DESC(?len) LIMIT 10 "
    "Vecinos: mon:neighbor, p.ej. ?c a mon:Country ; mon:name \"Germany\" . "
    "?n mon:neighbor ?c ; mon:name ?name ."
)


_LIT_RE = re.compile(r'"([^"]+)"')


def _normalize_prefix(query: str) -> str:
    """Envuelve el IRI del PREFIX en <> si el modelo lo escribió sin los signos."""
    return re.sub(
        r"(PREFIX\s+mon:\s*)(?!<)(https?://\S+?#)",
        r"\1<\2>",
        query,
        flags=re.IGNORECASE,
    )


def _canonical_name(raw: str) -> str | None:
    """Devuelve el mon:name canónico de Mondial, o None si no hay alias/match."""
    key = raw.strip().lower()
    for prefix in _NAME_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    if key in _NAME_ALIASES:
        return _NAME_ALIASES[key]
    if key in _NAME_BY_LOWER:
        return _NAME_BY_LOWER[key]
    return None


def _rewrite_name_literals(query: str) -> str:
    """Reemplaza literales entre comillas por el nombre canónico del grafo."""

    def repl(match: re.Match) -> str:
        raw = match.group(1)
        canonical = _canonical_name(raw)
        if canonical and canonical != raw:
            return f'"{canonical}"'
        return match.group(0)

    return _LIT_RE.sub(repl, query)


# Carga los grafos de Mondial Europe (datos + ontología) en memoria.
def load_graph() -> Graph:
    """Parsea mondial-europe.rdf y mondial-meta.rdf (RDF/XML) y devuelve el grafo RDF."""
    g = Graph()
    g.parse(DATA_DIR / "mondial-europe.rdf", format="xml")
    g.parse(DATA_DIR / "mondial-meta.rdf", format="xml")
    return g


_graph = load_graph()

# Índice lowercase -> mon:name exacto, para corregir mayúsculas y alias.
_NAME_BY_LOWER = {
    str(row[0]).lower(): str(row[0])
    for row in _graph.query(
        f"PREFIX mon: <{MONDIAL_PREFIX}> SELECT DISTINCT ?n WHERE {{ ?x mon:name ?n }}"
    )
}


# Tool que el agente usa para recuperar datos del grafo mediante SPARQL.
@tool
def sparql_query(query: str) -> str:
    """Ejecuta una consulta SPARQL SELECT sobre el grafo Mondial Europe y devuelve las filas como texto.

    Reglas para escribir la consulta:
    - Empezá con: PREFIX mon: <http://www.semwebtech.org/mondial/10/meta#>
    - Los literales de mon:name usan el nombre de Mondial (Loira -> "Loire", Sena -> "Seine",
      Rin -> "Rhein", Danubio -> "Donau", España -> "Spain").
    - Cada triple tiene 3 términos: "sujeto predicado objeto .". No encadenes predicados.
    - Dónde queda un río:
        SELECT ?country ?len WHERE {
          ?r a mon:River ; mon:name "Loire" ; mon:length ?len ; mon:locatedIn ?loc .
          ?loc a mon:Country ; mon:name ?country .
        }
    - Ríos más largos (ORDER BY, no busques un nombre "longest"):
        SELECT ?name ?len WHERE {
          ?r a mon:River ; mon:name ?name ; mon:length ?len .
        } ORDER BY DESC(?len) LIMIT 10
    - Países vecinos = mon:neighbor. Plantilla:
        SELECT ?name WHERE {
          ?c a mon:Country ; mon:name "Germany" .
          ?n mon:neighbor ?c ; mon:name ?name .
        } LIMIT 50
    """
    query = _rewrite_name_literals(_normalize_prefix(query))
    # DEBUG: descomentar para ver en los logs de la API la consulta SPARQL enviada al grafo.
    # print(query, flush=True)
    try:
        results = _graph.query(query)
    except Exception as exc:
        return f"SPARQL_ERROR: Error en la consulta SPARQL: {exc}\n{_HINT}"

    rows = []
    for i, row in enumerate(results):
        if i >= MAX_ROWS:
            rows.append(f"... (resultados truncados a {MAX_ROWS} filas)")
            break
        rows.append(", ".join(str(v) for v in row))

    if not rows:
        return f"NO_RESULTS: La consulta no devolvió resultados.\n{_HINT}"
    return "ROWS:\n" + "\n".join(rows)
