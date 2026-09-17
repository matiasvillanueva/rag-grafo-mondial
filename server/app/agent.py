"""Agente LangChain que responde consultas geográficas con una sola llamada SPARQL."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .graph import MONDIAL_PREFIX, sparql_query

DEFAULT_MODEL = "gemini-3.6-flash"


def _load_env() -> None:
    """Carga .env del cwd o de la raíz del repo (junto a docker-compose.yml)."""
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / ".env",
        here.parents[2] / ".env",
        here.parents[1] / ".env",
    ]
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
            return
    load_dotenv(override=False)


_load_env()


def _api_key() -> str:
    """Lee la API key de Gemini (GEMINI_API_KEY o GOOGLE_API_KEY)."""
    raw = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return raw.strip().strip('"').strip("'")


def llm_provider() -> str:
    """Proveedor del LLM (siempre Gemini)."""
    return "gemini"


def llm_model() -> str:
    """Modelo Gemini. MODEL=llama* se ignora y cae en el default."""
    raw = os.getenv("MODEL", "").strip()
    if not raw or raw.startswith(("llama", "mistral", "qwen", "phi")):
        return DEFAULT_MODEL
    return raw


def build_llm():
    """ChatGoogleGenerativeAI con la API key de Gemini."""
    key = _api_key()
    if not key:
        raise RuntimeError("Se requiere GEMINI_API_KEY o GOOGLE_API_KEY.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    os.environ.setdefault("GEMINI_API_KEY", key)
    os.environ.setdefault("GOOGLE_API_KEY", key)
    return ChatGoogleGenerativeAI(model=llm_model(), api_key=key, temperature=0)

SYSTEM_PROMPT = f"""Sos un asistente geográfico de Europa. Respondés SOLO con datos del grafo de conocimiento Mondial (Europa), consultándolo con la tool sparql_query.

El grafo es RDF y usa este prefijo:
PREFIX mon: <{MONDIAL_PREFIX}>

Clases principales: mon:Country, mon:City, mon:River, mon:Lake, mon:Sea, mon:Province.
Propiedades útiles:
- mon:name (nombre), mon:population (población), mon:area (superficie)
- mon:capital (país -> ciudad capital), mon:isCapitalOf
- mon:cityIn (ciudad -> país/provincia)
- mon:borders / mon:neighbor (países vecinos)
- mon:length (largo de un río), mon:flowsInto, mon:locatedIn

IMPORTANTE - los literales de mon:name usan el nombre de Mondial (inglés o alemán).
Traducí ANTES de armar el FILTER/triple. Países:
- España -> "Spain", Alemania -> "Germany", Países Bajos -> "Netherlands"
- República Checa -> "Czech Republic", Reino Unido -> "United Kingdom"
- Francia -> "France", Italia -> "Italy", Grecia -> "Greece", Suiza -> "Switzerland"
Ríos (ojo: Mondial usa a menudo el nombre alemán):
- Loira -> "Loire", Sena -> "Seine", Rin/Rhine -> "Rhein"
- Danubio/Danube -> "Donau", Támesis -> "Thames", Ródano -> "Rhone"
- Elba -> "Elbe", Vístula -> "Weichsel", Tajo -> "Tejo", Duero -> "Douro"
- Tíber -> "Tevere", Mosa/Meuse -> "Maas"

Reglas:
1. Antes de responder SIEMPRE ejecutá una consulta SPARQL SELECT con la tool sparql_query.
2. Escribí siempre el SELECT empezando exactamente con "PREFIX mon: <{MONDIAL_PREFIX}>".
3. Cada patrón de triple es "sujeto predicado objeto ." (tres términos). No encadenes predicados en un mismo triple.
4. Para "países que limitan con X" usá SOLO mon:neighbor (no mezcles mon:cityIn ni mon:isCapitalOf). Plantilla fija:
   PREFIX mon: <{MONDIAL_PREFIX}>
   SELECT ?name WHERE {{
     ?c a mon:Country ; mon:name "Germany" .
     ?n mon:neighbor ?c ; mon:name ?name .
   }} LIMIT 50
   Para España es el mismo patrón con "Spain".
5. "Dónde queda el río X" / ubicación de un río. Plantilla fija (nombre Mondial):
   PREFIX mon: <{MONDIAL_PREFIX}>
   SELECT ?country ?len WHERE {{
     ?r a mon:River ; mon:name "Loire" ; mon:length ?len ; mon:locatedIn ?loc .
     ?loc a mon:Country ; mon:name ?country .
   }}
   Para el Sena usá "Seine"; para el Rin usá "Rhein"; para el Danubio usá "Donau".
6. "Ríos más largos" / ranking por largo. NO busques un mon:name "longest". Plantilla fija:
   PREFIX mon: <{MONDIAL_PREFIX}>
   SELECT ?name ?len WHERE {{
     ?r a mon:River ; mon:name ?name ; mon:length ?len .
   }} ORDER BY DESC(?len) LIMIT 10
7. LIMIT: si piden "todos", usá LIMIT 50; si piden un número N, usá LIMIT N.
8. Llamá a sparql_query una sola vez. No reintentes aunque vuelva vacío o con error.
   No inventes predicados. Para rankings usá ORDER BY, no un FILTER por nombre.
9. No inventes datos. Respondé únicamente con las filas (líneas bajo ROWS:).
10. Respondé en español, breve y claro. Nunca vuelques el JSON de las tool-calls.
"""


_llm = build_llm()


NO_DATO = "No dispongo de ese dato en el grafo Mondial Europe."


def _message_text(content) -> str:
    """Gemini 3 a veces devuelve content como lista de bloques, no un string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(p for p in parts if p).strip()
    return str(content)


# Punto de entrada usado por la API para responder una pregunta.
def answer(question: str) -> str:
    """Una llamada SPARQL: si hay filas, redacta; si no, falla."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    first = _llm.bind_tools([sparql_query]).invoke(messages)
    messages.append(first)

    tool_calls = getattr(first, "tool_calls", None) or []
    if not tool_calls:
        return NO_DATO

    tool_result = None
    for i, tc in enumerate(tool_calls):
        if i == 0:
            tool_result = sparql_query.invoke(tc.get("args") or {})
            content = tool_result
        else:
            content = "Ignorada: una sola consulta SPARQL por pregunta."
        messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))

    if not (isinstance(tool_result, str) and tool_result.startswith("ROWS:")):
        return NO_DATO

    # Sin tools: el modelo no puede reintentar SPARQL.
    final = _llm.invoke(messages)
    return _message_text(final.content)
