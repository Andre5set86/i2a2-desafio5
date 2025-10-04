# server/agent_lc.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder 

# Imports compatíveis com seu layout (pacote ou arquivos soltos)
try:
    from tools import (
        TOOLS,
        HistInput, SummaryInput, MinMaxInput, FraudRatioInput, TimePatternInput,
        TopNInput, ModeInput, ClusterInput, OutliersInput, ScatterInput,
        CorrelationInput, FeatureImportanceInput,
    )
    from data_repo import DataRepo
except Exception:
    from tools import (
        TOOLS,
        HistInput, SummaryInput, MinMaxInput, FraudRatioInput, TimePatternInput,
        TopNInput, ModeInput, ClusterInput, OutliersInput, ScatterInput,
        CorrelationInput, FeatureImportanceInput,
    )
    from data_repo import DataRepo


# ============== MEMÓRIA POR SESSÃO ==============
# Pool simples em memória (processo do servidor). Em produção, trocar por Redis/DB.
_MEMORY_POOL: Dict[str, ConversationBufferMemory] = {}

def get_memory(session_id: str) -> ConversationBufferMemory:
    mem = _MEMORY_POOL.get(session_id)
    if mem is None:
        # padronize as chaves para bater com o agente
        mem = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",            # <<< o agente usa "input" como entrada
            output_key="output",          # <<< e "output" como saída
            return_messages=True,
        )
        _MEMORY_POOL[session_id] = mem
    return mem

def clear_memory(session_id: str) -> None:
    _MEMORY_POOL.pop(session_id, None)


# ---- Schema vazio p/ tools sem parâmetros ----
class _Empty(BaseModel):
    pass

# ---- Mapeia cada tool ao seu schema (IMPORTANTÍSSIMO p/ function calling) ----
SCHEMAS = {
    "list_columns": _Empty,
    "histogram": HistInput,
    "summary": SummaryInput,
    "minmax": MinMaxInput,
    "fraud_ratio": FraudRatioInput,
    "time_patterns": TimePatternInput,
    "topn": TopNInput,
    "mode": ModeInput,
    "cluster": ClusterInput,
    "outliers": OutliersInput,
    "scatterplot": ScatterInput,
    "correlation": CorrelationInput,
    "feature_importance": FeatureImportanceInput,
}

DESCRIPTIONS: Dict[str, str] = {
    "list_columns": "Lista colunas e tipos do CSV.",
    "histogram": "Gera histograma de uma coluna numérica. Requer 'column'; opcional 'bins'.",
    "summary": "Estatísticas: count, mean, std, var, min, median, max. Pode filtrar por 'columns' e 'by_class'.",
    "minmax": "Mínimo e máximo por coluna. 'columns' opcional; se vazio, todas.",
    "fraud_ratio": "Totais e proporção de fraudes com base na coluna 'Class' (0/1).",
    "time_patterns": "Série temporal agregada por frequência ('H','D',...). Requer coluna 'Time'.",
    "topn": "Top N linhas por coluna (ex.: 'Amount'). Campos: column, n, desc.",
    "mode": "Valores mais frequentes de uma coluna (top N).",
    "cluster": "KMeans em colunas numéricas (opcional 'columns'); parâmetros: k, sample.",
    "outliers": "Detecta outliers por 'iqr' ou 'zscore' em uma coluna numérica.",
    "scatterplot": "Gera gráfico de dispersão entre duas colunas numéricas (x,y).",
    "correlation": "Matriz de correlação (pearson/spearman) + top pares.",
    "feature_importance": "Importância de variáveis (árvore) para prever 'target' (padrão 'Class').",
}


def _wrap_tool(repo: DataRepo, name: str, fn, schema: Optional[Type[BaseModel]]):
    def _runner(**kwargs):
        try:
            if schema is None:
                args = {}
            else:
                model = schema(**(kwargs or {}))
                args = getattr(model, "model_dump", getattr(model, "dict"))()
        except ValidationError as ve:
            req = list(getattr(schema, "model_fields", {}).keys()) if schema else []
            return {"error": f"Parâmetros inválidos para '{name}'", "detail": str(ve), "required": req, "given": kwargs or {}}
        except Exception as e:
            return {"error": f"Falha ao validar parâmetros de '{name}'", "detail": str(e), "given": kwargs or {}}

        try:
            return fn(repo, args)
        except Exception as e:
            return {"error": f"Erro na ferramenta '{name}'", "detail": str(e), "args": args}

    args_schema = schema or _Empty
    desc = DESCRIPTIONS.get(name, f"Ferramenta '{name}'.")
    return StructuredTool.from_function(
        func=_runner,
        name=name,
        description=desc,
        args_schema=args_schema,
        return_direct=False,
    )


def build_lc_agent(
    api_key: str,
    repo: DataRepo,
    model: str = "gpt-4o-mini",
    temperature: float = 0.4,
    session_id: str = "default",
):
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    tools = [_wrap_tool(repo, n, fn, SCHEMAS.get(n, _Empty)) for n, fn in TOOLS.items()]
    memory = get_memory(session_id)

    # >>> CRÍTICO: injeta o placeholder do histórico no prompt do agente
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        agent_kwargs={
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
        },
        max_iterations=8,
        early_stopping_method="generate",
        verbose=False,
        handle_parsing_errors=True,
    )
    return agent


def _collect_image_urls_from_obs(observation: Any) -> List[str]:
    urls: List[str] = []
    def _walk(obj: Any):
        if isinstance(obj, dict):
            if isinstance(obj.get("image_url"), str):
                urls.append(obj["image_url"])
            if isinstance(obj.get("image_path"), str):
                import os
                urls.append(f"/images/{os.path.basename(obj['image_path'])}")
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
    _walk(observation)
    # dedup
    seen = set(); out = []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out


def run_lc_agent(api_key: str, repo: DataRepo, question: str, session_id: str = "default") -> Dict[str, Any]:
    agent = build_lc_agent(api_key=api_key, repo=repo, session_id=session_id)
    res = agent.invoke({"input": question})
    output = res.get("output") or res
    inter = res.get("intermediate_steps", []) or res.get("INTERMEDIATE_STEPS", []) or []
    images: List[str] = []
    for item in inter:
        if isinstance(item, tuple) and len(item) == 2:
            images.extend(_collect_image_urls_from_obs(item[1]))
        else:
            images.extend(_collect_image_urls_from_obs(item))
    # dedup final
    seen = set(); images = [u for u in images if not (u in seen or seen.add(u))]
    return {"answer": output, "images": images}
