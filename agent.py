import os
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from openai import OpenAI

# Imports com fallback: pacote ou arquivos soltos
try:
    from .data_repo import DataRepo
    from .tools import TOOLS
except ImportError:  # execução fora de pacote
    from data_repo import DataRepo
    from tools import TOOLS

class LLM:
    def __init__(self, model: str = "gpt-5-mini", temperature: float = 0.1, api_key: Optional[str] = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Forneça a chave da OpenAI na interface ou via OPENAI_API_KEY.")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()

SYSTEM_PROMPT = (
    """
Você é um analista de dados que trabalha com um CSV de transações de cartão de crédito.
Responda à pergunta do usuário usando APENAS as ferramentas disponíveis.

• Quando precisar executar algo, emita UMA chamada de ferramenta no formato JSON EXATO:
  {"tool": "<nome_da_ferramenta>", "args": { ... }}
  – Não escreva texto fora do JSON quando quiser executar uma ferramenta.
  – Use apenas uma ferramenta por etapa. Se precisar de mais, peça nova observação.

Se já tiver informação suficiente da observação anterior para responder, responda em português, direto e com números‑chave.

Ferramentas:
- list_columns() → listar colunas e dtypes.
- histogram(column, bins=50) → histograma; retorna caminho da imagem.
- summary(by_class=False, columns=None) → count, mean, std, var, min, median, max.
- minmax(columns=None) → mínimos e máximos por coluna.
- fraud_ratio() → totais e proporção de fraudes.
- time_patterns(freq="H") → série temporal agregada.
- topn(column="Amount", n=10, desc=True) → top-N linhas.
- mode(column, top=5) → valores mais frequentes (moda) e contagens.
- cluster(columns=None, k=2, sample=20000) → KMeans; tamanhos/centróides/inércia.
- outliers(column, method="iqr"|"zscore", z=3.0, iqr_factor=1.5) → outliers.
- scatterplot(x, y, sample=20000) → gráfico de dispersão (imagem).
- correlation(columns=None, method="pearson"|"spearman", sample=50000) → matriz de correlação (imagem) + top pares.
- feature_importance(target="Class", test_size=0.25) → importância de variáveis + AUC.
    """
)


def run_agent(llm: LLM, repo: DataRepo, question: str, max_steps: int = 3) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Pergunta: {question}"},
    ]
    observation: Optional[str] = None

    for _ in range(max_steps):
        if observation is not None:
            messages.append({"role": "user", "content": f"OBSERVAÇÃO: {observation}"})
        reply = llm.chat(messages)

        # Tenta interpretar como JSON de tool-call
        tool_call: Optional[Dict[str, Any]] = None
        try:
            tool_call = json.loads(reply)
            if not isinstance(tool_call, dict) or "tool" not in tool_call:
                tool_call = None
        except Exception:
            tool_call = None

        if tool_call:
            tool_name = tool_call.get("tool")
            args = tool_call.get("args", {})
            func = TOOLS.get(tool_name)
            if not func:
                observation = f"ERRO: ferramenta '{tool_name}' inexistente."
                continue
            try:
                result = func(repo, args)
                observation = json.dumps({"tool": tool_name, "result": result}, ensure_ascii=False)
            except Exception as e:
                observation = f"ERRO na ferramenta {tool_name}: {str(e)}"
            continue
        else:
            return reply

    return observation or "Não foi possível concluir a análise."