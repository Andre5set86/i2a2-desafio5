import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import gradio as gr

from data_repo import DataRepo
from agent import LLM, run_agent
from tools import TOOLS

# ---------------- Helpers de formatação ----------------

def _extract_images_from_payload(payload_text: str) -> List[str]:
    try:
        data = json.loads(payload_text)
        if isinstance(data, dict):
            result = data.get("result")
            if isinstance(result, dict):
                img = result.get("image_path")
                if img:
                    return [img]
        return []
    except Exception:
        return []


def _mk_table(kv: Dict[str, Any]) -> str:
    lines = ["| Item | Valor |", "|---|---|"]
    for k, v in kv.items():
        lines.append(f"| {k} | {v} |")
    return "".join(lines)


def _prettify(answer: str) -> str:
    #"""Padroniza a resposta da LLM para leigos: títulos, listas e tabelas legíveis."""
    # Se vier JSON (resultado de ferramenta), transformar em um texto amigável
    try:
        data = json.loads(answer)
        if not isinstance(data, dict):
            return answer
        tool = data.get("tool")
        result = data.get("result", {})
        if tool == "fraud_ratio":
            return (
                "### Proporção de Fraudes"
                + _mk_table({
                    "Transações": result.get("total"),
                    "Legítimas": result.get("legit"),
                    "Fraudes": result.get("frauds"),
                    "Proporção de Fraudes": f"{result.get('fraud_ratio', 0):.6f}",
                })
            )
        if tool == "minmax":
            rows = result.get("minmax", {})
            lines = ["### Intervalos (mín–máx)", "| Coluna | Mín | Máx |", "|---|---:|---:|"]
            for col, mm in rows.items():
                lines.append(f"| {col} | {mm.get('min')} | {mm.get('max')} |")
            return "".join(lines)
        if tool == "correlation":
            pairs = result.get("top_pairs", [])
            lines = ["### Correlações mais fortes (top 10)", "| Variável A | Variável B | |ρ| |", "|---|---|---:|"]
            for a, b, v in pairs:
                lines.append(f"| {a} | {b} | {v:.3f} |")
            lines.append("*O heatmap foi gerado do lado direito.*")
            return "".join(lines)
        if tool == "outliers":
            if result.get("method") == "zscore":
                return (
                    "### Outliers (Z-score)"
                    + _mk_table({
                        "Limite Z": result.get("z_threshold"),
                        "Qtde outliers": result.get("count"),
                        "Proporção": f"{result.get('proportion', 0):.6f}",
                        "Média": f"{result.get('mean', 0):.4f}",
                        "Desvio-padrão": f"{result.get('std', 0):.4f}",
                        "Mín": result.get("min"),
                        "Máx": result.get("max"),
                    })
                )
            else:
                return (
                    "### Outliers (IQR)"
                    + _mk_table({
                        "IQR": f"{result.get('iqr', 0):.4f}",
                        "Limite inferior": result.get("lower"),
                        "Limite superior": result.get("upper"),
                        "Qtde outliers": result.get("count"),
                        "Proporção": f"{result.get('proportion', 0):.6f}",
                        "Mín": result.get("min"),
                        "Máx": result.get("max"),
                    })
                )
        if tool == "mode":
            modes = result.get("mode", [])
            counts = result.get("counts", [])
            lines = [f"### Valores mais frequentes em `{result.get('column')}`", "| Valor | Contagem |", "|---|---:|"]
            for v, c in zip(modes, counts):
                lines.append(f"| {v} | {c} |")
            return "".join(lines)
        if tool == "cluster":
            return (
                "### Agrupamentos (KMeans)"
                + _mk_table({
                    "k (nº clusters)": result.get("k"),
                    "Amostra usada": result.get("n_rows"),
                    "Inércia": result.get("inertia"),
                    "Tamanhos dos clusters": result.get("cluster_sizes"),
                })
                + "*Centroides e colunas estão incluídos no payload interno.*"
            )
        if tool == "feature_importance":
            lines = ["### Importância das Variáveis (árvore de decisão)", "| Variável | Importância |", "|---|---:|"]
            for feat, imp in result.get("top_importances", [])[:15]:
                lines.append(f"| {feat} | {imp:.4f} |")
            auc = result.get("auc")
            if auc is not None:
                lines.append(f"*AUC (validação):* **{auc:.3f}**")
            return "".join(lines)
        # Caso geral: renderiza JSON de forma legível
        return "### Resultado```json" + json.dumps(result, indent=2, ensure_ascii=False) + "```"
    except Exception:
        # Texto normal da LLM
        return answer


class AppState:
    def __init__(self):
        # Modelo fixo por solicitação: gpt-4o-mini @ 0.4
        self.model_name = "gpt-4o-mini"
        self.temperature = 0.4
        self.api_key: Optional[str] = None
        self.repo: Optional[DataRepo] = None
        self.llm: Optional[LLM] = None

    def _ensure_llm(self):
        if self.llm is None:
            self.llm = LLM(model=self.model_name, temperature=self.temperature, api_key=self.api_key)

    def get_numeric_columns(self) -> List[str]:
        if self.repo is None:
            return []
        import numpy as np
        return [c for c in self.repo.df.columns if np.issubdtype(self.repo.df[c].dtype, np.number)]

    def load_csv(self, file_obj):
        if file_obj is None:
            empty = gr.update(choices=[], value=None)
            return ("Nenhum arquivo enviado.", empty, empty, empty, empty, empty)
        path = Path(file_obj.name)
        self.repo = DataRepo.load(str(path))
        self.llm = None
        rows, cols = self.repo.df.shape
        num_cols = self.get_numeric_columns()
        first = num_cols[0] if num_cols else None
        dd_update = gr.update(choices=num_cols, value=first, interactive=True)
        ms_update = gr.update(choices=num_cols, value=None, interactive=True)
        status = f"CSV carregado: {path.name} — Linhas: {rows}, Colunas: {cols}"
        return status, dd_update, dd_update, dd_update, dd_update, ms_update

    def ask(self, question: str):
        if not question.strip():
            return ("Digite uma pergunta.", [])
        if self.repo is None:
            return ("Envie um CSV primeiro.", [])
        if not self.api_key:
            return ("Informe sua OpenAI API Key na seção de autenticação.", [])
        self._ensure_llm()
        raw = run_agent(self.llm, self.repo, question)
        return (_prettify(raw), _extract_images_from_payload(raw))

    def _tool_call(self, name: str, args: Dict[str, Any]):
        if self.repo is None:
            return ("Envie um CSV primeiro.", [])
        fn = TOOLS.get(name)
        if not fn:
            return (f"Ferramenta '{name}' não encontrada.", [])
        try:
            result = fn(self.repo, args)
            txt = json.dumps({"tool": name, "result": result}, ensure_ascii=False, indent=2)
            return (_prettify(txt), _extract_images_from_payload(txt))
        except Exception as e:
            return (f"Erro em {name}: {e}", [])

    # Ações rápidas
    def run_correlation(self, method: str, sample: int):
        return self._tool_call("correlation", {"method": method, "sample": sample})

    def run_feature_importance(self, target: str, test_size: float):
        return self._tool_call("feature_importance", {"target": target, "test_size": test_size})

    def run_outliers(self, column: str, method: str, z: float, iqr_factor: float):
        return self._tool_call("outliers", {"column": column, "method": method, "z": z, "iqr_factor": iqr_factor})

    def run_scatter(self, x: str, y: str, sample: int):
        return self._tool_call("scatterplot", {"x": x, "y": y, "sample": sample})

    def run_mode(self, column: str, top: int):
        return self._tool_call("mode", {"column": column, "top": top})

    def run_cluster(self, cols: Optional[List[str]], k: int, sample: int):
        payload = {"columns": cols or None, "k": k, "sample": sample}
        return self._tool_call("cluster", payload)


def build_app():
    state = AppState()

    with gr.Blocks(title="Agente IA – Cartão de Crédito") as demo:
        # Header
        gr.Markdown("""
        # Agente IA – Análise de Transações de Cartão
        **Modelo:** gpt-4o-mini · **Temperatura:** 0.4
        """)

        # Linha com Chave + CSV
        with gr.Row():
            api_key_tb = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...", lines=1)
            csv_in = gr.File(label="CSV de transações", file_types=[".csv"], interactive=True)
            apply_btn = gr.Button("Aplicar & Carregar", variant="primary")
        status = gr.Markdown(visible=True)

        # Layout principal com menu lateral recolhível + área Q&A
        with gr.Row():
            # Coluna da esquerda: menu suspenso (toggle)
            with gr.Column(scale=1, min_width=140) as sidebar_col:
                toggle = gr.Button("☰ Análises Rápidas", size="sm")
                quick_panel = gr.Group(visible=False)
                with quick_panel:
                    gr.Markdown("### Análises Rápidas")
                    with gr.Tab("Correlação"):
                        corr_method = gr.Dropdown(["pearson","spearman"], value="pearson", label="Método")
                        corr_sample = gr.Slider(1000, 150000, value=50000, step=1000, label="Amostra (linhas)")
                        corr_btn = gr.Button("Gerar correlação")
                    with gr.Tab("Importância"):
                        fi_target = gr.Textbox(value="Class", label="Alvo (target)")
                        fi_test = gr.Slider(0.1, 0.5, value=0.25, step=0.05, label="Test size")
                        fi_btn = gr.Button("Calcular importância")
                    with gr.Tab("Outliers"):
                        out_col = gr.Dropdown(choices=[], label="Coluna numérica")
                        out_method = gr.Radio(["iqr","zscore"], value="iqr", label="Método")
                        out_z = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Z (zscore)")
                        out_iqr = gr.Slider(0.5, 5.0, value=1.5, step=0.1, label="Fator IQR")
                        out_btn = gr.Button("Detectar outliers")
                    with gr.Tab("Dispersão"):
                        sc_x = gr.Dropdown(choices=[], label="X")
                        sc_y = gr.Dropdown(choices=[], label="Y")
                        sc_sample = gr.Slider(1000, 100000, value=20000, step=1000, label="Amostra (linhas)")
                        sc_btn = gr.Button("Gerar dispersão")
                    with gr.Tab("Moda"):
                        mode_col = gr.Dropdown(choices=[], label="Coluna")
                        mode_top = gr.Slider(1, 50, value=5, step=1, label="Top N")
                        mode_btn = gr.Button("Calcular moda")
                    with gr.Tab("Clusters"):
                        cl_cols = gr.Dropdown(choices=[], multiselect=True, label="Colunas (opcional)")
                        cl_k = gr.Slider(2, 20, value=2, step=1, label="K (nº clusters)")
                        cl_sample = gr.Slider(1000, 100000, value=20000, step=1000, label="Amostra (linhas)")
                        cl_btn = gr.Button("Rodar KMeans")

            # Coluna da direita: Q&A em destaque com respostas lado a lado
            with gr.Column(scale=5):
                question = gr.Textbox(label="Pergunta", placeholder="Ex.: Qual a proporção de fraudes?", lines=3)
                ask_btn = gr.Button("Perguntar", variant="primary")
                with gr.Row():
                    answer = gr.Markdown(label="Resposta (LLM)", height=380)
                    gallery = gr.Gallery(label="Figuras geradas", show_label=True, columns=1, height=380)

        # Handlers
        def apply_and_load(key: str, file_obj):
            state.api_key = key.strip() if key else None
            state.llm = None
            if file_obj is None:
                return "Envie um CSV para continuar.", gr.update(visible=False), None, None, None, None
            s, dd1, dd2, dd3, dd4, ms = state.load_csv(file_obj)
            return s, gr.update(visible=False), dd1, dd2, dd3, dd4, ms

        def toggle_sidebar(vis: bool):
            return gr.update(visible=not vis)

        apply_btn.click(
            fn=apply_and_load,
            inputs=[api_key_tb, csv_in],
            outputs=[status, quick_panel, out_col, sc_x, sc_y, mode_col, cl_cols]
        )
        toggle.click(fn=toggle_sidebar, inputs=quick_panel, outputs=quick_panel)

        ask_btn.click(fn=state.ask, inputs=question, outputs=[answer, gallery])

        # Botões rápidos
        corr_btn.click(fn=state.run_correlation, inputs=[corr_method, corr_sample], outputs=[answer, gallery])
        fi_btn.click(fn=state.run_feature_importance, inputs=[fi_target, fi_test], outputs=[answer, gallery])
        out_btn.click(fn=state.run_outliers, inputs=[out_col, out_method, out_z, out_iqr], outputs=[answer, gallery])
        sc_btn.click(fn=state.run_scatter, inputs=[sc_x, sc_y, sc_sample], outputs=[answer, gallery])
        mode_btn.click(fn=state.run_mode, inputs=[mode_col, mode_top], outputs=[answer, gallery])
        cl_btn.click(fn=state.run_cluster, inputs=[cl_cols, cl_k, cl_sample], outputs=[answer, gallery])

    return demo