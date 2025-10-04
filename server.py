from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import shutil
import json
from typing import Optional, Dict, Any

from agent_lc import run_lc_agent, get_memory, clear_memory

from data_repo import DataRepo
from agent import LLM, run_agent
from tools import TOOLS

app = FastAPI(title="Credit Card Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Diretórios ---
UPLOADS_DIR = Path("uploads"); UPLOADS_DIR.mkdir(exist_ok=True)
IMAGES_DIR = Path("outputs"); IMAGES_DIR.mkdir(exist_ok=True)

# Servir a pasta outputs/ em /images
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

STATE: Dict[str, Any] = {
    "api_key": None,
    "repo": None,
    "model": "gpt-4o-mini",
    "temperature": 0.4,
}

class AskPayload(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class QuickPayload(BaseModel):
    params: Optional[Dict[str, Any]] = None

# ---------- helpers ----------
def _public_image_url(path_like: Optional[str]) -> Optional[str]:
    """
    Converte um caminho local (ex.: outputs/corr_...png) em URL pública (/images/arquivo.png).
    Se vier None ou vazio, retorna None.
    """
    if not path_like:
        return None
    name = Path(path_like).name
    return f"/images/{name}"

def _normalize_result_images(obj: Any) -> Any:
    """
    Percorre dict aninhado e, se encontrar a chave 'image_path',
    adiciona 'image_url' e transforma 'image_path' em apenas o filename.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "image_path":
                url = _public_image_url(v)
                new_obj["image_path"] = Path(v).name if v else v
                if url:
                    new_obj["image_url"] = url
            else:
                new_obj[k] = _normalize_result_images(v)
        return new_obj
    elif isinstance(obj, list):
        return [_normalize_result_images(v) for v in obj]
    else:
        return obj

# ---------- rotas ----------

@app.post("/ask_lc")
def ask_langchain(payload: AskPayload):
    if not STATE.get("repo"):
        return JSONResponse(status_code=400, content={"error": "CSV não carregado"})
    if not STATE.get("api_key"):
        return JSONResponse(status_code=400, content={"error": "API key não definida"})
    sid = payload.session_id or "default"
    result = run_lc_agent(api_key=STATE["api_key"], repo=STATE["repo"], question=payload.question, session_id=sid)
    return {"answer": result.get("answer", ""), "images": result.get("images", []), "session_id": sid}

@app.post("/memory/clear")
def memory_clear(session_id: Optional[str] = Form("default")):
    clear_memory(session_id or "default")
    return {"ok": True, "session_id": session_id or "default"}

@app.get("/memory/debug/{session_id}")
def memory_debug(session_id: Optional[str] = "default"):
    mem = get_memory(session_id or "default")
    # cuidado: objetos Message não são JSON; sumarize:
    msgs = mem.chat_memory.messages if hasattr(mem, "chat_memory") else []
    return {"session_id": session_id or "default", "messages": len(msgs)}


@app.post("/set_key")
def set_key(api_key: str = Form(...)):
    STATE["api_key"] = api_key.strip()
    return {"ok": True}
import json

@app.post("/upload_csv")
def upload_csv(file: UploadFile = File(...)):
    try:
        dst = UPLOADS_DIR / file.filename
        with dst.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        repo = DataRepo.load(str(dst))
        STATE["repo"] = repo

        rows, cols = repo.df.shape

        import numpy as np
        num_cols = [c for c in repo.df.columns if np.issubdtype(repo.df[c].dtype, np.number)]

        # >>> AMOSTRA (até 10 linhas) com tipos JSON nativos
        sample = json.loads(repo.df.head(10).to_json(orient="records"))

        return {
            "ok": True,
            "rows": rows,
            "cols": cols,
            "numeric_columns": num_cols,
            "sample": sample,            # <<< adicionado
            "columns": list(repo.df.columns),  # útil no front
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})

@app.post("/ask")
def ask(payload: AskPayload):
    if not STATE.get("repo"):
        return JSONResponse(status_code=400, content={"error": "CSV não carregado"})
    if not STATE.get("api_key"):
        return JSONResponse(status_code=400, content={"error": "API key não definida"})
    llm = LLM(model=STATE["model"], temperature=STATE["temperature"], api_key=STATE["api_key"])
    raw = run_agent(llm, STATE["repo"], payload.question)

    # Se a LLM retornou JSON com result.image_path, normaliza (adiciona image_url)
    try:
        obj = json.loads(raw)
        obj_norm = _normalize_result_images(obj)
        return {"raw": json.dumps(obj_norm, ensure_ascii=False)}
    except Exception:
        # Não era JSON -> devolve como texto cru
        return {"raw": raw}

@app.post("/quick/{tool}")
def quick(tool: str, payload: QuickPayload):
    if tool not in TOOLS:
        return JSONResponse(status_code=400, content={"error": f"Ferramenta '{tool}' não encontrada"})
    if not STATE.get("repo"):
        return JSONResponse(status_code=400, content={"error": "CSV não carregado"})
    try:
        params = payload.params or {}
        result = TOOLS[tool](STATE["repo"], params)
        result = _normalize_result_images(result)
        return {"tool": tool, "result": result}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/images/list")
def list_images():
    """
    Lista arquivos já existentes em outputs/ para debug/consulta.
    """
    files = sorted([p.name for p in IMAGES_DIR.glob("*") if p.is_file()])
    return {"count": len(files), "files": files, "base": "/images/"}

@app.get("/health")
def health():
    return {"ok": True}
