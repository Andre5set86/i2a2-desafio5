import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

# Imports com fallback (pacote ou scripts soltos)
try:
    from .data_repo import DataRepo
except ImportError:
    from data_repo import DataRepo

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

class ToolError(Exception):
    pass

# ---------------- Schemas de entrada ----------------
class HistInput(BaseModel):
    column: str
    bins: int = Field(50, ge=5, le=200)

class SummaryInput(BaseModel):
    by_class: bool = False
    columns: Optional[List[str]] = None

class MinMaxInput(BaseModel):
    columns: Optional[List[str]] = None

class FraudRatioInput(BaseModel):
    pass

class TimePatternInput(BaseModel):
    freq: str = Field("H")  # 'H','D',...

class TopNInput(BaseModel):
    column: str = "Amount"
    n: int = Field(10, ge=1, le=100)
    desc: bool = True

class ModeInput(BaseModel):
    column: str
    top: int = Field(5, ge=1, le=50)

class ClusterInput(BaseModel):
    columns: Optional[List[str]] = None
    k: int = Field(2, ge=2, le=20)
    sample: Optional[int] = Field(20000, ge=1000, le=100000)
    random_state: int = 42

class OutliersInput(BaseModel):
    column: str
    method: str = Field("iqr", pattern="^(iqr|zscore)$")
    z: float = Field(3.0, ge=1.0, le=10.0)
    iqr_factor: float = Field(1.5, ge=0.5, le=5.0)

class ScatterInput(BaseModel):
    x: str
    y: str
    sample: Optional[int] = Field(20000, ge=1000, le=100000)

class CorrelationInput(BaseModel):
    columns: Optional[List[str]] = None
    method: str = Field("pearson", pattern="^(pearson|spearman)$")
    sample: Optional[int] = Field(50000, ge=1000, le=150000)

class FeatureImportanceInput(BaseModel):
    target: str = Field("Class")
    test_size: float = Field(0.25, ge=0.1, le=0.5)
    max_features: Optional[int] = Field(None, ge=1, le=100)
    random_state: int = 42

# ---------------- Helpers ----------------

def _ensure_numeric(df: pd.DataFrame, col: str):
    if col not in df.columns:
        raise ToolError(f"Coluna '{col}' inexistente.")
    if not np.issubdtype(df[col].dtype, np.number):
        raise ToolError(f"Coluna '{col}' não é numérica.")


def _select_numeric(repo: DataRepo, include: Optional[List[str]] = None) -> pd.DataFrame:
    if include:
        cols = [c for c in include if c in repo.df.columns and np.issubdtype(repo.df[c].dtype, np.number)]
    else:
        cols = [c for c in repo.df.columns if np.issubdtype(repo.df[c].dtype, np.number) and c != "Class"]
    if not cols:
        raise ToolError("Nenhuma coluna numérica válida encontrada.")
    return repo.df[cols].dropna()

# ---------------- Ferramentas ----------------

def tool_list_columns(repo: DataRepo, args: Dict[str, Any] = None) -> Dict[str, Any]:
    dtypes = {c: str(repo.df[c].dtype) for c in repo.df.columns}
    return {"columns": list(repo.df.columns), "dtypes": dtypes}


def tool_histogram(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = HistInput(**args)
    _ensure_numeric(repo.df, payload.column)
    series = repo.df[payload.column].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series, bins=payload.bins, edgecolor="black")
    ax.set_title(f"Histograma de {payload.column}")
    ax.set_xlabel(payload.column)
    ax.set_ylabel("Frequência")
    ax.grid(alpha=0.3)
    fname = OUTPUT_DIR / f"hist_{payload.column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return {"image_path": str(fname), "count": int(series.shape[0])}


def tool_summary(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = SummaryInput(**args)
    num_cols = [c for c in repo.df.columns if np.issubdtype(repo.df[c].dtype, np.number)]
    cols = payload.columns or num_cols
    subset = repo.df[cols]
    if payload.by_class and "Class" in repo.df.columns:
        out = subset.groupby(repo.df["Class"]).agg(["count", "mean", "std", "var", "min", "median", "max"]).to_dict()
        return {"by_class": True, "stats": out}
    else:
        out = subset.agg(["count", "mean", "std", "var", "min", "median", "max"]).to_dict()
        return {"by_class": False, "stats": out}


def tool_minmax(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = MinMaxInput(**args)
    cols = payload.columns or list(repo.df.columns)
    result = {}
    for c in cols:
        if np.issubdtype(repo.df[c].dtype, np.number):
            result[c] = {"min": float(np.nanmin(repo.df[c])), "max": float(np.nanmax(repo.df[c]))}
    return {"minmax": result}


def tool_fraud_ratio(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    if "Class" not in repo.df.columns:
        raise ToolError("Coluna 'Class' ausente.")
    total = int(repo.df.shape[0])
    frauds = int((repo.df["Class"] == 1).sum())
    legit = total - frauds
    ratio = frauds / total if total else 0.0
    return {"total": total, "legit": legit, "frauds": frauds, "fraud_ratio": ratio}


def tool_time_patterns(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = TimePatternInput(**args)
    if "Time" not in repo.df.columns:
        raise ToolError("Coluna 'Time' ausente.")
    base = pd.to_datetime(0, unit="s")
    ts = base + pd.to_timedelta(repo.df["Time"], unit="s")
    s = pd.Series(1, index=ts)
    agg = s.resample(payload.freq).sum().fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(agg.index, agg.values)
    ax.set_title(f"Contagem de transações por período ({payload.freq})")
    ax.set_ylabel("# transações")
    ax.set_xlabel("tempo relativo")
    ax.grid(alpha=0.3)
    fname = OUTPUT_DIR / f"time_count_{payload.freq}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return {"image_path": str(fname), "points": int(len(agg))}


def tool_topn(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = TopNInput(**args)
    _ensure_numeric(repo.df, payload.column)
    df_sorted = repo.df.sort_values(payload.column, ascending=not payload.desc).head(payload.n)
    return {"rows": df_sorted.to_dict(orient="records")}

# ----- Extras (modo, cluster, outliers, scatter, correlation, feature importance)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


def tool_mode(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = ModeInput(**args)
    col = payload.column
    if col not in repo.df.columns:
        raise ToolError(f"Coluna '{col}' inexistente.")
    series = repo.df[col].dropna()
    vc = series.value_counts().head(payload.top)
    return {"column": col, "mode": vc.index.tolist(), "counts": vc.values.tolist()}


def tool_cluster(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = ClusterInput(**args)
    X = _select_numeric(repo, payload.columns)
    if payload.sample and len(X) > payload.sample:
        X = X.sample(payload.sample, random_state=payload.random_state)
    km = KMeans(n_clusters=payload.k, random_state=payload.random_state, n_init=10)
    labels = km.fit_predict(X.values)
    sizes = pd.Series(labels).value_counts().sort_index().tolist()
    centers = km.cluster_centers_.tolist()
    return {
        "k": payload.k,
        "n_rows": int(len(X)),
        "cluster_sizes": sizes,
        "centers": centers,
        "inertia": float(km.inertia_),
        "columns": X.columns.tolist(),
    }


def tool_outliers(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = OutliersInput(**args)
    col = payload.column
    _ensure_numeric(repo.df, col)
    s = repo.df[col].dropna()
    n = int(s.shape[0])
    if payload.method == "zscore":
        mean, std = float(s.mean()), float(s.std(ddof=1))
        if std == 0:
            count = 0
        else:
            z = (s - mean) / std
            count = int((z.abs() > payload.z).sum())
        return {
            "method": "zscore",
            "z_threshold": payload.z,
            "count": count,
            "proportion": (count / n) if n else 0.0,
            "mean": mean,
            "std": std,
            "min": float(s.min()),
            "max": float(s.max()),
        }
    else:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = float(q3 - q1)
        lower = float(q1 - payload.iqr_factor * iqr)
        upper = float(q3 + payload.iqr_factor * iqr)
        mask = (s < lower) | (s > upper)
        count = int(mask.sum())
        return {
            "method": "iqr",
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
            "count": count,
            "proportion": (count / n) if n else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
        }


def tool_scatterplot(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = ScatterInput(**args)
    for c in (payload.x, payload.y):
        _ensure_numeric(repo.df, c)
    df2 = repo.df[[payload.x, payload.y]].dropna()
    if payload.sample and len(df2) > payload.sample:
        df2 = df2.sample(payload.sample, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df2[payload.x], df2[payload.y], s=6, alpha=0.5)
    ax.set_xlabel(payload.x)
    ax.set_ylabel(payload.y)
    ax.set_title(f"Dispersão: {payload.x} vs {payload.y}")
    ax.grid(alpha=0.3)
    fname = OUTPUT_DIR / f"scatter_{payload.x}_vs_{payload.y}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return {"image_path": str(fname), "points": int(len(df2))}


def tool_correlation(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = CorrelationInput(**args)
    X = _select_numeric(repo, payload.columns)
    if payload.sample and len(X) > payload.sample:
        X = X.sample(payload.sample, random_state=42)
    corr = X.corr(method=payload.method)
    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.imshow(corr.values, interpolation='nearest', aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(cax)
    ax.set_title(f"Correlação ({payload.method})")
    fig.tight_layout()
    fname = OUTPUT_DIR / f"corr_{payload.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fname)
    plt.close(fig)
    corr_abs = corr.abs()
    mask = np.triu(np.ones(corr_abs.shape), k=1).astype(bool)
    vals = corr_abs.where(mask).stack().sort_values(ascending=False)
    top_pairs = [(str(i), str(j), float(v)) for (i, j), v in vals.head(10).items()]
    return {"image_path": str(fname), "top_pairs": top_pairs}


def tool_feature_importance(repo: DataRepo, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = FeatureImportanceInput(**args)
    if payload.target not in repo.df.columns:
        raise ToolError(f"Coluna alvo '{payload.target}' não encontrada.")
    X = _select_numeric(repo)
    y = repo.df[payload.target]
    df2 = pd.concat([X, y], axis=1).dropna()
    y = df2[payload.target]
    X = df2.drop(columns=[payload.target])
    if payload.max_features and payload.max_features < X.shape[1]:
        var = X.var().sort_values(ascending=False)
        keep = var.head(payload.max_features).index
        X = X[keep]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=payload.test_size, random_state=payload.random_state,
        stratify=y if y.nunique() <= 20 else None
    )
    clf = DecisionTreeClassifier(random_state=payload.random_state, class_weight="balanced")
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    feats = X.columns[order].tolist()
    imps = importances[order].tolist()
    try:
        proba = clf.predict_proba(X_test)[:, -1]
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = None
    top = list(zip(feats[:20], [float(v) for v in imps[:20]]))
    return {"n_features": int(X.shape[1]), "top_importances": top, "auc": auc}

# ---------------- Registro de Ferramentas ----------------
TOOLS: Dict[str, Any] = {
    "list_columns": tool_list_columns,
    "histogram": tool_histogram,
    "summary": tool_summary,
    "minmax": tool_minmax,
    "fraud_ratio": tool_fraud_ratio,
    "time_patterns": tool_time_patterns,
    "topn": tool_topn,
    "mode": tool_mode,
    "cluster": tool_cluster,
    "outliers": tool_outliers,
    "scatterplot": tool_scatterplot,
    "correlation": tool_correlation,
    "feature_importance": tool_feature_importance,
}