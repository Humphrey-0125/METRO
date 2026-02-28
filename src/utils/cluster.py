import json
import math
import numpy as np
from pathlib import Path
from sklearn.cluster import OPTICS
from typing import List, Tuple, Dict, Any, Iterable, Optional
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# =========================
# OPTICS
# =========================
def optics_fit_predict(
    X,
    min_samples: int = 5,
    xi: float = 0.05,
    min_cluster_size: int = 5,
    metric: str = "cosine",
):
    model = OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        metric=metric,
    )
    labels = model.fit_predict(X)
    return model, labels


def optics_compute_centroids(X: np.ndarray, labels: np.ndarray):
    unique_labels = np.unique(labels)
    cluster_labels = [l for l in unique_labels if l != -1]

    if len(cluster_labels) == 0:
        new_labels = np.full_like(labels, -1)
        return new_labels, None

    cluster_labels_sorted = sorted(cluster_labels)
    label_map = {old: idx for idx, old in enumerate(cluster_labels_sorted)}

    new_labels = np.full_like(labels, -1)
    centroids_list = []

    for old_label, new_label in label_map.items():
        mask = (labels == old_label)
        if not np.any(mask):
            continue
        X_cluster = X[mask]
        centroid = X_cluster.mean(axis=0)
        centroids_list.append(centroid)
        new_labels[mask] = new_label

    centroids = np.stack(centroids_list, axis=0) if len(centroids_list) > 0 else None
    return new_labels, centroids


# =========================
# HDBSCAN (NEW)
# =========================
def hdbscan_fit_predict(
    X: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
):
    """
    HDBSCAN 密度聚类，返回 (model, labels)。

    强烈建议：如果你想用 “cosine” 语义距离，
    先对 embedding 做 L2 normalize，然后 metric 用 "euclidean"（更稳、更快）：
      cosine(u,v) 与 ||u-v|| 在单位球上单调等价。

    参数：
    - min_cluster_size: 最小簇大小（越大越保守，簇更少、更纯）
    - min_samples: 核心点邻域阈值（越大越保守、噪声更多）。None 表示等于 min_cluster_size 的默认行为（hdbscan 内部逻辑）
    - metric: "euclidean"（推荐配合 normalize）或 "cosine"
    - cluster_selection_method: "eom"(常用) 或 "leaf"(更细碎)
    """
    try:
        import hdbscan
    except ImportError as e:
        raise ImportError(
            "未安装 hdbscan。请先执行：pip install hdbscan"
        ) from e

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
    )
    labels = model.fit_predict(X)
    return model, labels


def hdbscan_compute_centroids(X: np.ndarray, labels: np.ndarray):
    """
    对 HDBSCAN 的 labels 计算质心，并把簇标签重映射为 0..n_clusters-1
    噪声点保持为 -1。
    """
    unique_labels = np.unique(labels)
    cluster_labels = [l for l in unique_labels if l != -1]

    if len(cluster_labels) == 0:
        new_labels = np.full_like(labels, -1)
        return new_labels, None

    cluster_labels_sorted = sorted(cluster_labels)
    label_map = {old: idx for idx, old in enumerate(cluster_labels_sorted)}

    new_labels = np.full_like(labels, -1)
    centroids_list = []

    for old_label, new_label in label_map.items():
        mask = (labels == old_label)
        if not np.any(mask):
            continue
        X_cluster = X[mask]
        centroid = X_cluster.mean(axis=0)
        centroids_list.append(centroid)
        new_labels[mask] = new_label

    centroids = np.stack(centroids_list, axis=0) if len(centroids_list) > 0 else None
    return new_labels, centroids


# =========================
# Utils
# =========================
def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file is not an array.")
    return data


def load_ids_and_embeddings_auto(path: str) -> Tuple[List[str], np.ndarray]:
    p = Path(path)
    ids: List[str] = []
    embs: List[List[float]] = []

    def push(obj: Dict[str, Any]):
        rid = obj.get("id")
        if not rid:
            dlg = obj.get("dialog_id")
            ses = obj.get("session_id")
            stx = obj.get("state_index")
            if dlg is not None and ses is not None and stx is not None:
                rid = f"{dlg}:{ses}:{stx}"
        if not rid:
            return

        emb = obj.get("embedding")
        if emb is None:
            emb = obj.get("embed")
        if emb is None:
            emb = obj.get("history_embedding")
        if not isinstance(emb, (list, tuple)) or len(emb) == 0:
            return

        ids.append(str(rid))
        embs.append(list(emb))

    if p.suffix.lower() == ".jsonl":
        for obj in _iter_jsonl(p):
            push(obj)
    else:
        data = _load_json_array(p)
        for obj in data:
            if isinstance(obj, dict):
                push(obj)

    if not ids:
        raise ValueError("未找到任何有效的 id/embedding。")
    X = np.asarray(embs, dtype=np.float32)
    return ids, X


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def kmeans_fit_predict(
    X,
    k: int,
    minibatch: bool = True,
    random_state: int = 0,
    batch_size: int = 2048,
    use_gpu: bool = False,
):
    if use_gpu:
        try:
            import cupy as cp
            from cuml.cluster import KMeans as cuKMeans
            from cuml.cluster import MiniBatchKMeans as cuMiniBatchKMeans
            gpu_available = True
        except ImportError:
            print("[WARN] cuML or CuPy not available, falling back to CPU KMeans.")
            gpu_available = False
    else:
        gpu_available = False

    if gpu_available:
        print(f"[GPU] Using cuML {'MiniBatchKMeans' if minibatch else 'KMeans'} with k={k}")
        X_gpu = cp.asarray(X)

        if minibatch:
            model_gpu = cuMiniBatchKMeans(
                n_clusters=k,
                random_state=random_state,
                batch_size=batch_size,
            )
        else:
            model_gpu = cuKMeans(
                n_clusters=k,
                random_state=random_state,
            )

        model_gpu.fit(X_gpu)
        labels_gpu = model_gpu.predict(X_gpu)
        labels = cp.asnumpy(labels_gpu)
        centers_gpu = model_gpu.cluster_centers_
        centroids = cp.asnumpy(centers_gpu)

        class _DummyModel:
            pass

        model = _DummyModel()
        model.cluster_centers_ = centroids
        return model, labels

    print(f"[CPU] Using sklearn {'MiniBatchKMeans' if minibatch else 'KMeans'} with k={k}")
    if minibatch:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,
        )
    else:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
        )
    labels = model.fit_predict(X)
    return model, labels


def choose_k_by_silhouette(
    X: np.ndarray, k_min: int = 5, k_max: int = 50, step: int = 5, metric: str = "cosine", minibatch: bool = True
) -> Dict[str, Any]:
    scores: Dict[int, float] = {}
    best_k, best_s = None, -1.0

    for k in tqdm(range(k_min, k_max + 1, step), desc="Grid(k) for silhouette"):
        try:
            model, labels = kmeans_fit_predict(X, k, minibatch=minibatch)
            s = silhouette_score(X, labels, metric=metric)
        except Exception:
            s = float("nan")
        scores[k] = s
        val = np.nan_to_num(s, nan=-1.0)
        if best_k is None or val > best_s:
            best_k, best_s = k, val

    return {"k": best_k, "silhouette": scores.get(best_k, float("nan")), "scores": scores}


def labels_to_groups(ids: List[str], labels: np.ndarray) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for rid, lab in zip(ids, labels):
        key = str(int(lab))
        groups.setdefault(key, []).append(rid)
    return groups


def export_clusters(
    out_dir: str,
    ids: List[str],
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
):
    """
    导出：
    - clusters.json：每个簇成员 + 质心（可选）+ meta
    - labels.jsonl：逐条 id -> cluster

    兼容：
    - KMeans：labels = 0..k-1
    - OPTICS/HDBSCAN：labels = -1(噪声) 或 >=0
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    groups = labels_to_groups(ids, labels)

    # n_clusters：只统计非噪声簇
    unique = set(int(x) for x in np.unique(labels).tolist()) if len(labels) else set()
    non_noise = sorted([x for x in unique if x != -1])
    n_clusters = int(centroids.shape[0]) if centroids is not None else len(non_noise)

    clusters_payload = {
        "n_items": len(ids),
        "n_clusters": int(n_clusters),
        "clusters": [],
        "meta": meta or {},
    }

    for cid_str, members in groups.items():
        cid = int(cid_str)
        entry = {"cluster": cid, "size": len(members), "members": members}

        # 只有非噪声簇且提供了 centroids 才写 centroid
        if centroids is not None and cid != -1:
            if 0 <= cid < centroids.shape[0]:
                entry["centroid"] = centroids[cid].tolist()

        clusters_payload["clusters"].append(entry)

    with (out / "clusters.json").open("w", encoding="utf-8") as f:
        json.dump(numpy_to_python(clusters_payload), f, ensure_ascii=False, indent=2)

    with (out / "labels.jsonl").open("w", encoding="utf-8") as f:
        for rid, lab in zip(ids, labels):
            f.write(json.dumps({"id": rid, "cluster": int(lab)}, ensure_ascii=False) + "\n")
