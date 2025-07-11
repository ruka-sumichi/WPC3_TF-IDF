"""
ベクトル同士の類似度評価

"""

import numpy as np
from tqdm import tqdm

def find_top3_matches(
    query_names: list[str],
    query_codes: list[int],
    query_embeddings: np.ndarray,
    base_names: list[str],
    base_codes: list[int],
    base_embeddings: np.ndarray,
    top_k_final: int = 3,
) -> list[dict]:
       
    # 単位ベクトル化（ゼロ割防止に1e-10を加える）
    query_norms = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-10)
    base_norms = base_embeddings / (np.linalg.norm(base_embeddings, axis=1, keepdims=True) + 1e-10)

    # コサイン類似度行列（クエリ × ベース）
    sim_matrix = np.dot(query_norms, base_norms.T)  # shape = (Q, B)

    results = []
    for i in tqdm(range(sim_matrix.shape[0]), desc="類似検索", unit="件"): # sim_matrix.shape[0] はクエリの数。
        sims = sim_matrix[i]
        top_indices = np.argsort(sims)[::-1][:top_k_final] # 類似度が高い順に並べて、上位 top_k_final 件のインデックスを取得。[::-1] で降順.

        result = {
            "query_code": query_codes[i],
            "query_name": query_names[i]
        }

        for rank, idx in enumerate(top_indices, 1):
            result[f"matched_code_{rank}"] = base_codes[idx]
            result[f"matched_name_{rank}"] = base_names[idx]
            result[f"score_{rank}"] = round(float(sims[idx]), 3)

        results.append(result)

    return results