"""
query_names と base_names をまとめてベクトル化

"""

import numpy as np
from typing import List, Tuple
from tqdm import tqdm  # 進捗バーを表示するライブラリ
from sklearn.feature_extraction.text import TfidfVectorizer


def get_vectors(query_names: List[str], base_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:

    print(f"[INFO] ベクトル化する食品名数（合計）: {len(base_names)} base + {len(query_names)} query")
    
    all_names = base_names + query_names
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3)) # fit_transform() は まとめて処理したほうが精度が安定する（語彙の一致やスケーリングが一貫する）

    tfidf_matrix = vectorizer.fit_transform(all_names)
    tfidf_array = tfidf_matrix.toarray()

    base_vecs = []
    query_vecs = []

    for i in tqdm(range(len(all_names)), desc="ベクトル分離", unit="件"): # リストの最初から最後まで（0番目からlen-1番目まで）」を順番に処理
        if i < len(base_names):
            base_vecs.append(tfidf_array[i])
        else:
            query_vecs.append(tfidf_array[i])

    return np.array(query_vecs), np.array(base_vecs)