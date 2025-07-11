import pandas as pd
from preprocess import normalize_all_names
from vectorizer import get_vectors
from matcher import find_top3_matches
from output_writer import write_output

import pandas as pd
from preprocess import normalize_all_names
from vectorizer import get_vectors
from matcher import find_top3_matches
from output_writer import write_output
from tqdm import tqdm

def main():
    # 🔁 ファイルパス（A = 比較対象, B = クエリ施設）
    base_path = r"C:\Sumichika\WPC__AImenu\data\fuzi.csv"    # 施設A（ベース）
    query_path = r"C:\Sumichika\WPC__AImenu\data\nihon.csv"  # 施設B（クエリ）

    # 🔁 データ読み込み
    df_base = pd.read_csv(base_path, encoding="cp932")
    df_query = pd.read_csv(query_path, encoding="cp932")

    # 🔁 クエリ側：料理コードでユニーク化（＝横持ち化）
    df_query_unique = df_query.drop_duplicates("nm051_code").copy()

    # 🔁 正規化（料理コード単位で）
    df_query_unique["normalized_name"] = normalize_all_names(df_query_unique["nm051_dsp_name"].tolist())

    # 🔁 正規化（代表料理名に対してのみ）
    query_names_raw = df_query_unique["nm051_dsp_name"].tolist()
    query_names = df_query_unique["normalized_name"].tolist()
    query_codes = df_query_unique["nm051_code"].tolist()

    # ✅ デバッグ：クエリ側
    print("[DEBUG] クエリ: 全行件数（raw） =", len(df_query))
    print("[DEBUG] クエリ: nm051_code ユニーク件数 =", len(df_query["nm051_code"].unique()))
    print("[DEBUG] クエリ: 正規化後ユニーク件数 =", len(set(query_names)))


    df_base_grouped = df_base.groupby("nm051_code").agg({
        "nm051_dsp_name": "first",
        "nm013_name": lambda x: "、".join(sorted(set(x.dropna())))
    }).reset_index()

    # 🔁 ベース側（施設A）は全行対象
    base_names_raw = df_base_grouped["nm051_dsp_name"].tolist()
    base_names = normalize_all_names(base_names_raw)
    base_codes = df_base_grouped["nm051_code"].tolist()

    base_ingredients_map = df_base_grouped.set_index("nm051_code")["nm013_name"].to_dict()

    print("[INFO] 横持ち後のクエリ件数（B施設）:", len(query_names))
    print("[INFO] 横持ち後のベース件数（A施設）:", len(base_names))

    # 🔁 材料情報（必要なら表示用に）
    query_ingredients_map = df_query.groupby("nm051_code")["nm013_name"] \
        .apply(lambda x: "、".join(sorted(set(x.dropna())))).to_dict()
    base_ingredients_map = df_base.groupby("nm051_code")["nm013_name"] \
        .apply(lambda x: "、".join(sorted(set(x.dropna())))).to_dict()

    # ✅ ベクトル計算（ベースは1回で済む）
    _, base_embeddings = get_vectors(["ダミー"], base_names)

    # 🔁 チャンク処理（例：500件ずつ）
    chunk_size = 500
    all_results = []

    for i in tqdm(range(0, len(query_names), chunk_size), desc="クエリ処理中", unit="chunk"):
        q_names_chunk = query_names[i:i+chunk_size]
        q_names_raw_chunk = query_names_raw[i:i+chunk_size]
        q_codes_chunk = query_codes[i:i+chunk_size]

        q_embeddings, _ = get_vectors(q_names_chunk, base_names)

        results_chunk = find_top3_matches(
            query_names=q_names_raw_chunk,
            query_codes=q_codes_chunk,
            query_embeddings=q_embeddings,
            base_names=base_names_raw,
            base_codes=base_codes,
            base_embeddings=base_embeddings,
            top_k_final=3,
            query_ingredients_map=query_ingredients_map,
            base_ingredients_map=base_ingredients_map
        )

        all_results.extend(results_chunk)

    # ✅ 出力
    output_path = "output/matched_top3.csv"
    write_output(all_results, output_path)

if __name__ == "__main__":
    main()


""" def main():
    # ファイルパス設定
    facility_a_path = r"C:\Sumichika\WPC3_TF-IDF\data\A_comparison.csv"
    base_path = r"C:\Sumichika\WPC3_TF-IDF\data\B_base.csv"

    # データ読み込み
    df_a = pd.read_csv(facility_a_path)
    df_base = pd.read_csv(base_path)

    df_query_unique = df_a.drop_duplicates("food_code")

    query_names_raw = df_query_unique["food_name"].tolist()                   # 加工していない食品名
    query_codes = df_query_unique["food_code"].tolist()    # 整えた食品名
    query_names = normalize_all_names(query_names_raw)                     # コード

    base_names = normalize_all_names(df_base["food_name"].tolist())
    base_codes = df_base["food_code"].tolist()
    base_names_raw = df_base["food_name"].tolist()

    # TF-IDF ベクトル取得（query と base を同時に渡す）
    query_embeddings, base_embeddings = get_vectors(query_names, base_names)

    # 1件ずつ類似マッチ処理
    results = find_top3_matches(
        query_names=query_names_raw, 
        query_codes=query_codes,                   
        query_embeddings=query_embeddings,
        base_names=base_names_raw,
        base_codes=base_codes,
        base_embeddings=base_embeddings,
        top_k_final=3
    )

    # 出力
    output_path = "output/matched_top3.csv"
    write_output(results, output_path)

if __name__ == "__main__":
    main()
 """