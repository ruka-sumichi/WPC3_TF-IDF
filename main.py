import pandas as pd
from preprocess import normalize_all_names
from vectorizer import get_vectors
from matcher import find_top3_matches
from output_writer import write_output

def main():
    # ファイルパス設定
    facility_a_path = r"C:\Sumichika\data\A_comparison.csv"
    base_path = r"C:\Sumichika\data\B_base.csv"

    # データ読み込み
    df_a = pd.read_csv(facility_a_path)
    df_base = pd.read_csv(base_path)

    df_query_unique = df_a.drop_duplicates("food_code")

    query_names_raw = df_query_unique["food_name"].tolist()       # 加工していない食品名
    query_codes = df_query_unique["food_code"].tolist()    　     # 整えた食品名
    query_names = normalize_all_names(query_names_raw)            # コード

    base_names = normalize_all_names(df_base["food_name"].tolist())
    base_codes = df_base["food_code"].tolist()
    base_names_raw = df_base["food_name"].tolist()

    # TF-IDF ベクトル取得
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
