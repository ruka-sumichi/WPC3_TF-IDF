"""
CSVで出力

"""

import pandas as pd
import os
from datetime import datetime

def write_output(results: list[dict], output_path: str) -> None:

    df = pd.DataFrame(results)

    # 出力先ディレクトリがなければ作成
    base_dir = os.path.dirname(output_path)
    os.makedirs(base_dir, exist_ok=True)

    # 元のファイル名と拡張子を分離
    base_name, ext = os.path.splitext(os.path.basename(output_path))

    # 日付文字列を追加
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    dated_name = f"{base_name}_{date_str}{ext}"
    full_path = os.path.join(base_dir, dated_name)

    # CSV 出力
    df.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Output saved to {full_path}")
