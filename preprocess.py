import re
import unicodedata
import pandas as pd


_NOISE_PATTERN = re.compile(
    r'(個包装|b品|ポリ容器|ソボロにする|業務用|終売|終|約|時期物|切身|切り身|'
    r'瓶|中国|輸入|星型に抜く|南ア2号|キング|やさしい素材|乾燥|ハーフ|冷凍|冷|凍|'
    r'風味|風|個|本|枚|缶|粒|袋|パック|お徳用|国産|国内産|徳用|カット|入り)'
)

# 空白・全角空白を含む
_SPACE_PATTERN = re.compile(r'[\s\u3000]+')

# カッコ＋中身（全角・半角）
_KAKKO_PATTERN = re.compile(r'（.*?）|\(.*?\)|【.*?】|［.*?］|「.*?」|『.*?』|<.*?>')

    
def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ''

    name = _KAKKO_PATTERN.sub('', name)                        # カッコ除去
    name = _SPACE_PATTERN.sub('', name)                        # 空白除去
    name = _NOISE_PATTERN.sub('', name)                        # ノイズ語除去（数字・単位語含む）
    name = unicodedata.normalize('NFKC', name).lower()         # 正規化＋小文字化
    name = re.sub(r'[^ぁ-んァ-ン一-龯ー]', '', name)            # 日本語以外削除
    return name.strip()



# 複数の食品名を一括正規化する。
def normalize_all_names(names: list[str]) -> list[str]:
    return [normalize_name(name) for name in names]

