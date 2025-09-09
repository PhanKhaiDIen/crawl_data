import pandas as pd
import numpy as np
import json
from pathlib import Path

# ===================== CONFIG =====================
IN_PATH  = "data_converted.xlsx"
OUT_PATH = "encoded_data.xlsx"
ENCODER_DIR = Path("encoders")
LOW_CARD_MAX_UNIQUE = 30  # ngưỡng cho one-hot

# Các cột numeric sẽ cố gắng ép kiểu (nếu tồn tại)
NUMERIC_COLS_CANDIDATES = [
    "price", "discount_percent_clean", "rating_average", "quantity_sold_value", "price_vnd",
    # giữ thêm từ bước trước nếu có
    "price_log1p"
]

# Các cột boolean (đã chuẩn hoá ở bước trước)
BOOL_COLS_CANDIDATES = [
    "has_discount", "has_model_hint", "has_image_path", "has_thumbnail_url", "sane_price_discount"
]

# Các cột categorical tiềm năng (tuỳ dữ liệu thực tế mà có/không)
# Lưu ý: không encode các text dài như "name" ở đây.
CATEGORICAL_COLS_CANDIDATES = [
    "brand", "brand_key",
    "model_key",
    "currency",
    "seller_id",
    "source",
    "visible_impression_info_amplitude_category_l1_name",
    # các bucket (sẽ mã hoá ordinal, không one-hot)
    "price_bucket", "rating_bucket", "sold_bucket", "discount_bucket",
]

# Thứ tự ordinal cho các bucket
ORDINAL_MAPS = {
    "discount_bucket": ["≤0%", "0-10%", "10-20%", "20-30%", "30-50%", "50-100%", ">100%"],
    "price_bucket":    ["≤0.5M", "0.5-1M", "1-2M", "2-5M", "5-10M", "10-20M", ">20M"],
    "rating_bucket":   ["≤2", "2-3", "3-4", "4-4.5", "4.5-5", ">5"],
    "sold_bucket":     ["0", "1-10", "11-50", "51-100", "101-500", "501-1000", ">1000"],
}

# ===================== HELPERS =====================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_int_bool(series: pd.Series) -> pd.Series:
    """Chuyển về 0/1/NA với dtype=Int64 (không tạo cột mới, ghi trực tiếp vào cột gốc)."""
    if pd.api.types.is_bool_dtype(series) or series.dropna().isin([True, False]).all():
        return series.astype("Int64")
    # fallback: map từ chuỗi
    map_vals = {"true": 1, "false": 0, "1": 1, "0": 0, True: 1, False: 0}
    return series.astype(str).str.lower().map(map_vals).astype("Int64")

def ordinal_encode_series(series: pd.Series, order: list) -> pd.Series:
    mapping = {v: i for i, v in enumerate(order)}
    return series.map(mapping).astype("Int64")

def frequency_encode_series(series: pd.Series):
    freq = series.value_counts(dropna=False)
    freq_map = freq.to_dict()
    enc = series.map(freq_map)
    return enc.astype("Int64"), {str(k): int(v) for k, v in freq_map.items()}

def label_encode_stable_series(series: pd.Series):
    # Ổn định: sort theo string của giá trị duy nhất (bỏ NaN), bắt đầu từ 0
    uniques = pd.Series(series.unique())
    uniques_str = uniques.dropna().astype(str).sort_values().tolist()
    label_map = {v: i for i, v in enumerate(uniques_str, start=0)}
    enc = series.astype(str).where(series.notna()).map(label_map)
    return enc.astype("Int64"), label_map

# ===================== MAIN =====================
def main():
    ensure_dir(ENCODER_DIR)

    if not Path(IN_PATH).exists():
        raise FileNotFoundError(f"Không thấy file input: {IN_PATH}")

    df = pd.read_excel(IN_PATH)

    # 1) Numeric: ép kiểu an toàn (không tạo cột mới; KHÔNG sắp xếp cột)
    for c in NUMERIC_COLS_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) Boolean: 0/1/NA (ghi trực tiếp)
    bool_cols = [c for c in BOOL_COLS_CANDIDATES if c in df.columns]
    for c in bool_cols:
        df[c] = to_int_bool(df[c])

    # 3) Ordinal cho bucket: TẠO THÊM cột <col>_ord (cột mới sẽ nằm ở CUỐI, không reorder)
    ordinal_maps_saved = {}
    for col, order in ORDINAL_MAPS.items():
        if col in df.columns:
            df[f"{col}_ord"] = ordinal_encode_series(df[col], order)
            ordinal_maps_saved[col] = {v: i for i, v in enumerate(order)}

    # 4) Chọn categorical còn lại (loại bucket vì đã ordinal)
    cat_cols = []
    for c in CATEGORICAL_COLS_CANDIDATES:
        if c in df.columns and c not in ORDINAL_MAPS:
            cat_cols.append(c)

    # 5) Phân loại low-card vs high-card
    onehot_cols, high_card_cols = [], []
    for c in cat_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique <= LOW_CARD_MAX_UNIQUE:
            onehot_cols.append(c)
        else:
            high_card_cols.append(c)

    # 6) One-hot cho low-card: THÊM cột one-hot ở CUỐI (không xoá cột gốc)
    for c in onehot_cols:
        dummies = pd.get_dummies(df[c], prefix=c, dtype="Int64")
        # gắn lần lượt để giữ thứ tự hiện tại + thêm cột mới cuối cùng
        for newc in dummies.columns:
            df[newc] = dummies[newc]

    # 7) Frequency + stable label cho high-card: THÊM cột <col>_freq, <col>_le
    freq_maps_saved = {}
    label_maps_saved = {}
    for c in high_card_cols:
        fe, fmap = frequency_encode_series(df[c])
        le, lmap = label_encode_stable_series(df[c])
        df[f"{c}_freq"] = fe
        df[f"{c}_le"] = le
        freq_maps_saved[c]  = fmap
        label_maps_saved[c] = lmap

    # 8) Lưu kết quả (không reorder)
    df.to_excel(OUT_PATH, index=False)

    # 9) Lưu metadata encoder
    meta = {
        "low_card_max_unique": LOW_CARD_MAX_UNIQUE,
        "onehot_cols": onehot_cols,
        "high_card_cols": high_card_cols,
        "ordinal_maps": ordinal_maps_saved,
        "frequency_maps": freq_maps_saved,
        "label_maps": label_maps_saved,
        "bool_cols": bool_cols,
        "numeric_cols": [c for c in NUMERIC_COLS_CANDIDATES if c in df.columns],
        "note": "Bucket -> *_ord; High-card -> *_freq & *_le; One-hot -> <col>_<value>. Không reorder cột."
    }
    with open(ENCODER_DIR / "encoders_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved encoded file: {OUT_PATH} | shape: {df.shape}")
    print(f"Saved encoders meta: {ENCODER_DIR/'encoders_meta.json'}")
    print(f"One-hot: {onehot_cols}")
    print(f"High-card: {high_card_cols}")

if __name__ == "__main__":
    main()
