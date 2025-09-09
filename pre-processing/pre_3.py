# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path

# ===================== CẤU HÌNH =====================
IN_PATH  = "transformed_data.xlsx"
OUT_PATH = "normalized_data.xlsx"

APPLY_BRAND_DICT   = True   # Ánh xạ brand đồng nhất
APPLY_MODEL_RULES  = True   # Tạo model_norm từ model_hint theo rule (KHÔNG ghi đè model_hint)
CONVERT_TO_VND     = False  # Quy đổi price -> price_vnd

BRAND_MAP = {
    "apple inc.": "Apple",
    "apple vietnam": "Apple",
    "samsung electronics": "Samsung",
    "xiaomi co., ltd.": "Xiaomi",
    "huawei technologies": "Huawei",
    "huawei": "Huawei",
    "oppo vietnam": "OPPO",
    "oppo": "OPPO",
}

MODEL_RULES = [
    (r"\b(global|chinh hang|chính hãng|hàng chính hãng|new|202\d|202[0-5])\b", ""),
    (r"\s{2,}", " "),
    (r"^-+|-+$", ""),
]

RATES_TO_VND = {
    "VND": 1.0, "USD": 25500.0, "EUR": 28000.0, "JPY": 175.0,
    "KRW": 19.5, "SGD": 19000.0, "THB": 720.0,
}

# ===================== HELPERS =====================
def strip_accents_lower_ascii(s: str):
    """lower + bỏ dấu HOÀN TOÀN (đ/Đ -> d)."""
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = s.replace("đ", "d").replace("Đ", "d")
    nfkd = unicodedata.normalize("NFKD", s)
    no_accent = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_accent).strip()

def slugify(s: str) -> str:
    """Tạo key an toàn để join/so khớp (chỉ a-z0-9 và '-')."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s if s else np.nan

def apply_brand_dictionary(brand: str) -> str:
    """Chuẩn hoá brand theo BRAND_MAP (so khớp dạng chuẩn không dấu)."""
    if pd.isna(brand):
        return brand
    key = strip_accents_lower_ascii(brand)
    return BRAND_MAP.get(key, brand)

def build_model_norm(model_hint: str) -> str:
    """Tạo model_norm từ model_hint theo rule (KHÔNG ghi đè model_hint)."""
    if pd.isna(model_hint):
        return model_hint
    s = str(model_hint).strip().lower()
    for pat, repl in MODEL_RULES:
        s = re.sub(pat, repl, s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s if s else np.nan

def to_bool(s):
    if pd.isna(s): return pd.NA
    if isinstance(s, (bool, np.bool_)): return bool(s)
    s = str(s).strip().lower()
    if s in {"1", "true", "yes", "y"}:  return True
    if s in {"0", "false", "no", "n"}: return False
    return pd.NA

# ===================== NORMALIZATION =====================
def normalize(in_path: str, out_path: str):
    if not Path(in_path).exists():
        raise FileNotFoundError(f"Không thấy file input: {in_path}")

    df = pd.read_excel(in_path)

    # 1) Chuẩn hoá kiểu dữ liệu cốt lõi
    for c in ["price", "discount_percent_clean", "rating_average", "quantity_sold_value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["has_discount", "has_model_hint", "has_image_path", "has_thumbnail_url", "sane_price_discount"]:
        if c in df.columns:
            df[c] = df[c].apply(to_bool).astype("boolean")

    if "currency" in df.columns:
        df["currency"] = (
            df["currency"].astype(str).str.strip().str.upper()
              .replace({"NAN": np.nan, "NONE": np.nan, "NULL": np.nan, "": np.nan})
        )

    # 2) Làm sạch discount/price
    if "discount_percent_clean" in df.columns:
        df["discount_percent_clean"] = df["discount_percent_clean"].clip(lower=0, upper=100)
    if "price" in df.columns:
        df.loc[df["price"] < 0, "price"] = np.nan

    # 3) Chuẩn hoá brand (tùy chọn) — LƯU Ý: có thay đổi brand gốc
    if APPLY_BRAND_DICT and "brand" in df.columns:
        df["brand"] = df["brand"].apply(apply_brand_dictionary)

    # 4) KHÔNG ghi đè model_hint — tạo model_norm riêng, rồi model_key từ model_norm (hoặc fallback model_hint)
    if "model_hint" in df.columns:
        if APPLY_MODEL_RULES:
            df["model_norm"] = df["model_hint"].apply(build_model_norm)
        else:
            df["model_norm"] = pd.NA
    else:
        df["model_norm"] = pd.NA

    # 5) Tạo key chuẩn hoá
    df["name_key"]  = df["name_norm"].apply(slugify)   if "name_norm"  in df.columns else pd.NA
    df["brand_key"] = df["brand_norm"].apply(slugify) if "brand_norm" in df.columns else (
        df.get("brand", pd.Series(pd.NA, index=df.index))
        .apply(lambda x: slugify(strip_accents_lower_ascii(x)) if pd.notna(x) else np.nan)
    )
    if "model_norm" in df.columns and df["model_norm"].notna().any():
        df["model_key"] = df["model_norm"].apply(slugify)
    elif "model_hint" in df.columns:
        df["model_key"] = df["model_hint"].apply(slugify)
    else:
        df["model_key"] = pd.NA

    # 6) (Tuỳ chọn) Quy đổi price về VND
    if CONVERT_TO_VND and {"price","currency"}.issubset(df.columns):
        def convert(row):
            p, cur = row["price"], row["currency"]
            if pd.isna(p) or pd.isna(cur): return np.nan
            rate = RATES_TO_VND.get(str(cur).upper())
            return p * rate if rate else np.nan
        df["price_vnd"] = df.apply(convert, axis=1)

    # 7) Khử trùng lặp theo bộ khóa (ưu tiên có ảnh & sold cao)
    rank = pd.Series(0, index=df.index, dtype="float64")
    if "has_image_path" in df.columns:
        rank += df["has_image_path"].fillna(False).astype(int) * 2
    if "quantity_sold_value" in df.columns:
        rank += df["quantity_sold_value"].fillna(0)
    df["_rank_norm"] = rank

    keys = [c for c in ["name_key", "brand_key", "model_key", "price"] if c in df.columns]
    if keys:
        df = df.sort_values("_rank_norm", ascending=False).drop_duplicates(subset=keys, keep="first")
    df = df.drop(columns=["_rank_norm"], errors="ignore")

    # 8) Dọn text
    for c in [
        "name","brand","seller_id","source","image_path","thumbnail_url",
        "visible_impression_info_amplitude_category_l1_name",
        "impression_info_0_metadata_delivery_zone"
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

    # 9) Schema xuất (giữ cả model_hint gốc + model_norm + model_key)
    base_order = [
        "visible_impression_info_amplitude_category_l1_name",
        "id", "seller_id",
        "name", "name_norm", "name_key",
        "model_hint", "model_norm", "model_key",
        "brand", "brand_norm", "brand_key",
        "price", "currency", "discount_percent_clean", "has_discount",
        "price_vnd",
        "rating_average", "quantity_sold_value",
        "image_path", "source", "thumbnail_url",
        "impression_info_0_metadata_delivery_zone",
    ]
    out_cols = [c for c in base_order if c in df.columns]

    tail2 = [
        "discount_bucket","price_log1p","price_bucket","rating_bucket","sold_bucket",
        "name_len","name_word_count","has_model_hint","has_image_path","has_thumbnail_url","sane_price_discount"
    ]
    out_cols += [c for c in tail2 if c in df.columns and c not in out_cols]

    df_out = df[out_cols]
    df_out.to_excel(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} | shape: {df_out.shape}")

if __name__ == "__main__":
    normalize(IN_PATH, OUT_PATH)
