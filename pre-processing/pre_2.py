# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path

# ===================== CẤU HÌNH =====================
IN_PATH  = "integrated_data.xlsx"
OUT_PATH = "transformed_data.xlsx"

# ===================== HELPERS =====================
def strip_accents_lower_ascii(s: str):
    """
    Chuẩn hoá cho matching: lower + bỏ dấu HOÀN TOÀN (kể cả 'đ/Đ' -> 'd').
    Ví dụ: 'Điện thoại OPPO' -> 'dien thoai oppo'
    """
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = s.replace("đ", "d").replace("Đ", "d")
    nfkd = unicodedata.normalize("NFKD", s)
    no_accent = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_accent).strip()

def to_num_general(x):
    """Chuẩn hoá số đa định dạng: xử lý nghìn/thập phân an toàn."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan","none","null","-"}:
        return np.nan
    s = re.sub(r"[^\d.,-]", "", s)
    if "," in s and "." in s:
        s = s.replace(",", "")        # giả định , là thousands
    elif "," in s and "." not in s:
        s = s.replace(",", ".")       # giả định , là decimal
    if s.count(".") > 1:
        s = s.replace(".", "")
    try:
        return float(s) if s else np.nan
    except:
        return np.nan

def map_currency_from_zone(zone: str) -> str:
    """Suy ra mã tiền tệ từ delivery zone (có thể mở rộng thêm)."""
    if pd.isna(zone):
        return np.nan
    z = str(zone).strip().lower()
    if any(k in z for k in ["vn", "viet", "việt", "vietnam", "việt nam"]): return "VND"
    if any(k in z for k in ["us", "usa", "america", "united states"]):     return "USD"
    if any(k in z for k in ["eu", "europe", "european"]):                   return "EUR"
    if any(k in z for k in ["jp", "japan", "nihon", "nippon"]):             return "JPY"
    if any(k in z for k in ["kr", "korea", "south korea"]):                 return "KRW"
    if any(k in z for k in ["sg", "singapore"]):                            return "SGD"
    if any(k in z for k in ["th", "thailand", "thai"]):                     return "THB"
    return np.nan

def move_col_after(df: pd.DataFrame, col_to_move: str, after_col: str) -> pd.DataFrame:
    """Đưa cột col_to_move đứng ngay sau after_col (nếu cả hai đều tồn tại)."""
    if col_to_move not in df.columns or after_col not in df.columns:
        return df
    cols = df.columns.tolist()
    cols.remove(col_to_move)
    insert_at = cols.index(after_col) + 1
    cols = cols[:insert_at] + [col_to_move] + cols[insert_at:]
    return df[cols]

# ===================== TRANSFORMATION =====================
def transform(in_path: str, out_path: str):
    if not Path(in_path).exists():
        raise FileNotFoundError(f"Không thấy file: {in_path}")

    df = pd.read_excel(in_path)

    # 1) Chuẩn hoá cho matching
    if "name" in df.columns:
        df["name_norm"] = df["name"].astype(str).map(strip_accents_lower_ascii)
    if "brand" in df.columns:
        df["brand_norm"] = df["brand"].astype(str).map(strip_accents_lower_ascii)

    # 2) Ép numeric
    for col in ["price","discount_percent","rating_average","quantity_sold_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].map(to_num_general), errors="coerce")

    # 3) Currency từ zone -> sau price
    if "impression_info_0_metadata_delivery_zone" in df.columns:
        df["currency"] = df["impression_info_0_metadata_delivery_zone"].map(map_currency_from_zone)
        df = move_col_after(df, "currency", "price")

    # 4) Discount clean + has_discount ngay sau discount_percent_clean
    if "discount_percent" in df.columns:
        dp = df["discount_percent"]
        df["discount_percent_clean"] = dp.clip(lower=-5, upper=100).round(2)
        df["has_discount"] = df["discount_percent_clean"].fillna(0).gt(0)
        df["discount_bucket"] = pd.cut(
            df["discount_percent_clean"],
            bins=[-np.inf, 0, 10, 20, 30, 50, 100, np.inf],
            labels=["≤0%", "0-10%", "10-20%", "20-30%", "30-50%", "50-100%", ">100%"]
        )

    # 5) Price features
    if "price" in df.columns:
        price = df["price"]
        df["price_log1p"] = np.where(price > 0, np.log1p(price), np.nan)
        df["price_bucket"] = pd.cut(
            price,
            bins=[-np.inf, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, np.inf],
            labels=["≤0.5M", "0.5-1M", "1-2M", "2-5M", "5-10M", "10-20M", ">20M"]
        )

    # 6) Rating features
    if "rating_average" in df.columns:
        ra = df["rating_average"]
        df["rating_bucket"] = pd.cut(
            ra, bins=[-np.inf, 2, 3, 4, 4.5, 5, np.inf],
            labels=["≤2", "2-3", "3-4", "4-4.5", "4.5-5", ">5"]
        )

    # 7) Sales features
    if "quantity_sold_value" in df.columns:
        q = df["quantity_sold_value"].fillna(0)
        df["sold_bucket"] = pd.cut(
            q, bins=[-0.1, 0, 10, 50, 100, 500, 1000, np.inf],
            labels=["0", "1-10", "11-50", "51-100", "101-500", "501-1000", ">1000"]
        )

    # 8) Text features
    if "name" in df.columns:
        df["name_len"] = df["name"].astype(str).str.len()
        df["name_word_count"] = df["name"].astype(str).str.split().str.len()
        pattern = re.compile(r"\b([A-Za-z]{1,4}\d{1,4}|[A-Za-z0-9]+-[A-Za-z0-9]+)\b")
        df["model_hint"] = (
            df["name"].astype(str)
            .str.findall(pattern)
            .apply(lambda lst: " ".join(dict.fromkeys(lst)) if isinstance(lst, list) else np.nan)
        )
        df["has_model_hint"] = df["model_hint"].astype(str).str.strip().ne("")

    # 9) Image flags
    if "image_path" in df.columns:
        df["has_image_path"] = df["image_path"].astype(str).str.strip().ne("")
    if "thumbnail_url" in df.columns:
        df["has_thumbnail_url"] = df["thumbnail_url"].astype(str).str.strip().ne("")

    # 10) Sanity
    if "price" in df.columns and "discount_percent_clean" in df.columns:
        df["sane_price_discount"] = ~((df["price"] <= 0) & df["has_discount"])

    # ===== Reorder columns =====
    # Nhóm đầu
    front_cols = [c for c in [
        "visible_impression_info_amplitude_category_l1_name",
        "name","model_hint","name_norm","brand","brand_norm","price","currency", "discount_percent_clean","has_discount","rating_average", "quantity_sold_value","seller_id"
    ] if c in df.columns]

    # Đuôi 1
    tail1 = [c for c in [
        "image_path","source","thumbnail_url","impression_info_0_metadata_delivery_zone"
    ] if c in df.columns]

    # Đuôi 2
    tail2_order = [
        "discount_bucket", "price_log1p", "price_bucket", "rating_bucket", "sold_bucket",
        "name_len", "name_word_count", "has_model_hint",
        "has_image_path", "has_thumbnail_url", "sane_price_discount"
    ]
    tail2 = [c for c in tail2_order if c in df.columns]

    already = set(front_cols + tail1 + tail2)
    others = [c for c in df.columns if c not in already]

    df = df[front_cols + others + tail1 + tail2]

    df.to_excel(out_path, index=False)
    print(f"Saved: {out_path}  | shape: {df.shape}")

if __name__ == "__main__":
    transform(IN_PATH, OUT_PATH)
