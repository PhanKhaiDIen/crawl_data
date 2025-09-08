import pandas as pd
import numpy as np
import unicodedata, re
from pathlib import Path

# ===== cấu hình =====
FILES = [
    "data_cleaned/phone.xlsx",
    "data_cleaned/laptop.xlsx",
    "data_cleaned/camera.xlsx",
    "data_cleaned/speaker.xlsx",
    "data_cleaned/tv.xlsx"
]
SAVE_OUTPUT = True
OUTPUT_PATH = "integrated_data.xlsx"

# ===== helpers =====
def strip_accents_lower(s):
    if pd.isna(s): return s
    s = str(s).strip().lower()
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def to_num(x):
    if pd.isna(x): return np.nan
    s = str(x)
    # giữ số và . , -
    s = re.sub(r"[^\d.,-]", "", s)
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    if s.count(".") > 1:
        s = s.replace(".", "")
    try:
        return float(s) if s else np.nan
    except:
        return np.nan

def choose_best_column(df, candidates):
    avail = [c for c in candidates if c in df.columns]
    if not avail: return None
    nn = {c: df[c].notna().sum() for c in avail}
    return max(nn, key=nn.get)

def unify_schema(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Đưa dataset đã clean về schema chung tối thiểu; không làm nặng phần cleaning."""
    id_col    = choose_best_column(df, ["id","product_id","sku"])
    name_col  = choose_best_column(df, ["name","product_name","title"])
    brand_col = choose_best_column(df, ["brand","brand_clean","brand_name"])
    price_col = choose_best_column(df, ["price","price_num","final_price"])
    img_col   = choose_best_column(df, ["image_path","image","image_url","imageUrl"])
    thumb_col = choose_best_column(df, ["thumbnail_url","thumbnail","image_url","imageUrl"])
    rating    = choose_best_column(df, ["rating_average","rating","avg_rating"])
    qty       = choose_best_column(df, ["quantity_sold_value","quantity_sold","sold"])
    seller    = choose_best_column(df, ["seller_product_id","seller_id","shop_sku"])
    cat_l1    = choose_best_column(df, ["category_l1","category","visible_impression_info_amplitude_category_l1_name"])

    out = pd.DataFrame()
    if id_col:   out["id"] = df[id_col]
    if name_col: out["name"] = df[name_col]
    if brand_col:out["brand"] = df[brand_col]
    if price_col:out["price"] = pd.to_numeric(df[price_col].map(to_num), errors="coerce")
    if img_col:  out["image_path"] = df[img_col]
    if thumb_col:out["thumbnail_url"] = df[thumb_col]
    if rating:   out["rating_average"] = df[rating]
    if qty:      out["quantity_sold_value"] = pd.to_numeric(df[qty], errors="coerce")
    if seller:   out["seller_product_id"] = df[seller]
    out["category_l1"] = df[cat_l1] if cat_l1 else source_name

    # chuẩn hóa text
    for c in ["name","brand","category_l1"]:
        if c in out.columns:
            out[c] = out[c].astype(str).map(strip_accents_lower)
    return out

def deduplicate_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Khử trùng lặp: ưu tiên có ảnh và quantity_sold_value lớn."""
    tmp = df.copy()
    tmp["_has_img"] = tmp["image_path"].astype(str).str.strip().ne("") if "image_path" in tmp.columns else False
    if "quantity_sold_value" not in tmp.columns:
        tmp["quantity_sold_value"] = np.nan
    tmp["_rank"] = tmp["_has_img"].astype(int) * 2 + tmp["quantity_sold_value"].fillna(0)

    # Khoá trùng: ưu tiên 'id', nếu không có thì (name, brand, price)
    if "id" in tmp.columns and tmp["id"].notna().any():
        key_cols = ["id"]
    else:
        tmp["name_norm"]  = tmp["name"].astype(str).map(strip_accents_lower) if "name" in tmp.columns else ""
        tmp["brand_norm"] = tmp["brand"].astype(str).map(strip_accents_lower) if "brand" in tmp.columns else ""
        key_cols = [c for c in ["name_norm","brand_norm","price"] if c in tmp.columns]

    tmp = tmp.sort_values(by=["_rank"], ascending=False)
    tmp = tmp.drop_duplicates(subset=key_cols, keep="first")
    return tmp.drop(columns=[c for c in ["_has_img","_rank","name_norm","brand_norm"] if c in tmp.columns])

# ===== đọc & hợp nhất =====
frames = []
for f in FILES:
    p = Path(f)
    if not p.exists():
        print(f"Bỏ qua: không thấy file {f}")
        continue
    raw = pd.read_excel(p)
    uni = unify_schema(raw, source_name=p.stem)
    frames.append(uni)
    print(f"✔ {p.name} -> {uni.shape}")

if not frames:
    raise RuntimeError("Không có file hợp lệ để integration.")

df_integrated = pd.concat(frames, ignore_index=True, sort=False)

# Loại bản ghi thiếu tối thiểu
for col in ["name","brand"]:
    if col in df_integrated.columns:
        before = len(df_integrated)
        df_integrated = df_integrated[df_integrated[col].astype(str).str.strip().ne("")]
        if len(df_integrated) != before:
            print(f"Loại {before-len(df_integrated)} hàng thiếu '{col}'")

# Khử trùng lặp theo ưu tiên
df_integrated = deduplicate_priority(df_integrated)

# Sắp cột
preferred = [c for c in [
    "id","name","brand","price","rating_average",
    "quantity_sold_value","seller_product_id","category_l1",
    "image_path","thumbnail_url","source_file"
] if c in df_integrated.columns]
rest = [c for c in df_integrated.columns if c not in preferred]
df_integrated = df_integrated[preferred + rest]

print("Integration xong. Shape:", df_integrated.shape)
df_integrated.head(10)

# Lưu file
if SAVE_OUTPUT:
    df_integrated.to_excel(OUTPUT_PATH, index=False)
    print("Đã lưu:", OUTPUT_PATH)
