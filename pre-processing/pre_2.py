import os, re, json, html, argparse, unicodedata
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ========== Config ==========
FOLD_D_FOR_NAME = True  # True => “điện” -> “dien”; False => “đien”

# ========== Helpers ==========
def is_nanlike(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return False

def safe_str(x) -> str:
    if is_nanlike(x) or x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

_VN_D_MAP = str.maketrans({"đ": "d", "Đ": "D"})

def vn_fold_no_diacritics(s: Any, fold_d: bool = True, lower: bool = True) -> str:
    s = safe_str(s)
    if not s:
        return ""
    if fold_d:
        s = s.translate(_VN_D_MAP)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = unicodedata.normalize("NFC", s)
    if lower:
        s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_html_emoji_safe(s: Any) -> str:
    s = safe_str(s)
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_price_robust(x) -> float:
    s = safe_str(x)
    if not s:
        return np.nan
    s = re.sub(r"[^\d,.\-]", "", s)
    if s.count(",") and s.count("."):
        if s.count(",") > s.count("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if s.count(",") == 1 and s.count(".") == 0 and len(s.split(",")[-1]) in (1, 2):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def try_parse_dt_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def to_bool_safe(x):
    if isinstance(x, bool):
        return x
    s = vn_fold_no_diacritics(x, lower=True)
    if s in {"1","true","yes","y","t","co","có","có"}:
        return True
    if s in {"0","false","no","n","f","khong","không","không"}:
        return False
    return np.nan

# ========== Insert helper ==========
def insert_after(df: pd.DataFrame, after_col: str, new_col: str, values) -> None:
    """Chèn cột mới ngay sau 'after_col'; nếu không có 'after_col' thì thêm ở cuối."""
    if new_col in df.columns:
        df.drop(columns=[new_col], inplace=True)
    pos = df.columns.get_loc(after_col) + 1 if after_col in df.columns else len(df.columns)
    df.insert(pos, new_col, values)

# ========== Ảnh (chỉ đếm) ==========
def image_count_smart(val) -> int:
    s = safe_str(val)
    if not s:
        return 0
    try:
        maybe = json.loads(s)
        if isinstance(maybe, list):
            return len(maybe)
    except Exception:
        pass
    return len([p for p in re.split(r"[|,;]+", s) if p.strip()])

# ========== Name & key ==========
COLOR_LIST = ["black","white","silver","gold","blue","green","red","pink","purple","gray","grey","yellow","orange","beige","brown","navy","cyan","magenta"]

STOP_MARKETING = {
    "dien","thoai","smartphone","chinh","hang","chinhhang","moi","new","fullbox","likenew","like","full","seal",
    "bh","bao","hanh","baohanh","cong","ty","cty","xachtay","quoc","te","quocte","ban","phien","phienban",
    "gia","re","giare","khuyen","mai","khuyenmai","uu","dai","uudai","tang","qua","tragop","tra","gop",
    "chuan","hangcongty","hangcty","hangquoc","chinhthuc","hangchinhhang"
}
SPEC_TOKENS = {
    "ram","rom","ssd","hdd","hz","mhz","ghz","mah","w","kw","mp","nm","dpi","inch","in","\"","camera",
    "wifi","bluetooth","chip","cpu","gpu","sim","esim","nano","lte","volte","nfc","ir","usb","typec","type","mic","loa","speaker"
}
MODEL_SUFFIX_KEEP = {"pro","max","plus","ultra","fe","se","lite","edge","neo","gt","fold","flip"}
MODEL_BASE_KEEP = {"iphone","galaxy","note","redmi","mi","poco","pixel","oneplus","realme","oppo","vivo","nova","mate","honor","xperia","zenfone","rog","moto","nokia","y","a","s","m","t"}

def remove_memory_specs_only(s: str) -> str:
    s = re.sub(r"\b\d{1,2}\s*[/xX\-\+]\s*\d{2,4}\s*(gb|g)\b", " ", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\s*(gb|tb)\b", " ", s)
    s = re.sub(r"\b(ram|rom)\s*\d+(?:\.\d+)?\s*g\b", " ", s)
    return s

def remove_other_specs_and_marketing(s: str) -> str:
    s = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", s)
    s = re.sub(r"\b\d{1,2}(?:\.\d)?\s*(inch|in|\"|”)\b", " ", s)
    s = re.sub(r"\b\d{3,5}\s*mah\b", " ", s)
    s = re.sub(r"\b\d{1,4}\s*(w|mp)\b", " ", s)
    s = re.sub(r"\b\d{1,3}%\b", " ", s)
    colors_re = r"\b(" + "|".join(re.escape(c) for c in COLOR_LIST) + r")\b"
    s = re.sub(colors_re, " ", s)
    s = re.sub(r"\b20\d{2}\b", " ", s)
    toks = re.findall(r"[a-z0-9]+", s)
    toks2 = []
    for t in toks:
        if t in STOP_MARKETING: continue
        if t in SPEC_TOKENS: continue
        if t.isdigit() and len(t) >= 4:
            continue
        toks2.append(t)
    return " ".join(toks2)

def remove_specs_and_marketing_v2(basic: str) -> str:
    s = remove_memory_specs_only(basic)
    s = remove_other_specs_and_marketing(s)
    return s

def build_name_key_fixed(raw_name: Any, brand_std: str) -> Tuple[str, str]:
    s_raw = clean_html_emoji_safe(raw_name)
    basic = vn_fold_no_diacritics(s_raw, fold_d=FOLD_D_FOR_NAME, lower=True)  # đảm bảo “điện” -> “dien”

    s = remove_specs_and_marketing_v2(basic)
    tokens = re.findall(r"[a-z0-9]+", s)
    brand_tokens = re.findall(r"[a-z0-9]+", vn_fold_no_diacritics(brand_std, fold_d=True) or "")

    def keep_token(tok: str) -> bool:
        if tok in MODEL_BASE_KEEP: return True
        if re.match(r"[a-z]+[0-9]+", tok) or re.match(r"[0-9]+[a-z]+", tok): return True
        if tok in MODEL_SUFFIX_KEEP: return True
        if tok in {"5g","4g"}: return True
        if len(tok) == 1 and tok.isalpha(): return True
        return False

    toks = []
    brand0 = brand_tokens[0] if brand_tokens else None
    if brand0:
        toks.append(brand0)
    for t in tokens:
        if brand0 and t == brand0:  # bỏ lặp brand
            continue
        if keep_token(t):
            toks.append(t)
        if len(" ".join(toks)) > 40:
            break

    if len(toks) < (1 if brand0 else 0) + 2:
        core = [t for t in tokens if t != brand0][:4]
        toks = ([brand0] if brand0 else []) + core

    dedup = []
    for w in toks:
        if not dedup or w != dedup[-1]:
            dedup.append(w)

    key = " ".join(dedup).strip()
    return basic, key

# ========== Brand normalize ==========
BRAND_MAP = {
    "samsung electronics":"samsung","samsung":"samsung",
    "apple inc":"apple","apple":"apple",
    "xiaomi":"xiaomi","oppo":"oppo","vivo":"vivo","huawei":"huawei","realme":"realme",
    "asus":"asus","lenovo":"lenovo","nokia":"nokia","sony":"sony","canon":"canon","nikon":"nikon",
    "google":"google","oneplus":"oneplus","motorola":"motorola","honor":"honor"
}
def standardize_brand(brand: Any) -> str:
    s = vn_fold_no_diacritics(brand, fold_d=True, lower=True)
    key = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return BRAND_MAP.get(key, key)

# ========== Column aliases ==========
ALIASES = {
    "id":         ["id","product_id","_id","item_id"],
    "name":       ["name","title","product_name"],
    "brand":      ["brand","brand_clean","brand_name","manufacturer"],
    "price":      ["price","final_price","sale_price"],
    "list_price": ["list_price","original_price","regular_price"],
    "url":        ["url","product_url","link","url_path","url_key"],
    "images":     ["images","image_paths","image_path","img","thumbnail","image_first","thumbnail_url"],
    "category":   ["category_l1","category_path","category","breadcrumbs","cat_path"],
    "updated_at": ["updated_at","last_update","modified_at"],
    "created_at": ["created_at","created_time","create_time"],
}
RAM_ALIASES = ["ram","ram_gb","memory_ram"]
ROM_ALIASES = ["rom","storage","rom_gb","capacity","dung_luong","dungluong",
               "bo nho","bo_nho","bo nho trong","bo_nho_trong","storage_gb"]

def find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    for c in df.columns:
        low = c.lower()
        for cand in candidates:
            if cand in low:
                return c
    return None

def select_columns(df: pd.DataFrame, overrides: Dict[str, str] = None) -> Dict[str, Optional[str]]:
    overrides = overrides or {}
    sel = {}
    for key, cands in ALIASES.items():
        if key in overrides and overrides[key] in df.columns:
            sel[key] = overrides[key]
        else:
            sel[key] = find_first_col(df, cands)
    return sel

# ========== RAM/Storage & currency ==========
def parse_unit_number(val: Any) -> Optional[float]:
    s = vn_fold_no_diacritics(val, fold_d=True, lower=True)
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*tb?\b", s)
    if m:
        return float(m.group(1)) * 1024.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*gb?\b", s)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", s)
    if m:
        return float(m.group(1))
    return None

def extract_ram_storage_from_name(name: Any) -> Tuple[Optional[float], Optional[float]]:
    s = vn_fold_no_diacritics(name, fold_d=True, lower=True)
    if not s:
        return (None, None)
    ram = None
    storage = None

    m = re.search(r"\b(\d{1,2})\s*[/xX\-\+]\s*(\d{2,4})\s*(tb|gb)?\b", s)
    if m:
        v1 = float(m.group(1)); v2 = float(m.group(2)); unit2 = m.group(3)
        if unit2 == "tb": v2 *= 1024.0
        ram, storage = v1, v2
        if ram > storage: ram, storage = min(ram, storage), max(ram, storage)

    if ram is None:
        m = re.search(r"\bram\s*(\d+(?:\.\d+)?)\s*g(b)?\b", s)
        if m: ram = float(m.group(1))
    if storage is None:
        m = re.search(r"\b(rom|storage|bo nho(?: trong)?)\s*(\d+(?:\.\d+)?)\s*(tb|gb)\b", s)
        if m:
            storage = float(m.group(2)) * (1024.0 if m.group(3) == "tb" else 1.0)
        else:
            m2 = re.search(r"\b(\d+(?:\.\d+)?)\s*tb\b", s)
            if m2: storage = float(m2.group(1)) * 1024.0

    gbs = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\s*gb\b", s)]
    if ram is None and gbs:
        cand_ram = [v for v in gbs if v <= 24]
        if cand_ram: ram = max(cand_ram)
    if storage is None and gbs:
        cand_sto = [v for v in gbs if v >= 32]
        if cand_sto: storage = max(cand_sto)

    if (ram is None or storage is None) and gbs and len(gbs) >= 2:
        lo, hi = min(gbs), max(gbs)
        if ram is None and lo <= 24: ram = lo
        if storage is None and hi >= 32: storage = hi

    if ram is not None and (ram <= 0 or ram > 64): ram = None
    if storage is not None and (storage <= 0 or storage > 8192): storage = None

    return (storage, ram)

def detect_currency(val: Any) -> str:
    s = safe_str(val); sl = s.lower()
    if ("₫" in s) or ("đ" in sl) or ("vnđ" in sl) or ("vnd" in sl): return "VND"
    if ("$" in s) or ("usd" in sl) or ("us$" in sl): return "USD"
    if ("€" in s) or ("eur" in sl): return "EUR"
    if ("£" in s) or ("gbp" in sl): return "GBP"
    if ("¥" in s) or ("jpy" in sl) or ("yen" in sl): return "JPY"
    if ("₩" in s) or ("krw" in sl): return "KRW"
    if ("฿" in s) or ("thb" in sl): return "THB"
    if ("₱" in s) or ("php" in sl): return "PHP"
    return ""

# ========== Excel writer ==========
def get_excel_writer(path: str):
    try:
        return pd.ExcelWriter(path, engine="xlsxwriter")
    except ModuleNotFoundError:
        try:
            return pd.ExcelWriter(path, engine="openpyxl")
        except ModuleNotFoundError:
            raise RuntimeError("Thiếu engine .xlsx. Cài `pip install xlsxwriter` hoặc `pip install openpyxl`.")

# ========== Applicability (SILENT) ==========
def infer_product_type_min(name: str, category: str = "") -> str:
    s = f"{vn_fold_no_diacritics(name, fold_d=True)} {vn_fold_no_diacritics(category, fold_d=True)}"
    if re.search(r"\b(phone|smartphone|dien thoai|iphone|galaxy|redmi|poco|pixel|oneplus|oppo|vivo|tablet|ipad|laptop|notebook|ultrabook|macbook|pc|desktop)\b", s):
        return "needs_both"
    if re.search(r"\b(ssd|hdd|hard ?drive|o c(ung|u?ng)|drive|memory ?card|sd ?card|micro ?sd|microsd|tf ?card)\b", s):
        return "storage_only"
    if re.search(r"\b(ram|dimm|so-?dimm|sodimm|ddr[2345]?)\b", s):
        return "ram_only"
    if re.search(r"\b(tv|tivi|television|charger|adapter|cu sac|sac nhanh|power adapter|cap sac|day sac|camera|may anh|dslr|mirrorless|watch|smartwatch|dong ho|case|op lung|bao da|kinh cuong luc|bao ve)\b", s):
        return "none"
    return "none"

def spec_app_mask(pt: str) -> Tuple[bool, bool]:
    if pt == "needs_both":   return True, True
    if pt == "storage_only": return False, True
    if pt == "ram_only":     return True, False
    return False, False

# ========== Core transform ==========
def transform_to_excel(input_path: str, output_path: str, overrides: Dict[str, str] = None, sheet_name: str = "transformed"):
    # Load
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Không thấy file input: {input_path}")
    if input_path.lower().endswith(".xlsx"):
        try:
            df = pd.read_excel(input_path)
        except Exception as e:
            raise RuntimeError(f"Không đọc .xlsx: {e}\nCài openpyxl: `pip install openpyxl`.")
    elif input_path.lower().endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Input phải là .xlsx hoặc .csv")

    sel = select_columns(df, overrides)
    t = df.copy()

    # Clean object cols
    for c in [c for c in t.columns if t[c].dtype == object]:
        t[c] = t[c].map(clean_html_emoji_safe)

    # brand chuẩn hoá -> ghi thẳng vào 'brand' (nếu có)
    brand_src = sel["brand"]
    if brand_src:
        t["brand"] = t[brand_src].map(standardize_brand)

    # name gốc để trích spec/chỉ số; sau đó ghi đè cột 'name' bằng bản sạch + name_clean/name_len
    raw_name_col = sel["name"]
    raw_name_series = t[raw_name_col].copy() if raw_name_col else pd.Series([""]*len(t), index=t.index)

    # RAM/Storage tạm từ name (không lưu *_from_name)
    if raw_name_col:
        sr = raw_name_series.map(extract_ram_storage_from_name)
        storage_from_name = pd.Series([a for a, b in sr], index=t.index)
        ram_from_name     = pd.Series([b for a, b in sr], index=t.index)
    else:
        storage_from_name = pd.Series([np.nan]*len(t), index=t.index)
        ram_from_name     = pd.Series([np.nan]*len(t), index=t.index)

    # name clean + name_clean (không lặp brand)
    if raw_name_col:
        keys_df = t.apply(
            lambda row: pd.Series(
                build_name_key_fixed(
                    row.get(raw_name_col, ""),
                    row.get("brand", row.get(brand_src, ""))
                )
            ),
            axis=1
        )
        name_basic = keys_df[0]
        name_key   = keys_df[1]
        t[raw_name_col] = name_basic
        insert_after(t, raw_name_col, "name_clean", name_key)
        insert_after(t, "name_clean", "name_len", t[raw_name_col].map(lambda x: len(safe_str(x))))

    # RAM/Storage từ cột chuyên dụng + fallback từ name, rồi mask theo applicability
    ram_col = find_first_col(t, RAM_ALIASES)
    rom_col = find_first_col(t, ROM_ALIASES)
    ram_series = t[ram_col].map(parse_unit_number) if ram_col else pd.Series([np.nan]*len(t), index=t.index)
    sto_series = t[rom_col].map(parse_unit_number) if rom_col else pd.Series([np.nan]*len(t), index=t.index)
    t["ram_gb"] = ram_series.fillna(ram_from_name)
    t["storage_gb"] = sto_series.fillna(storage_from_name)

    # Extra features từ name gốc
    if raw_name_col:
        def extract_screen_inch(name: Any) -> float:
            s = vn_fold_no_diacritics(name, fold_d=True, lower=True)
            m = re.search(r"\b(\d{1,2}(?:\.\d)?)\s*(inch|in|\")\b", s)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return np.nan
            return np.nan

        def extract_color(name: Any) -> str:
            s = vn_fold_no_diacritics(name, fold_d=True, lower=True)
            for c in COLOR_LIST:
                if re.search(rf"\b{c}\b", s):
                    return c
            return ""

        t["screen_inch"] = raw_name_series.map(extract_screen_inch)
        t["color"]       = raw_name_series.map(extract_color)

    # price & discount_percent (tính nếu thiếu)
    price_col_num = list_col_num = None
    if sel["price"]:
        t[sel["price"]+"_num"] = t[sel["price"]].map(parse_price_robust); price_col_num = sel["price"]+"_num"
    if sel["list_price"]:
        t[sel["list_price"]+"_num"] = t[sel["list_price"]].map(parse_price_robust); list_col_num = sel["list_price"]+"_num"
    if ("discount_percent" not in t.columns) and price_col_num and list_col_num:
        with np.errstate(divide="ignore", invalid="ignore"):
            t["discount_percent"] = np.where(
                (t[list_col_num] > 0) & (~t[price_col_num].isna()),
                (t[list_col_num] - t[price_col_num]) / t[list_col_num] * 100.0,
                np.nan
            ).round(2)

    # currency: tạo sau 'price' và mặc định VND nếu không nhận diện được
    if sel["price"] and sel["price"] in t.columns:
        currency_series = t[sel["price"]].map(detect_currency)
        currency_series = currency_series.replace("", np.nan).fillna("VND")
        insert_after(t, sel["price"], "currency", currency_series)
    else:
        t["currency"] = "VND"

    # discount: tạo sau 'discount_percent' (KHÔNG xoá discount_percent)
    if "discount_percent" in t.columns:
        dp = pd.to_numeric(t["discount_percent"], errors="coerce")
        insert_after(t, "discount_percent", "discount", (dp > 0).fillna(False))
    else:
        t["discount"] = False

    # images: chỉ đếm; chèn sau image_path (nếu có)
    img_col = sel["images"]
    if img_col and img_col in t.columns:
        insert_after(t, img_col, "image_count", t[img_col].map(image_count_smart))
    # bỏ 'image_first' nếu có (không cần)
    t.drop(columns=["image_first"], errors="ignore")

    # SILENT applicability masks
    def _blank_series(df_):
        return pd.Series([""] * len(df_), index=df_.index)
    name_for_app = t[raw_name_col].astype(str) if raw_name_col else _blank_series(t)
    cat_col = sel.get("category")
    cat_for_app = t[cat_col].astype(str) if cat_col and cat_col in t.columns else _blank_series(t)
    pt_series = pd.Series([infer_product_type_min(n, c) for n, c in zip(name_for_app, cat_for_app)], index=t.index)
    ram_app_mask = pd.Series([spec_app_mask(pt)[0] for pt in pt_series], index=t.index)
    sto_app_mask = pd.Series([spec_app_mask(pt)[1] for pt in pt_series], index=t.index)
    if "ram_gb" in t.columns:      t.loc[~ram_app_mask, "ram_gb"] = np.nan
    if "storage_gb" in t.columns:  t.loc[~sto_app_mask, "storage_gb"] = np.nan

    # booleans tự nhận diện
    for c in list(t.columns):
        low = c.lower()
        if low.startswith(("is_","has_")) or low.endswith("_flag") or low in {"available","in_stock"}:
            t[c+"_bool"] = t[c].map(to_bool_safe)

    # timestamps (nếu có)
    if sel["updated_at"]:
        t["updated_at_std"] = try_parse_dt_series(t[sel["updated_at"]])
    if sel["created_at"]:
        t["created_at_std"] = try_parse_dt_series(t[sel["created_at"]])

    # dedup: ưu tiên updated_at_std nếu có
    keys = []
    if sel["id"]:   keys.append(sel["id"])
    if sel["url"]:  keys.append(sel["url"])
    if sel["name"]: keys.append(sel["name"])
    t = t.drop_duplicates()
    if keys:
        if "updated_at_std" in t.columns:
            t = (t.sort_values("updated_at_std", ascending=False)
                   .drop_duplicates(subset=keys, keep="first"))
        elif sel["updated_at"]:
            t = (t.sort_values(sel["updated_at"], ascending=False, na_position="last")
                   .drop_duplicates(subset=keys, keep="first"))
        else:
            t = t.drop_duplicates(subset=keys, keep="first")

    # GHI FILE (không reorder toàn bộ; giữ thứ tự cột gốc, cột mới chèn đúng vị trí)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with get_excel_writer(output_path) as writer:
        t.to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"[✓] Saved → {output_path}")
    print(f"Rows: {len(t):,} | Columns: {len(t.columns)}")

# ========== CLI ==========
def main():
    ap = argparse.ArgumentParser(description="Transform giữ đủ cột gốc; name clean, brand chuẩn hóa; currency(VND), discount; specs; rồi xuất Excel")
    ap.add_argument("--input", default="integrated_data.xlsx", help="đường dẫn input .xlsx/.csv (mặc định: integrated_data.xlsx)")
    ap.add_argument("--output", default="transformed_data.xlsx", help="đường dẫn output .xlsx (mặc định: transformed_data.xlsx)")
    ap.add_argument("--sheet", default="transformed", help="tên sheet (mặc định: transformed)")
    ap.add_argument("--override", nargs="*", help="override alias, ví dụ: name=product_title brand=maker images=photos")
    args = ap.parse_args()

    overrides = {}
    if args.override:
        for kv in args.override:
            if "=" in kv:
                k, v = kv.split("=", 1)
                overrides[k.strip()] = v.strip()

    transform_to_excel(args.input, args.output, overrides, sheet_name=args.sheet)

if __name__ == "__main__":
    main()