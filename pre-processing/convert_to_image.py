import os
from pathlib import Path
from PIL import Image, ImageOps
import pandas as pd

# ========= CẤU HÌNH =========
INPUT_XLSX     = "normalized_data.xlsx"     # chỉ xử lý các ảnh xuất hiện ở đây
OUTPUT_XLSX    = "data_converted.xlsx"
SRC_DIR        = Path("../crawl_data/images")  # chỉ tìm ảnh trong thư mục này
OUT_DIR        = Path("images_png")         # nơi lưu PNG
TARGET_SIZE    = (256, 256)
KEEP_ASPECT    = True                       # True: giữ tỷ lệ + pad; False: resize ép
OVERWRITE_PNG  = False                  # False: png tồn tại thì bỏ qua

# ========= HÀM PHỤ =========
def safe_category(val):
    s = str(val).strip().lower()
    if not s or s in {"nan","none","null"}:
        s = "unknown"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)

def to_png(img_path, out_path, keep_aspect=True):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        if keep_aspect:
            im = ImageOps.pad(im, TARGET_SIZE, method=Image.BICUBIC,
                            color=(255,255,255), centering=(0.5,0.5))
        else:
            im = im.resize(TARGET_SIZE, Image.BICUBIC)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, format="PNG", optimize=True)

def build_basename_index(root: Path) -> dict:
    """Map basename.lower() -> full path, chỉ trong SRC_DIR."""
    idx = {}
    if not root.exists():
        return idx
    for p in root.rglob("*"):
        if p.is_file():
            idx[p.name.lower()] = p
    return idx

# ========= CHẠY =========
df = pd.read_excel(INPUT_XLSX)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lập chỉ mục các file hiện có trong SRC_DIR
basename_index = build_basename_index(SRC_DIR)
print(f"Đã lập chỉ mục {len(basename_index)} file trong '{SRC_DIR}'")

new_paths = []
n_total = len(df)
n_ok = n_skip = n_err = 0
unresolved = []

for i, row in df.iterrows():
    raw = str(row.get("image_path", "")).strip()
    cat = safe_category(row.get("category_l1", "unknown"))
    rid = row.get("id", None)

    if not raw:
        new_paths.append("")
        n_skip += 1
        continue

    p = Path(raw)
    local = None
    if p.exists() and SRC_DIR in p.resolve().parents:
        local = p
    else:
        local = basename_index.get(p.name.lower())

    if local is None or not Path(local).exists():
        new_paths.append("")
        n_skip += 1
        if len(unresolved) < 10:
            unresolved.append((i, raw))
        continue

    if pd.notna(rid):
        base_name = f"{str(rid).strip()}.png"
    else:
        base_name = f"{cat}_{i}.png"

    out_path = OUT_DIR / cat / base_name

    try:
        if out_path.exists() and not OVERWRITE_PNG:
            new_paths.append(str(out_path))
            n_ok += 1
            continue

        to_png(local, out_path, keep_aspect=KEEP_ASPECT)
        new_paths.append(str(out_path))
        n_ok += 1
    except Exception as e:
        print(f"[LỖI] #{i} | {local} -> {out_path}: {e}")
        new_paths.append("")
        n_err += 1

# cập nhật & lưu
df["image_path"] = new_paths
df.to_excel(OUTPUT_XLSX, index=False)

print("\n===== TÓM TẮT =====")
print(f"Tổng hàng trong normalized_data.xlsx : {n_total}")
print(f"Convert PNG thành công             : {n_ok}")
print(f"Không tìm thấy trong SRC_DIR       : {n_skip}")
print(f"Lỗi xử lý                           : {n_err}")
print(f"Ảnh PNG lưu tại                       : {OUT_DIR.resolve()}")
print(f"Dataset lưu                            : {OUTPUT_XLSX}")

if unresolved:
    print("\nMột số dòng không tìm thấy ảnh (mẫu):")
    for idx, rp in unresolved:
        print(f" - row #{idx}: image_path='{rp}'")
