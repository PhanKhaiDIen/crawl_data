# -*- coding: utf-8 -*-
import os
from pathlib import Path
from PIL import Image, ImageOps
import pandas as pd

# ===================== CẤU HÌNH =====================
INPUT_XLSX       = "normalized_data.xlsx"    # dữ liệu nguồn (đã normalization)
OUTPUT_XLSX      = "data_converted.xlsx"     # dữ liệu sau khi cập nhật đường dẫn PNG
SRC_DIR          = Path("../crawl_data/images")  # chỉ tìm ảnh trong thư mục này
OUT_DIR          = Path("images_png")        # nơi lưu ảnh PNG
TARGET_SIZE      = (256, 256)                # kích thước đầu ra
KEEP_ASPECT      = True                      # True: giữ tỉ lệ + pad; False: resize ép
OVERWRITE_PNG    = False                     # False: PNG tồn tại thì bỏ qua convert
SAVE_TO_NEW_COL  = False                     # True: lưu sang cột mới (không ghi đè image_path)
NEW_COLUMN_NAME  = "image_png_path"          # tên cột mới nếu SAVE_TO_NEW_COL=True

# Chấp nhận các đuôi ảnh
VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ===================== HÀM PHỤ =====================
def safe_category(val) -> str:
    """Chuẩn hoá tên thư mục theo 'source' (chỉ a-zA-Z0-9, '-' và '_')."""
    s = str(val).strip().lower()
    if not s or s in {"nan", "none", "null"}:
        s = "unknown"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)

def to_png(img_path: Path, out_path: Path, keep_aspect: bool = True):
    """
    Mở ảnh, EXIF transpose, chuyển RGB, resize/pad về TARGET_SIZE, lưu PNG.
    """
    with Image.open(img_path) as im:
        # Sửa xoay theo EXIF trước khi xử lý
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        if keep_aspect:
            # Pad nền trắng để giữ tỷ lệ
            im = ImageOps.pad(
                im, TARGET_SIZE,
                method=Image.LANCZOS,
                color=(255, 255, 255),
                centering=(0.5, 0.5),
            )
        else:
            # Ép resize đúng kích thước
            im = im.resize(TARGET_SIZE, Image.LANCZOS)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, format="PNG", optimize=True)

def build_basename_index(root: Path) -> dict:
    """
    Lập chỉ mục: basename.lower() -> full path (chỉ trong SRC_DIR).
    Nếu có trùng tên, sẽ giữ file gặp sau cùng và in cảnh báo.
    """
    idx = {}
    clash = set()
    if not root.exists():
        return idx

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            key = p.name.lower()
            if key in idx:
                clash.add(key)
            idx[key] = p
    if clash:
        print(f"[WARN] Có {len(clash)} tên file trùng trong '{root}'. Sẽ lấy file gặp sau cùng.")
    return idx


# ===================== CHẠY =====================
def main():
    if not Path(INPUT_XLSX).exists():
        raise FileNotFoundError(f"Không thấy file: {INPUT_XLSX}")

    df = pd.read_excel(INPUT_XLSX)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Lập chỉ mục trong SRC_DIR để tìm theo basename khi đường dẫn không nằm trong SRC_DIR
    basename_index = build_basename_index(SRC_DIR)
    print(f"Đã lập chỉ mục {len(basename_index)} file trong '{SRC_DIR}'")

    new_paths = []
    n_total = len(df)
    n_ok = n_skip = n_err = 0
    unresolved = []

    for i, row in df.iterrows():
        raw_path = str(row.get("image_path", "")).strip()
        src = safe_category(row.get("source", "unknown"))
        rid = row.get("id", None)

        # Bỏ qua khi trống
        if not raw_path:
            new_paths.append("")
            n_skip += 1
            continue

        p = Path(raw_path)

        # Nếu path không thuộc VALID_EXTS và cũng không có trong index -> bỏ
        # (Trường hợp p không có suffix nhưng vẫn có thể tìm theo basename)
        if p.suffix and p.suffix.lower() not in VALID_EXTS and p.name.lower() not in basename_index:
            new_paths.append("")
            n_skip += 1
            if len(unresolved) < 10:
                unresolved.append((i, raw_path, "unsupported ext"))
            continue

        # Tìm file local trong SRC_DIR:
        # - Nếu raw_path trỏ trực tiếp tới file trong SRC_DIR -> dùng luôn
        # - Ngược lại: tìm theo basename trong chỉ mục
        if p.exists() and SRC_DIR in p.resolve().parents:
            local = p
        else:
            local = basename_index.get(p.name.lower())

        if local is None or not Path(local).exists():
            new_paths.append("")
            n_skip += 1
            if len(unresolved) < 10:
                unresolved.append((i, raw_path, "not found in SRC_DIR"))
            continue

        # Đặt tên file đích: ưu tiên id, thêm prefix source để tránh đè giữa nguồn
        if pd.notna(rid):
            base_name = f"{src}_{str(rid).strip()}.png"
        else:
            base_name = f"{src}_{i}.png"

        out_path = OUT_DIR / src / base_name

        try:
            if out_path.exists() and not OVERWRITE_PNG:
                new_paths.append(str(out_path))
                n_ok += 1
                continue

            to_png(local, out_path, keep_aspect=KEEP_ASPECT)
            new_paths.append(str(out_path))
            n_ok += 1
        except Exception as e:
            print(f"[LỖI] row #{i} | {local} -> {out_path}: {e}")
            new_paths.append("")
            n_err += 1

    # Cập nhật DataFrame
    if SAVE_TO_NEW_COL:
        df[NEW_COLUMN_NAME] = new_paths
    else:
        # Ghi đè vào image_path như pipeline trước đây
        df["image_path"] = new_paths

    df.to_excel(OUTPUT_XLSX, index=False)

    # Báo cáo
    print("\n===== TÓM TẮT =====")
    print(f"Tổng hàng trong {INPUT_XLSX:<28}: {n_total}")
    print(f"Convert PNG thành công               : {n_ok}")
    print(f"Không tìm thấy trong SRC_DIR/skip    : {n_skip}")
    print(f"Lỗi xử lý                            : {n_err}")
    print(f"Ảnh PNG lưu tại                      : {OUT_DIR.resolve()}")
    print(f"Dataset lưu                          : {OUTPUT_XLSX}")

    if unresolved:
        print("\nMột số dòng không xử lý được (mẫu):")
        for idx, rp, why in unresolved:
            print(f" - row #{idx}: image_path='{rp}' | reason={why}")

if __name__ == "__main__":
    main()
