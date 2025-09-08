import os
import time
import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# TF/Keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array


# ===================== CẤU HÌNH =====================
INPUT_XLSX    = "encoding_data.xlsx"              # file dữ liệu nguồn
OUTPUT_XLSX   = "data_with_dups.xlsx"             # file dữ liệu sau cập nhật
PAIRS_XLSX    = "duplicate_pairs.xlsx"            # danh sách cặp trùng chi tiết
TXT_REPORT    = "duplicates.txt"                  # báo cáo văn bản (tuỳ chọn)
IMAGE_COL     = "image_path"                      # tên cột đường dẫn ảnh trong Excel
ID_COL        = "id"                              # tên cột id (nếu không có, dùng index gốc)
SIM_THRESHOLD = 0.90                              # ngưỡng cosine similarity
CACHE_FILE    = "feature_cache.pkl"               # cache embedding để chạy nhanh lần sau
TARGET_SIZE   = (224, 224)                        # kích thước input ResNet50
# ====================================================


# ----------------- Tiện ích chung -------------------
def norm_path(p) -> str:
    """Chuẩn hoá đường dẫn để so khớp khoá map (lower + norm slash)."""
    try:
        return os.path.normpath(str(p)).strip().lower()
    except Exception:
        return str(p).strip().lower()


def load_dataframe(input_xlsx: str, image_col: str, id_col: str):
    """Đọc Excel, lọc chỉ các dòng có ảnh .png và tồn tại trên đĩa.
       Vẫn giữ lại các dòng không hợp lệ để gán NaN sau cùng.
    """
    # Nếu thiếu openpyxl, cài: pip install openpyxl
    df = pd.read_excel(input_xlsx)

    if image_col not in df.columns:
        raise ValueError(f"Không thấy cột '{image_col}' trong {input_xlsx}")

    s_path = df[image_col].astype(str).str.strip()
    is_png = s_path.str.lower().str.endswith(".png")
    exists = s_path.map(lambda p: Path(p).exists())

    mask = is_png & exists
    work = df[mask].copy().reset_index(drop=False)  # giữ index gốc
    work.rename(columns={"index": "_orig_index"}, inplace=True)

    print(f"[INFO] Tổng dòng trong Excel: {len(df)}")
    print(f"[INFO] Ảnh .png tồn tại sẽ xử lý: {len(work)}")

    if id_col not in df.columns:
        print(f"[WARN] Không thấy cột '{id_col}', sẽ dùng index gốc để map/ghi kết quả.")

    return df, work


# --------------- Cache embedding ảnh ----------------
def load_cache(cache_file: str) -> Dict[str, Dict[str, Any]]:
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_cache(cache: dict, cache_file: str) -> None:
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"[WARN] Không lưu được cache: {e}")


# -------- Trích xuất đặc trưng bằng ResNet50 --------
class FeatureExtractor:
    def __init__(self):
        self.model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    def _extract_one(self, img_path: str, target_size=TARGET_SIZE):
        try:
            img = load_img(img_path, target_size=target_size)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feat = self.model.predict(x, verbose=0)  # (1, 2048)
            return feat[0]
        except Exception as e:
            print(f"[LỖI ẢNH] {img_path}: {e}")
            return None

    def load_or_extract(self, image_paths: List[str], cache_file: str) -> dict:
        """Chỉ trích xuất cho các ảnh trong danh sách. Dùng cache nếu có."""
        cache = load_cache(cache_file)
        out = {}

        for p in tqdm(image_paths, desc="Extract features"):
            p_norm = norm_path(p)
            try:
                mtime = Path(p).stat().st_mtime
            except Exception:
                mtime = None

            use_cache = False
            if p_norm in cache:
                meta = cache[p_norm]
                if isinstance(meta, dict) and "feat" in meta and "mtime" in meta:
                    if mtime is None or abs(meta["mtime"] - mtime) < 1e-6:
                        use_cache = True
                        out[p] = meta["feat"]

            if not use_cache:
                feat = self._extract_one(p)
                if feat is not None:
                    out[p] = feat
                    cache[p_norm] = {"feat": feat, "mtime": mtime}

        # đồng bộ cache cho lần sau
        save_cache(cache, cache_file)
        return out


# ----------------- Tính độ tương đồng ----------------
def find_duplicate_pairs(feature_db: dict, threshold: float) -> pd.DataFrame:
    """
    Trả về DataFrame các cặp trùng: path_1, path_2, similarity
    """
    paths = list(feature_db.keys())
    if not paths:
        return pd.DataFrame(columns=["path_1", "path_2", "similarity"])

    X = np.vstack([feature_db[p] for p in paths])  # (N, 2048)
    S = cosine_similarity(X)                       # (N, N)

    rows = []
    n = len(paths)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(S[i, j])
            if sim >= threshold:
                rows.append([paths[i], paths[j], sim])

    pairs_df = pd.DataFrame(rows, columns=["path_1", "path_2", "similarity"])
    pairs_df = pairs_df.sort_values("similarity", ascending=False).reset_index(drop=True)
    print(f"[INFO] Số cặp nghi trùng (>= {threshold}): {len(pairs_df)}")
    return pairs_df


# ----------------- Gán cột vào Excel ------------------
def annotate_excel_with_duplicates(
    df_all: pd.DataFrame,
    pairs_df: pd.DataFrame,
    image_col: str,
    id_col: str,
    output_xlsx: str,
    fill_empty_rows: bool = True,      # giữ hàng thiếu path & để NaN các cột trùng
) -> None:
    """
    - Giữ nguyên mọi dòng gốc (kể cả image_path trống).
    - is_duplicate: 0/1
    - duplicate_with_paths: NaN nếu không có trùng; chứa chuỗi path phân tách ';' nếu có.
    - KHÔNG tạo 'duplicate_with_ids' tổng hợp.
    - Tạo các cột động 'duplicate_with_id_1', 'duplicate_with_id_2', ... (NaN nếu không có).
    """
    out = df_all.copy()

    # Khởi tạo mặc định
    if "is_duplicate" not in out.columns:
        out["is_duplicate"] = 0
    if "duplicate_with_paths" not in out.columns:
        out["duplicate_with_paths"] = np.nan

    has_id = id_col in out.columns

    # Chuẩn hoá path để map
    s_path = out.get(image_col, pd.Series([""] * len(out))).astype(str)
    path_norm_series = s_path.map(norm_path)

    path_to_index = {}
    for idx, p in enumerate(path_norm_series):
        if p:  # chỉ map các dòng có path hợp lệ (không trống)
            path_to_index[p] = idx

    path_to_id = {}
    if has_id:
        for idx, p in enumerate(path_norm_series):
            if p:
                path_to_id[p] = out.loc[idx, id_col]

    # Gom các cặp theo path_norm
    dups_paths = defaultdict(set)  # path_norm -> set(paths trùng)
    dups_ids   = defaultdict(set)  # path_norm -> set(ids trùng)

    for _, row in pairs_df.iterrows():
        p1, p2 = str(row["path_1"]), str(row["path_2"])
        np1, np2 = norm_path(p1), norm_path(p2)
        # chỉ gán nếu cả 2 có trong Excel
        if np1 in path_to_index and np2 in path_to_index:
            # paths
            dups_paths[np1].add(p2)  # lưu path "thô" để dễ đọc
            dups_paths[np2].add(p1)
            # ids
            if has_id:
                id2 = path_to_id.get(np2, None)
                id1 = path_to_id.get(np1, None)
                if id2 is not None:
                    dups_ids[np1].add(id2)
                if id1 is not None:
                    dups_ids[np2].add(id1)

    # 1) Hàng trống path: set is_duplicate=0, duplicate_with_paths = NaN
    if fill_empty_rows:
        empty_mask = (path_norm_series == "") | (path_norm_series.isna())
        out.loc[empty_mask, "is_duplicate"] = 0
        out.loc[empty_mask, "duplicate_with_paths"] = np.nan

    # 2) Với các hàng có path hợp lệ: gán theo cặp
    for npth, idx in path_to_index.items():
        others_p = sorted(list(dups_paths.get(npth, [])))
        if others_p:
            out.at[idx, "is_duplicate"] = 1
            out.at[idx, "duplicate_with_paths"] = ";".join(others_p)
        else:
            out.at[idx, "is_duplicate"] = 0
            out.at[idx, "duplicate_with_paths"] = np.nan

    # 3) Tạo các cột duplicate_with_id_1, duplicate_with_id_2, ... nếu có ID
    if has_id:
        # Tối đa số ID trùng của một hàng
        max_dup_ids = 0
        for npth in path_to_index.keys():
            max_dup_ids = max(max_dup_ids, len(dups_ids.get(npth, [])))

        if max_dup_ids > 0:
            # Tạo sẵn cột NaN
            id_cols = []
            for k in range(1, max_dup_ids + 1):
                col_name = f"duplicate_with_id_{k}"
                id_cols.append(col_name)
                if col_name not in out.columns:
                    out[col_name] = np.nan

            # Gán từng ID trùng vào duplicate_with_id_k
            for npth, idx in path_to_index.items():
                ids_list = sorted(list(dups_ids.get(npth, [])))
                for k in range(1, max_dup_ids + 1):
                    col_name = f"duplicate_with_id_{k}"
                    val = ids_list[k - 1] if k - 1 < len(ids_list) else np.nan
                    out.at[idx, col_name] = val

    # 4) Lưu file kết quả
    out.to_excel(output_xlsx, index=False)
    print(f"[OK] Đã lưu file kết quả: {output_xlsx}")


# ----------------- Báo cáo TXT (tuỳ chọn) ------------
def write_txt_report(pairs_df: pd.DataFrame, txt_path: str, threshold: float):
    with open(txt_path, "w", encoding="utf-8") as f:
        if pairs_df.empty:
            f.write("Không tìm thấy ảnh trùng lặp nào.\n")
        else:
            f.write(f"Tìm thấy {len(pairs_df)} cặp ảnh trùng (similarity >= {threshold}):\n")
            for _, r in pairs_df.iterrows():
                f.write(f"{r['path_1']} <--> {r['path_2']} | sim={r['similarity']:.4f}\n")
    print(f"[OK] Đã lưu: {txt_path}")


# ========================= MAIN ======================
def main():
    t0 = time.time()

    # 1) Đọc Excel & lấy danh sách ảnh .png tồn tại
    df_all, df_work = load_dataframe(INPUT_XLSX, IMAGE_COL, ID_COL)
    image_list = df_work[IMAGE_COL].astype(str).tolist()
    if not image_list:
        print("[STOP] Không có ảnh .png nào để xử lý.")
        return

    # 2) Trích xuất đặc trưng với cache
    extractor = FeatureExtractor()
    feature_db = extractor.load_or_extract(image_list, CACHE_FILE)

    # 3) Tìm cặp ảnh trùng
    pairs_df = find_duplicate_pairs(feature_db, SIM_THRESHOLD)
    pairs_df.to_excel(PAIRS_XLSX, index=False)
    print(f"[OK] Đã lưu: {PAIRS_XLSX}")

    # 4) Gán cờ vào Excel và lưu
    annotate_excel_with_duplicates(
        df_all=df_all,
        pairs_df=pairs_df,
        image_col=IMAGE_COL,
        id_col=ID_COL,
        output_xlsx=OUTPUT_XLSX,
        fill_empty_rows=True,
    )

    # 5) (Tuỳ chọn) Báo cáo TXT
    write_txt_report(pairs_df, TXT_REPORT, SIM_THRESHOLD)

    print(f"[DONE] Tổng thời gian: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
