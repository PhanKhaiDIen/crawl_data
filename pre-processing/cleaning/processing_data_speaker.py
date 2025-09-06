#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata


# In[32]:


df = pd.read_excel("../../crawl_data/data/data_speaker.xlsx")


# > Số dòng và số cột

# In[33]:


df.info()


# > Xem dữ liệu

# 10 dòng đầu

# In[34]:


df.head(10)


# 10 dòng cuối

# In[35]:


df.tail(10)


# Xem kiểu dữ liệu

# In[36]:


df.dtypes


# Xem phân phối chuẩn

# In[37]:


df.describe()


# > Tỉ lệ giá trị thiếu

# In[38]:


df.isnull().sum()


# In[39]:


num_missing_cols = (df.isnull().sum() > 0).sum()
print("Số cột có giá trị thiếu:", num_missing_cols)


# In[40]:


missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0]

missing_table = pd.DataFrame({
    "Cột": missing_summary.index,
    "Số lượng thiếu": missing_summary.values,
    "Tỉ lệ thiếu (%)": (missing_summary.values / len(df)) * 100
}).sort_values(by="Số lượng thiếu", ascending=False).reset_index(drop=True)

print(missing_table)


# In[41]:


missing_ratio = (missing_summary / len(df)) * 100

plt.figure(figsize=(10, len(missing_ratio) * 0.3))
bars = plt.barh(missing_ratio.index, missing_ratio.values)

plt.xlabel("Tỉ lệ thiếu (%)")
plt.ylabel("Tên cột")
plt.title("Tỉ lệ thiếu dữ liệu theo từng cột")

for bar, val in zip(bars, missing_summary.values):
    plt.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height()/2,
        str(val),
        va='center'
    )
plt.margins(y=0)
plt.subplots_adjust(top=0.98, bottom=0.02)
plt.show()


# # ***CLEANING***

# ***Bước làm sạch dữ liệu***

# Xử lý giá trị thiếu (missing values)

# Nếu cột thiếu toàn bộ hoặc >95% thì drop.
# Nếu cột quan trọng nhưng thiếu ít → thay thế (mean/median cho số, mode cho categorical).

# Làm gọn tên sản phẩm

# In[ ]:


mask_speaker = df['name'].str.contains('loa', case=False, na=False)
name_map = {
    'Loa Momo thông báo chuyển khoản - Tặng khay QRCode - Hàng chính hãng': 'Loa Momo',
    'Loa Bluetooth JBL Charge 6 - Hàng chính hãng': 'Loa Bluetooth JBL Charge 6',
    'Loa Bluetooth ngoài trời Xiaomi BHR4802GL | Chống nước IP67 | Thời gian chơi nhạc 10h | Bản quốc tế | Bảo hành 1 năm | Hàng Chính Hãng': 'Loa Xiaomi BHR4802GL',
    'Loa vi tính để bàn SUPER BASS cực đã dòng loa cao cấp cho laptop, pc, điện thoại MÀU ĐEN 106x66mm - Hàng Chính Hãng': 'Loa vi tính để bàn SUPER BASS',
    'Loa Bluetooth Mini Karaoke Không Dây, Kèm 2 Mic, Âm Thanh Sống Động, Đèn LED 16 Màu - HÀNG CHÍNH HÃNG MINIIN': 'Loa Bluetooth Mini Karaoke',
    'Loa Bluetooth Chữ G LED RGB, Sạc Nhanh Không Dây, Đèn Nháy Theo Nhạc, Đồng Hồ Báo Giờ Đa Chức Năng - HÀNG CHÍNH HÃNG MINIIN': 'Loa Bluetooth Chữ G',
    'Loa Bluetooth Anker Soundcore 3 A3117 - Hàng chính hãng': 'Loa Bluetooth Anker Soundcore 3 A3117',
    'Loa Máy Tính E1002 Dùng Cho Máy Tính Bàn, PC, Laptop, Âm Thanh 4D Chất Lượng Tốt - Hàng Chính Hãng': 'Loa Máy Tính E1002',
    'Loa vi tính soundbar Q2 cho máy tính laptop Hàng nhập khẩu': 'Loa vi tính soundbar Q2',
    '[NEW ARRIVAL] Loa Bluetooth Marshall Willen II Portable - Hàng chính hãng': 'Loa Bluetooth Marshall Willen II Portable',
    'Loa Bluetooth JBL Clip 5 - Hàng Chính Hãng': 'Loa Bluetooth JBL Clip 5',
    'Loa hội nghị Bluetooth Anker PowerConf S500 - Hàng Chính Hãng': 'Loa Bluetooth Anker PowerConf S500',
    'Loa CROWN 5 đế có bluetooth ( Hàng Chính Hãng )': 'Loa CROWN 5',
    'Loa máy trợ giảng không dây Aporo T18 Plus 2.4G Wifi Bluetooth 5.0 Hàng Chính Hãng (Lựa chọn thêm Hộp bảo vệ Aporo)': 'Loa Aporo T18 Plus 2.4G',
    'Loa Bluetooth Karaoke Su-Yosd YS104 - Loa xách tay mini chất liệu nhựa cao cấp, công suất 6W và micro không dây - Hàng nhập khẩu': 'Loa Bluetooth Karaoke Su-Yosd YS104',
    'LOA GỖ BLUETOOTH XM 520 - ÂM THANH CỰC HAY - JL - HÀNG CHÍNH HÃNG': 'LOA GỖ BLUETOOTH XM 520',
    'Loa vi tính loa gỗ công suất lớn dùng cho PC, Laptop, ĐT,Tivi - LEERFEI YST 3513 2.0 - NTH - Hàng Nhập Khẩu': 'Loa LEERFEI YST 3513 2.0',
    'Loa Bluetooth JBL Flip 6 - Hàng Chính Hãng': 'Loa JBL Flip 6',
    'Loa vi tính Q9 Sound Bar HD Led RGB cho máy tính, laptop, điện thoại, máy tính bảng hàng nhập khẩu': 'Loa vi tính Q9 Sound Bar',
    'Loa Vi Tính 2.0 TAKO A730 - Hàng Chính Hãng':'Loa 2.0 TAKO A730',
    'Loa Bluetooth nghe nhạc không dây 100W công suất lớn Super Bass có cổng usb, thẻ nhớ, line 3.5mm pin 10400MAH sạc Type C Chip DSP TWS lên 200W cao cấp Hàng Chính Hãng PKCB':'Loa Bluetooth PKCB',
    'Loa Máy Tính Để Bàn, Âm Thanh Siêu Trầm, Dành Cho Học Online, Nghe Nhạc, Xem Phim - Hàng Chính Hãng': 'Loa Máy Tính',
    'Loa Bluetooth HXSJ V6 Bản Mở Rộng Hỗ Trợ Kết Nối Bluetooth 5.0, Thẻ Nhớ, USB, Công suất 2 loa 10W Nhiều Màu Sắc - Hàng chính hãng': 'Loa HXSJ V6',
    'Máy trợ giảng không dây TAKSTAR E261W khoảng cách truyền 40M, công suất cao 25W -Tặng túi đựng máy trợ giảng - Hàng Chính Hãng':'Máy trợ giảng TAKSTAR E261W',
    'Combo Bộ Ba Loa Máy Tính Doron MC-D221 Super Bass , Âm Thanh Vòm 9D - Có Đèn Led - Hàng Nhập Khẩu' :'Bộ Ba Loa Máy Tính Doron MC-D221',
    'Loa máy tính Game thủ LDK.ai RGB 2.1 LeeFei - Hàng chính hãng': 'Loa LDK.ai',
    'Loa gỗ vi tính E350 Sound Bar HD nhỏ gọn - hàng nhập khẩu': 'Loa E350',
    'Loa Bluetooth Không Dây Siêu Bass Âm Thanh Trầm Nghe Nhạc Usb Thẻ Nhớ - Hàng Chính Hãng PKCB': 'Loa PKCB',
    'Loa Bluetooth PKCB Bản Mở Rộng, chống nước IPX5 Hỗ Trợ Kết Nối Bluetooth ,Thẻ Nhớ, USB - Hàng chính hãng': 'Loa PKCB IPX5',
    'Loa Bluetooth PKCB Bản Mở Rộng, chống nước IPX5 Hỗ Trợ Kết Nối Bluetooth ,Thẻ Nhớ, USB - Hàng chính hãng': 'Loa PKCB IPX5',
    'Loa Bluetooth PKCB Bản Mở Rộng, chống nước IPX5 Hỗ Trợ Kết Nối Bluetooth ,Thẻ Nhớ, USB - Hàng chính hãng': 'Loa PKCB IPX5',
    'Loa Bluetooth speaker Loa di động chip DPS khuyếch đại âm thanh công suất lớn 50W Kết nối TWS lên 100W  Hàng Chính Hãng': 'Loa di động',
    'Loa Nghe Nhạc Bluetooth Không Dây Mini LittleFUN - Hàng Chính Hãng': 'Loa LittleFUN',
    'Máy trợ giảng không dây SHIDU kết nối bằng tần số cao cấp, mic trợ giảng cho giáo viên và hướng dẫn viên du lịch, loa trợ giảng cài áo tiện lợi, Hàng nhập khẩu': 'Máy trợ giảng SHIDU',
    'Loa bluetooth mini 5.0 nhỏ gọn PKCB PF1002 - Hàng chính hãng': 'Loa PKCB PF1002',
    'Hàng Chính Hãng- Loa Máy Vi Tính SADA V-196, Hỗ Trợ Đèn Led, Âm Thanh Siêu Trầm': 'Loa SADA V-196',
    'Máy trợ giảng Shidu SD- M500UHF không dây - Hàng chính hãng': 'Máy trợ giảng Shidu SD- M500UHF',
    'Đầu jack chuyển đổi âm thanh từ cổng 3.5mm cái sang cổng 6.5mm đực chính hãng UGREEN 20503 - Hàng chính hãng': 'Đầu jack UGREEN 20503',
    'Loa Vi Tính SoundMax A-130/2.0 6W': 'Loa SoundMax A-130',
    'Loa bluetooth 5.3 PKCB 30W Super Bass TWS lên 60w TF Card/ Line in 3.5mm / AUX Stereo Surround, Loa Không Dây Nghe Nhạc Pin 3000Mah - Hàng Chính Hãng':'Loa AUX Stereo Surround',
    'Loa bluetooth LG xboom Grab - Hàng chính hãng': 'Loa LG xboom',
    'Loa Bluetooth Xiaomi Speaker - GiaPhucStore | Hàng Chính Hãng': 'Loa Xiaomi Speaker',
    'Loa vi tính để bàn cho laptop, pc, điện thoại MÀU ĐEN - Hàng Nhập Khẩu':'Loa vi tính',
    'Bộ 3 Loa Máy Tính D222 Có Đèn LED, Âm Thanh 9D Chất Lượng Tốt, Nghe Nhạc Hay -  Hàng Nhập Khẩu':'Bộ 3 Loa Máy Tính D222',
    'Loa kéo công suất lớn bass 50 đơn Bossinon W-AM5579AK- CÔNG SUẤT 1.200W- Hàng Chính Hãng': 'Loa Bossinon W-AM5579AK',
    'Loa karaoke Dalton K210C công suất 850W, 2 bass 10 inches - HÀNG CHÍNH HÃNG ( BẢO HÀNH 12 THÁNG )': 'Loa Dalton K210C',
    'MÁY TRỢ GIẢNG KHÔNG DÂY APORO T28 UHF CÓ CHỐNG NƯỚC IP67 CÔNG SUẤT LỚN HÀNG CHÍNH HÃNG': 'MÁY TRỢ GIẢNG APORO T28 UHF',
    'Máy trợ giảng không đây Aporo  T20 UHF2.4G hàng chính hãng cho giáo viên': 'Loa Aporo T20 UHF2',
    'Loa kéo 5 tấc Bossinon W-AM5578K - _Kích thước: 510 (W) x 510 (D) x 840 (H)mm _Công suất: 1200Watts- Hàng chính Hãng': 'Loa Bossinon W-AM5578K',
    'Loa Logitech Bluetooth Ultimate Ears Everboom - Chống nước IP67 -  Hàng Chính Hãng': 'Loa Logitech Ultimate Ears Everboom',
    'Loa Bluetooth iCore mini - Hàng chính hãng': 'Loa iCore mini',
    'Máy trợ giảng Chính Hãng Aporo T18 2.4G Plus, T25 2.4G, Hàng Chính Hãng  (Bảo hành 12 tháng Quà Khuyến Mại)': 'Máy trợ giảng Aporo T18',
    'Loa phóng thanh, Loa bán hàng có ghi âm, phát lại có cổng Usb - hàng chính hãng': 'Loa',
    'Loa Bluetooth Alpha Works W88 - Hàng Chính Hãng': 'Loa Alpha Works W88',
    'Loa Bluetooth Karaoke  QIXI SK-2036 , Tặng Kèm 2 Micro Không Dây Cao Cấp , Hát Karaoke Nghe Nhạc Bass mạnh -Hàng Chính Hãng': 'Loa QIXI SK-2036',
    'LOA BLUETOOTH KARAOKE SUYOSD YS-230 KÈM 2 Micro không dây -Hàng Chính Hãng': 'Loa SUYOSD YS-230',
    'Loa Bluetooth Marshall Acton III - Hàng Chính Hãng': 'Loa Marshall Acton III',
    'Loa vi tính cao cấp Logitech Z313 2.1 loa siêu trầm nhỏ gọn - Hàng Chính Hãng': 'Loa Logitech Z313',
    'Loa Bluetooth SoundMax R-800 - Hàng Chính Hãng': 'Loa SoundMax R-800',
    'Loa Bluetooth có đèn iCore B500 - Hàng Chính Hãng': 'Loa iCore B500',
    'Loa Bluetooth PKCB Bản Mở Rộng, chip DSP, Hỗ Trợ Kết Nối Bluetooth, USB, Thẻ Nhớ,  dây 3.5mm 100W TWS Âm Thanh Sống Động - Hàng chính hãng': 'Loa PKCB',
    'Loa Bluetooth JBL Charge 5 JBLCHARGE5 - Hàng chính hãng': 'Loa JBL Charge 5',
    '(Nghe To) Loa Cộng Hưởng Khuếch Đại Âm Thanh KhoNCC Hàng Chính Hãng - Vừa Giá Đỡ Điện Thoại Chắc Chắn - KPD-1087-LoaCH': 'Loa KPD-1087',
    'Loa bluetooth Hoco HC3 V5.0 - loa thể thao không dây cao cấp âm thanh sống động tương thích nhiều thiết bị - hàng chính hãng': 'Loa Hoco HC3',
    'Loa bluetooth Microlab MS210 - Hàng Chính Hãng': 'Loa MS210',
    'Loa máy tính latop SoundMax A160 - Hàng chính hãng': 'Loa SoundMax A160',
    'Loa Máy Tính Doron V-196 - Âm Thanh Siêu Trầm 4D, Bass Êm - Hàng Chính Hãng': 'Loa Doron V-196',
    'Loa vi tính 2.0 SoundMax A140 Tổng Công Suất': 'Loa SoundMax A140',
    'Loa Bluetooth không dây Speaker nghe USB thẻ nhớ Cao cấp Hàng chính hãng  ': 'Loa không dây',
    'Loa Bluetooth IPx56 Plus Bản Mở Rộng, chống nước, âm thanh Bass cực mạnh. Hỗ Trợ Kết Nối Bluetooth 5.0, Micro Nhiều Màu Sắc - Hàng chính hãng': 'Loa IPx56',
    'Loa Bluetooth 60W công suất lớn Super Bass chống nước IPX5 pin 6600MAH sạc nhanh Type C công nghệ AI Hàng Chính Hãng PKCB': 'Loa IPX5',
    'Loa Bluetooth mini TWS 5.0  wireless không dây - [Hàng Chính Hãng': 'Loa mini TWS 5.0',
    'Loa Bluetooth kèm pin sạc dự phòng Energizer BTS-103BK, Hỗ trợ chức năng Rảnh tay, FM, thẻ Micro SD, USB, AUX - Hàng chính hãng': 'Loa BTS-103BK',
    'Loa Bluetooth Không Dây Hoco BS33 - Hàng chính Hãng': 'Loa Hoco BS33',
    'Máy trợ giảng Unizone UZ-U2 - Máy trợ giảng không dây Hàn Quốc chính hãng mới nhất 2020': 'Máy trợ giảng Unizone UZ-U2',
    'Loa Bluetooth Kiêm Pin Sạc Dự Phòng Zealot S1 – Hàng Nhập Khẩu': 'Loa Zealot S1',
    'Máy trợ giảng SHIDU SD-S358 - Hàng Chính Hãng': 'Máy trợ giảng SHIDU SD-S358',
    'Loa Nghe nhạc USB, thẻ nhớ Craven 836 2pin (màu ngẫu nhiên) - hàng nhập khẩu': 'Loa Craven 836',
    'Máy Nghe Nhạc MP3 Ruizu X02 8GB - Hàng Chính Hãng': 'Máy Nghe Nhạc Ruizu X02',
    'Loa bluetooth 5.3 PKCB 20W Super Bass TWS lên 40w TF Card/ Line in 3.5mm / AUX Stereo Surround, Loa Không Dây Nghe Nhạc Pin 3000Mah - Hàng Chính Hãng': 'Loa 5.3 PKCB',
    'Loa Kéo Nanomax SK-12X5 ( Xám ) Bass 30cm 400w Karaoke Bluetooth Tặng Kèm 2 Mic Hát karaoke. Hàng chính hãng':'Loa Nanomax SK-12X5',
    'Loa Bluetooth Anker Soundcore Select 4 Go A31X1 - Hàng chính hãng': 'Loa Anker Soundcore Select 4 Go A31X1',
    'Loa Bluetooth di động Energizer BTS081BK, bảo hành 24 tháng 1 đổi 1 - Hàng chính hãng': 'Loa BTS081BK',
    '[NEW ARRIVAL] Loa Bluetooth Marshall Emberton III Portable - Hàng chính hãng':'Loa Marshall Emberton III Portable',
    'Loa Bluetooth SoundMax  AT-100 - Hàng chính hãng':'Loa AT-100',
    'Loa Bluetooth Logitech Ultimate Ears BOOM 4 - Chống nước, bụi - Hàng Chính Hãng': 'Loa BOOM 4',
    'Loa Bluetooth JBL Go 4 JBLGO4 - Hàng chính hãng': 'Loa JBL Go 4',
    'Loa Bluetooth JBL Xtreme 4 - Hàng Chính Hãng': 'Loa JBL Xtreme 4',
    '12LW801 Củ loa Bass 12inch - 3 tấc Ferrite 500W 8Ω 18 Sound-HÀNG CHÍNH HÃNG': 'Củ loa',
    'Loa Bluetooth S2025 kết nối không dây, âm thanh vượt trội đẳng cấp không gian-Hàng Chính Hãng': 'Loa S2025',
    'Loa Bluetooth WEKOME BELUGA D10 - Kết Nối Bluetooth 5.3 Thế Hệ Mới, Thiết Kế Tinh Tế, Sang Trọng,  Âm Thanh Cực Đỉnh - Hàng Chính Hãng': 'Loa WEKOME BELUGA D10',
    'Loa Kéo Nanomax SK-15D2 Xám Bass 40cm Công Suất 550w Karaoke Bluetooth Hàng Chính Hãng': 'Loa SK-15D2',
    'Loa Bluetooth Harman Kardon Go Play 3  - Hàng Chính Hãng': 'Loa Harman Kardon',
    'Loa Vi Tính Logitech Z407 Bluetooth, công suất 80W - Hàng chính hãng': 'Loa Logitech Z407',
    'Loa Bluetooth Sony SRS-XB100 - Hàng chính hãng': 'Loa Sony SRS-XB100',
    'Loa Bluetooth Mini Sothing DW13 Vintage Retro Âm Thanh 3D, Decor Phòng Ngủ, Quà Tặng- Hàng chính hãng': 'Loa Mini Sothing DW13',
    'Loa Bluetooth Sony SRS-XB100 - Hàng Chính Hãng': 'Loa Sony SRS-XB100',
    'Loa Bluetooth Marshall Stanmore III - Hàng chính hãng':'Loa Marshall Stanmore III',
    'Loa Bluetooth Marshall Woburn III - Hàng chính hãng':'Loa Marshall Woburn III',
}

df.loc[mask_speaker, 'name'] = df.loc[mask_speaker, 'name'].map(name_map)
df_speaker = df[mask_speaker].copy()


# In[42]:


# Ngưỡng drop cột theo tỷ lệ thiếu
MISSING_COL_THRESHOLD = 0.95  # >95% thiếu thì drop

def strip_accents_lower(s):
    if pd.isna(s): return s
    s = str(s).strip().lower()
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def to_num(x):
    """Bóc tách số từ chuỗi (ưu tiên VNĐ là số nguyên), lỗi -> NaN."""
    if pd.isna(x): return np.nan
    s = str(x)
    try:
        return float(s)
    except:
        pass
    s = re.sub(r"[^\d.]", "", s)
    if s.count(".") > 1:  # nghi ngờ dấu . là ngăn cách nghìn
        s = s.replace(".", "")
    try:
        return float(s) if s else np.nan
    except:
        return np.nan

print("Kích thước ban đầu:", df.shape)


# In[43]:


def parse_price(x):
    """Giữ lại chữ số và chuyển về float; trống -> NaN."""
    if pd.isna(x):
        return np.nan
    s = re.sub(r"[^\d]", "", str(x))
    return float(s) if s else np.nan

def strip_accents_lower(s):
    """Chuẩn hóa text: bỏ dấu, lowercase, strip khoảng trắng."""
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


# In[44]:


df_clean = df.copy()


# In[45]:


# Drop cột có tỷ lệ thiếu > ngưỡng

null_ratio = df_clean.isna().mean()
drop_cols = list(null_ratio[null_ratio > MISSING_COL_THRESHOLD].index)
df_clean = df_clean.drop(columns=drop_cols)
print(f"Đã drop {len(drop_cols)} cột thiếu > {int(MISSING_COL_THRESHOLD*100)}%")
print("Sau drop cột:", df_clean.shape)


# In[46]:


for key_col in ["id","name"]:
    if key_col in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean[key_col].notna()]
        print(f"Loại {before-len(df_clean)} hàng thiếu '{key_col}'")


# In[47]:


if "id" in df_clean.columns:
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["id"])
    print("Drop duplicates theo 'id':", before - len(df_clean))
elif set(["name","brand_name"]).issubset(df_clean.columns):
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["name","brand_name"])
    print("Drop duplicates theo (name, brand_name):", before - len(df_clean))

print("Kích thước hiện tại:", df_clean.shape)


# In[48]:


for col in ["price", "list_price"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(parse_price)
if set(["price","list_price"]).issubset(df_clean.columns):
    df_clean["discount_percent"] = np.where(
        df_clean["list_price"].gt(0) & df_clean["price"].notna(),
        (df_clean["list_price"] - df_clean["price"]) / df_clean["list_price"] * 100.0,
        np.nan
    )


# In[49]:


def choose_best_column(frame, candidates):
    avail = [c for c in candidates if c in frame.columns]
    if not avail: return None
    nn = {c: frame[c].notna().sum() for c in avail}
    return max(nn, key=nn.get)

def coalesce(series_list):
    out = None
    for s in series_list:
        if s is None: 
            continue
        out = s if out is None else out.combine_first(s)
    return out

price_candidates      = ["price","final_price","sale_price","deal_price","current_price","best_price"]
list_price_candidates = ["list_price","original_price","price_before_discount","regular_price","old_price","reference_price"]
rate_candidates       = ["discount_rate","discount_percent","discountPercentage","discount_percent_api"]
abs_disc_candidates   = ["discount","discount_amount","price_discount","saved_amount"]

price_col      = choose_best_column(df_clean, price_candidates)
list_price_col = choose_best_column(df_clean, list_price_candidates)
rate_col       = choose_best_column(df_clean, rate_candidates)
abs_col        = choose_best_column(df_clean, abs_disc_candidates)

print("Cột phát hiện:")
print(" - price      :", price_col)
print(" - list_price :", list_price_col)
print(" - rate(%)    :", rate_col)
print(" - discount₫  :", abs_col)

# Ép kiểu số
if price_col:      df_clean["_price_fix"]      = pd.to_numeric(df_clean[price_col].map(to_num), errors="coerce")
if list_price_col: df_clean["_list_price_fix"] = pd.to_numeric(df_clean[list_price_col].map(to_num), errors="coerce")
if rate_col:       df_clean["_rate_fix"]       = pd.to_numeric(df_clean[rate_col].map(to_num), errors="coerce")
if abs_col:        df_clean["_disc_abs_fix"]   = pd.to_numeric(df_clean[abs_col].map(to_num), errors="coerce")

# Ưu tiên 1
dp1 = None
if price_col and list_price_col:
    lp = df_clean["_list_price_fix"]; pr = df_clean["_price_fix"]
    with np.errstate(divide='ignore', invalid='ignore'):
        dp1 = (lp - pr) / lp * 100.0
        dp1 = dp1.where((lp > 0) & pr.notna())

# Ưu tiên 2
dp2 = None
if rate_col:
    r = df_clean["_rate_fix"]
    dp2 = pd.to_numeric(np.where(r <= 1.0, r * 100.0, r), errors="coerce")

# Ưu tiên 3
dp3 = None
if abs_col and list_price_col:
    lp = df_clean["_list_price_fix"]; da = df_clean["_disc_abs_fix"]
    with np.errstate(divide='ignore', invalid='ignore'):
        dp3 = pd.to_numeric(da / lp * 100.0, errors="coerce").where(lp > 0)

# Gộp & làm sạch
dp = coalesce([dp1, pd.Series(dp2) if dp2 is not None else None, dp3])
if dp is not None:
    dp = pd.to_numeric(dp, errors="coerce").clip(-5, 100).round(2)
    df_clean["discount_percent"] = dp
else:
    df_clean["discount_percent"] = np.nan

print("Số dòng có discount_percent:", int(df_clean["discount_percent"].notna().sum()), "/", len(df_clean))


# In[50]:


for col in ["name", "brand_name", "url_key", "url_path"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(str).map(strip_accents_lower)


# In[51]:


if "brand_name" in df_clean.columns:
    df_clean["brand_clean"] = df_clean["brand_name"].str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()

df_clean[["price","list_price","discount_percent"]].head(3) if "price" in df_clean.columns else df_clean.head(3)


# Xử lý missing còn lại
# 
# Numeric: điền median
# 
# Categorical: điền mode
# 
# -> Chỉ áp dụng cho các cột KHÔNG phải khóa/id/text quan trọng

# In[52]:


from pandas.api.types import is_numeric_dtype


# In[53]:


# Cột không nên đụng tới khi fill (nếu tồn tại)
do_not_touch = set([c for c in ["id","sku","url_key","url_path"] if c in df_clean.columns])

num_cols = [c for c in df_clean.columns if is_numeric_dtype(df_clean[c]) and c not in do_not_touch]
cat_cols = [c for c in df_clean.columns if not is_numeric_dtype(df_clean[c]) and c not in do_not_touch]


# In[54]:


# Fill numeric bằng median (nếu còn dữ liệu hợp lệ)
for c in num_cols:
    if df_clean[c].isna().any():
        if df_clean[c].notna().any():  # có ít nhất 1 giá trị không NaN
            med = df_clean[c].median(skipna=True)
            df_clean[c] = df_clean[c].fillna(med)
        else:
            print(f"Cột {c} toàn NaN, không thể tính median.")

# Fill categorical bằng mode (nếu có mode hợp lệ)
for c in cat_cols:
    if df_clean[c].isna().any():
        mode_vals = df_clean[c].mode(dropna=True)
        if not mode_vals.empty:
            df_clean[c] = df_clean[c].fillna(mode_vals.iloc[0])
        else:
            print(f"Cột {c} toàn NaN, không thể tính mode.")

print("Đã fill missing cho numeric & categorical (ổn định hơn).")


# In[55]:


print("Kích thước cuối:", df_clean.shape)

# Thống kê cột còn thiếu sau fill
missing_left = df_clean.isna().sum()
missing_left = missing_left[missing_left > 0].sort_values(ascending=False)
print("Cột còn thiếu (top 15):")
print(missing_left.head(15))

# Sắp xếp cột: ưu tiên cột hay dùng lên đầu (nếu có)
preferred = [c for c in ["id","sku","name","brand_name","brand_clean","price","final_price","list_price","original_price","discount_percent","url_key","url_path"] if c in df_clean.columns]
rest = [c for c in df_clean.columns if c not in preferred]
df_clean = df_clean[preferred + rest]


# In[56]:


df = df_clean.copy()


# In[57]:


df.head(10)


# Lọc các cột chỉ để lại các cột phục vụ cho đề tài

# In[58]:


# chọn cột quan trọng
keep_cols = [
    c
    for c in [
        "id",
        "name",
        "brand_clean",
        "brand_name",
        "price",
        "list_price",
        "discount_percent",
        "image_path",
        "thumbnail_url",
        "rating_average",
        "seller_product_id",
        "quantity_sold_value"
    ]
    if c in df_clean.columns
]
df_core = df_clean[keep_cols].copy()

# --- tạo cột brand: ưu tiên brand_clean nếu có ---
if "brand_clean" in df_core.columns:
    df_core["brand"] = df_core["brand_clean"]
elif "brand_name" in df_core.columns:
    df_core["brand"] = df_core["brand_name"]

# --- bỏ hẳn brand_name & brand_clean để không trùng lặp ---
df_core = df_core.drop(
    columns=[c for c in ["brand_clean", "brand_name"] if c in df_core.columns]
)

# --- chuẩn hóa số cho price ---
if "price" in df_core.columns:
    df_core["price"] = pd.to_numeric(df_core["price"].map(to_num), errors="coerce")

# --- loại bỏ list_price và discount_percent nếu toàn NaN/0 ---
for col in ["list_price", "discount_percent"]:
    if col in df_core.columns:
        if df_core[col].isna().all() or (df_core[col].fillna(0) == 0).all():
            df_core = df_core.drop(columns=[col])
            print(f"Đã drop {col} vì toàn NaN/0")

# --- loại bản ghi không có image_path ---
if "image_path" in df_core.columns:
    before = len(df_core)
    df_core = df_core[df_core["image_path"].astype(str).str.strip().ne("")]
    print("Loại hàng thiếu image_path:", before - len(df_core))

# --- drop trùng theo id nếu có ---
if "id" in df_core.columns:
    before = len(df_core)
    df_core = df_core.drop_duplicates(subset=["id"])
    print("Drop duplicates theo id:", before - len(df_core))

# --- sắp xếp cột chính ---
preferred = [
    c for c in ["id", "name", "brand", "image_path", "thumbnail_url"] if c in df_core.columns
]
rest = [c for c in df_core.columns if c not in preferred]
df_core = df_core[preferred + rest]

print("Kích thước sau rút gọn:", df_core.shape)
df_core.head(10)


# In[59]:


df = df_core.copy()


# In[60]:


df.to_excel("../data_cleaned/speaker.xlsx", index=False)

