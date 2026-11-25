"""資料介面：CSV 欄位定義與讀寫（0-3 介面規格的程式版本）。

兩種檔案，用 t_ms 對齊：
- 量測檔（*.csv）：模擬眼動儀輸出＋下游模組各自填的欄位。這是控制程式唯一能看到的資料。
- 真值檔（*_truth.csv）：腳本的標準答案，只給指標計算（1-7）與畫圖用，控制程式不可讀。

慣例：
- 無效的數值寫成空白，讀進來是 NaN。一定要先看 valid_L / valid_R，再用視線角。
- 尚未計算的下游欄位也是空白。每個模組讀前面的欄位、填自己的欄位，不改別人的欄位。
"""
import csv

import numpy as np

# (欄位名稱, 單位, 說明, 由誰填)
MEASURED_COLUMNS = [
    ("t_ms", "ms", "時間戳", "1-1 產生器"),
    ("theta_L_deg", "deg", "左眼水平視線角，向右為正；無效時空白", "1-1 產生器"),
    ("theta_R_deg", "deg", "右眼水平視線角，向右為正；無效時空白", "1-1 產生器"),
    ("valid_L", "0/1", "左眼這一筆有沒有量到", "1-1 產生器"),
    ("valid_R", "0/1", "右眼這一筆有沒有量到", "1-1 產生器"),
    ("confidence", "0-1", "兩眼可信度取較小者；無效為 0", "1-1 產生器"),
    ("dist_est_m", "m", "估計距離", "距離估算"),
    ("demand_D", "D", "屈光需求（鏡片附加度數）", "1-2"),
    ("state", "-", "FAR / MID / NEAR / HOLD / MANUAL", "1-5"),
    ("cmd_D", "D", "命令焦度", "1-5"),
    ("lens_D", "D", "鏡片實際焦度", "1-6"),
]

TRUTH_COLUMNS = [
    ("t_ms", "ms", "時間戳（與量測檔對齊）"),
    ("target_dist_m", "m", "使用者正在看的目標距離"),
    ("demand_true_D", "D", "真實屈光需求 = 1 / 目標距離"),
    ("version_true_deg", "deg", "目標方向角，向右為正"),
    ("theta_L_true_deg", "deg", "左眼真實視線角（含掃視過程，不含量測雜訊）"),
    ("theta_R_true_deg", "deg", "右眼真實視線角"),
    ("eye_event", "-", "fix 注視 / sac 掃視 / verg 輻輳移動 / blink 眨眼 / loss 追蹤遺失"),
    ("zone_intent", "-", "腳本預期的焦度區段：FAR / MID / NEAR；FAR/MID 表示兩者皆可"),
    ("section_id", "-", "區段編號，zone_intent 改變時才加 1；同一區段內的切換都算不必要切換"),
    ("target_id", "-", "注視目標編號，每次換目標（含同深度掃視）加 1"),
]

MEASURED_NAMES = [c[0] for c in MEASURED_COLUMNS]
TRUTH_NAMES = [c[0] for c in TRUTH_COLUMNS]

STRING_COLUMNS = {"state", "eye_event", "zone_intent"}
INT_COLUMNS = {"valid_L", "valid_R", "section_id", "target_id"}

_FORMATS = {
    "t_ms": "{:.6g}",
    "confidence": "{:.3f}",
}


def _fmt(name, value):
    if name in STRING_COLUMNS:
        return "" if value is None else str(value)
    if name in INT_COLUMNS:
        return str(int(value))
    v = float(value)
    if np.isnan(v):
        return ""
    return _FORMATS.get(name, "{:.6f}").format(v)


def write_csv(path, data, columns=MEASURED_NAMES):
    """把 {欄位: 陣列} 寫成 CSV。data 裡沒有的欄位寫空白。"""
    n = len(data[columns[0]])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        present = [c for c in columns if c in data]
        for i in range(n):
            row = []
            for c in columns:
                row.append(_fmt(c, data[c][i]) if c in present else "")
            w.writerow(row)


def read_csv(path):
    """讀 CSV 成 {欄位: numpy 陣列}。數值欄位的空白變成 NaN；旗標欄位為 int。"""
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        cols = {h: [] for h in header}
        for row in r:
            for h, v in zip(header, row):
                cols[h].append(v)
    out = {}
    for h, vals in cols.items():
        if h in STRING_COLUMNS:
            out[h] = np.array(vals, dtype=object)
        elif h in INT_COLUMNS:
            out[h] = np.array([int(v) if v != "" else 0 for v in vals], dtype=int)
        else:
            out[h] = np.array([float(v) if v != "" else np.nan for v in vals], dtype=float)
    return out
