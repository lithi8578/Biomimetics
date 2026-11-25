"""輻輳幾何：距離 ↔ 左右眼視線角。

座標系（俯視，只考慮水平面）：
- 原點在兩眼中點，x 向右、z 向前，單位公尺。
- 左眼在 x = -b/2，右眼在 x = +b/2，b 為瞳距。
- 視線角為水平角度，向右為正，單位度。
- 輻輳角 = 左眼視線角 - 右眼視線角（看越近越大）。
- 「目標距離」是目標到兩眼中點的直線距離；「方向角」是目標相對正前方的水平角度。

反算一律建議用 demand_from_angles（直接得到屈光度），
因為雜訊大時輻輳角可能變成負的，距離會變成無限大或沒有定義，但屈光度仍是有限的數字。
"""
import numpy as np

IPD_M = 0.064  # 預設瞳距 64 mm


def eye_angles(dist_m, version_deg=0.0, ipd_m=IPD_M):
    """由目標距離與方向角，算出左右眼的視線角（度）。"""
    d = np.asarray(dist_m, dtype=float)
    phi = np.radians(version_deg)
    x = d * np.sin(phi)
    z = d * np.cos(phi)
    theta_l = np.degrees(np.arctan2(x + ipd_m / 2, z))
    theta_r = np.degrees(np.arctan2(x - ipd_m / 2, z))
    return theta_l, theta_r


def vergence_deg(theta_l, theta_r):
    """輻輳角（度）。"""
    return np.asarray(theta_l, dtype=float) - np.asarray(theta_r, dtype=float)


def demand_from_angles(theta_l, theta_r, ipd_m=IPD_M):
    """由左右眼視線角直接算屈光需求 q = 1/距離（D）。

    兩條視線的交點：z = b / (tan θL - tan θR)，x = z · (tan θL + tan θR) / 2，
    所以 1/d = (tan θL - tan θR) / (b · sqrt(1 + m²))，m = (tan θL + tan θR) / 2。
    對輻輳角是線性的；雜訊造成負輻輳時回傳負值（代表「比無限遠還遠」），由下游決定怎麼截斷。
    """
    tl = np.tan(np.radians(theta_l))
    tr = np.tan(np.radians(theta_r))
    m = (tl + tr) / 2
    return (tl - tr) / (ipd_m * np.sqrt(1 + m * m))


def distance_from_angles(theta_l, theta_r, ipd_m=IPD_M):
    """由左右眼視線角算目標距離（m）。輻輳角 ≤ 0 時回傳 NaN。"""
    q = np.asarray(demand_from_angles(theta_l, theta_r, ipd_m), dtype=float)
    out = np.full(q.shape, np.nan)
    np.divide(1.0, q, out=out, where=q > 0)
    return out if out.ndim else float(out)


def version_from_angles(theta_l, theta_r):
    """由左右眼視線角算目標方向角（度）。"""
    tl = np.tan(np.radians(theta_l))
    tr = np.tan(np.radians(theta_r))
    return np.degrees(np.arctan((tl + tr) / 2))


def midline_distance(verg_deg, ipd_m=IPD_M):
    """技術報告的公式 z = b / [2 tan(θ/2)]，只在目標位於正前方時精確。"""
    return ipd_m / (2 * np.tan(np.radians(verg_deg) / 2))
