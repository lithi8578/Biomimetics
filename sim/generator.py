"""合成情境產生器（1-1）：把情境腳本轉成「模擬眼動儀」的輸出，取代我們沒有的眼動儀硬體。

流程：
1. 真值：依腳本算出每個取樣時刻的目標距離，再用輻輳幾何反推兩眼視線角。
   換目標時眼睛不是瞬間到位，而是做一次掃視（20–80 ms）；
   水平掃視期間兩眼不完全同步，會出現短暫的發散（saccade_vergence_k）。
2. 量測：真值視線角＋高斯雜訊＋固定偏差；眨眼、追蹤遺失、隨機遺失的樣本標成無效。

用法（在「仿生科技」資料夾執行）：
    python -m sim.generator --scenario all --preset mid --seed 0 --out-dir out
"""
import argparse
import os
from dataclasses import dataclass, replace

import numpy as np

from . import geometry, interface
from .scenarios import REQUIRED, get_scenario


@dataclass(frozen=True)
class NoiseConfig:
    fs_hz: float = 100.0  # 取樣率
    sigma_deg: float = 1.0  # 每眼視線角的高斯雜訊標準差
    bias_deg: tuple = (0.0, 0.0)  # 左右眼固定偏差（校正誤差）
    saccade_vergence_k: float = 0.1  # 掃視中的暫時發散 = k × 掃視幅度（度）
    vergence_ms: float = 0.0  # 輻輳移動時間；0 表示和掃視同時完成
    blink_rate_per_min: float = 0.0  # 隨機眨眼頻率（腳本裡的眨眼另計）
    blink_ms: tuple = (100.0, 300.0)  # 隨機眨眼長度範圍
    blink_margin_ms: float = 50.0  # 眨眼前後眼皮半閉：可信度下降、雜訊變 3 倍
    dropout_rate: float = 0.0  # 隨機遺失的樣本比例（每眼獨立）
    dropout_burst_ms: tuple = (10.0, 30.0)  # 每次隨機遺失的長度範圍
    ipd_m: float = geometry.IPD_M
    seed: int = 0


PRESETS = {
    "ideal": dict(sigma_deg=0.0, saccade_vergence_k=0.0),  # 理想眼睛＋完美眼動儀，用來除錯
    "low": dict(sigma_deg=0.5),
    "mid": dict(sigma_deg=1.0),
    "high": dict(sigma_deg=2.0),
}


def make_config(preset="mid", **overrides):
    return replace(NoiseConfig(), **{**PRESETS[preset], **overrides})


def saccade_duration_ms(amp_deg):
    """掃視時間的主序列近似：約 21 ms + 2.2 ms/度，限制在 20–80 ms。"""
    return float(np.clip(21.0 + 2.2 * amp_deg, 20.0, 80.0))


def _min_jerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _runs(mask):
    """回傳 mask 中連續為 True 的區段 [(起, 迄), ...]（迄不含）。"""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def build_truth(scn, cfg):
    """依腳本算出真值時間序列（還沒有量測雜訊）。"""
    n = int(round(scn.duration_s * cfg.fs_hz))
    t = np.arange(n) / cfg.fs_hz
    targets = sorted(scn.targets, key=lambda tg: tg.t_s)
    if targets[0].t_s != 0:
        raise ValueError("第一個 Target 必須從 0 秒開始")
    starts = np.array([tg.t_s for tg in targets])
    idx = np.searchsorted(starts, t + 1e-9, side="right") - 1

    dist = np.empty(n)
    version = np.empty(n)
    zone = np.empty(n, dtype=object)
    section = np.empty(n, dtype=int)
    sec, prev_zone = 0, targets[0].zone
    for i, tg in enumerate(targets):
        if tg.zone != prev_zone:
            sec += 1
            prev_zone = tg.zone
        m = idx == i
        dist[m] = tg.distance_at(t[m])
        version[m] = tg.version_deg
        zone[m] = tg.zone
        section[m] = sec

    goal_l, goal_r = geometry.eye_angles(dist, version, cfg.ipd_m)
    v_goal = (goal_l + goal_r) / 2  # 共軛（兩眼同向）成分
    g_goal = goal_l - goal_r  # 輻輳成分
    v = v_goal.copy()
    g = g_goal.copy()
    event = np.full(n, "fix", dtype=object)

    for i in range(1, len(targets)):
        k0 = int(np.searchsorted(t, targets[i].t_s - 1e-9))
        if k0 <= 0 or k0 >= n:
            continue
        k_end = int(np.searchsorted(t, targets[i + 1].t_s - 1e-9)) if i + 1 < len(targets) else n
        v0, g0 = v[k0 - 1], g[k0 - 1]
        d_l = (v_goal[k0] + g_goal[k0] / 2) - (v0 + g0 / 2)
        d_r = (v_goal[k0] - g_goal[k0] / 2) - (v0 - g0 / 2)
        amp = max(abs(d_l), abs(d_r))
        if amp < 0.01:
            continue
        ds = saccade_duration_ms(amp) / 1000
        dg = max(ds, cfg.vergence_ms / 1000)
        t0 = targets[i].t_s
        j = np.arange(k0, k_end)
        j = j[t[j] - t0 < dg]
        tau_s = (t[j] - t0) / ds
        tau_g = (t[j] - t0) / dg
        dv = abs(v_goal[k0] - v0)
        v[j] = v0 + (v_goal[j] - v0) * _min_jerk(tau_s)
        g[j] = (g0 + (g_goal[j] - g0) * _min_jerk(tau_g)
                - cfg.saccade_vergence_k * dv * np.sin(np.pi * np.clip(tau_s, 0, 1)))
        event[j[tau_s < 1]] = "sac"
        event[j[(tau_s >= 1) & (tau_g < 1)]] = "verg"

    for ev in scn.events:
        m = (t >= ev.t_s - 1e-9) & (t < ev.t_s + ev.dur_ms / 1000 - 1e-9)
        event[m] = ev.kind

    return {
        "t_ms": t * 1000,
        "target_dist_m": dist,
        "demand_true_D": 1.0 / dist,
        "version_true_deg": version,
        "theta_L_true_deg": v + g / 2,
        "theta_R_true_deg": v - g / 2,
        "eye_event": event,
        "zone_intent": zone,
        "section_id": section,
        "target_id": idx,
    }


def _random_blinks(rng, duration_s, cfg):
    """隨機眨眼：兩次之間至少隔 0.3 秒，平均頻率等於 blink_rate_per_min。"""
    if cfg.blink_rate_per_min <= 0:
        return []
    lo, hi = cfg.blink_ms
    min_gap = 0.3
    mean_gap = 60.0 / cfg.blink_rate_per_min - (lo + hi) / 2000 - min_gap
    out, t = [], 0.0
    while True:
        t += min_gap + rng.exponential(max(mean_gap, 1e-3))
        dur = rng.uniform(lo, hi)
        if t + dur / 1000 > duration_s:
            return out
        out.append((t, dur))
        t += dur / 1000


def _dropout_mask(rng, n, cfg):
    """隨機遺失：短暫的連續樣本遺失，總比例約為 dropout_rate。"""
    mask = np.zeros(n, bool)
    if cfg.dropout_rate <= 0:
        return mask
    lo, hi = cfg.dropout_burst_ms
    mean_len = max((lo + hi) / 2000 * cfg.fs_hz, 1.0)
    lam = -np.log(1 - cfg.dropout_rate)  # 修正重疊造成的低估
    nb = rng.poisson(lam * n / mean_len)
    starts = rng.integers(0, n, nb)
    lens = np.maximum(1, np.round(rng.uniform(lo, hi, nb) / 1000 * cfg.fs_hz)).astype(int)
    for s, L in zip(starts, lens):
        mask[s:s + L] = True
    return mask


def generate(scn, cfg=None):
    """產生 (量測, 真值) 兩份 {欄位: 陣列}。量測只含 1-1 負責的欄位。"""
    cfg = cfg or make_config()
    rng = np.random.default_rng(cfg.seed)
    truth = build_truth(scn, cfg)
    t = truth["t_ms"] / 1000
    n = len(t)

    blinks = [(ev.t_s, ev.dur_ms) for ev in scn.events if ev.kind == "blink"]
    rand_blinks = _random_blinks(rng, scn.duration_s, cfg)
    for s, dur in rand_blinks:
        truth["eye_event"][(t >= s) & (t < s + dur / 1000)] = "blink"
    blinks += rand_blinks

    inv_l = np.zeros(n, bool)
    inv_r = np.zeros(n, bool)
    margin = np.zeros(n, bool)
    mg = cfg.blink_margin_ms / 1000
    for s, dur in blinks:
        e = s + dur / 1000
        inside = (t >= s - 1e-9) & (t < e - 1e-9)
        inv_l |= inside
        inv_r |= inside
        margin |= ((t >= s - mg) & (t < s)) | ((t >= e) & (t < e + mg))
    for ev in scn.events:
        if ev.kind != "loss":
            continue
        m = (t >= ev.t_s - 1e-9) & (t < ev.t_s + ev.dur_ms / 1000 - 1e-9)
        if "L" in ev.eyes:
            inv_l |= m
        if "R" in ev.eyes:
            inv_r |= m
    inv_l |= _dropout_mask(rng, n, cfg)
    inv_r |= _dropout_mask(rng, n, cfg)

    scale = np.where(margin, 3.0, 1.0)
    th_l = truth["theta_L_true_deg"] + cfg.bias_deg[0] + rng.normal(0, 1, n) * cfg.sigma_deg * scale
    th_r = truth["theta_R_true_deg"] + cfg.bias_deg[1] + rng.normal(0, 1, n) * cfg.sigma_deg * scale
    th_l[inv_l] = np.nan
    th_r[inv_r] = np.nan

    def conf(inv):
        c = np.clip(rng.normal(0.95, 0.02, n), 0.0, 1.0)
        c[margin] = rng.uniform(0.3, 0.6, int(margin.sum()))
        c[inv] = 0.0
        return c

    measured = {
        "t_ms": truth["t_ms"],
        "theta_L_deg": th_l,
        "theta_R_deg": th_r,
        "valid_L": (~inv_l).astype(int),
        "valid_R": (~inv_r).astype(int),
        "confidence": np.minimum(conf(inv_l), conf(inv_r)),
    }
    return measured, truth


def save(scn, cfg, out_dir, tag):
    measured, truth = generate(scn, cfg)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{scn.name}_{tag}")
    interface.write_csv(base + ".csv", measured, interface.MEASURED_NAMES)
    interface.write_csv(base + "_truth.csv", truth, interface.TRUTH_NAMES)
    return base


def main(argv=None):
    p = argparse.ArgumentParser(description="合成情境產生器（工作包 A，1-1）")
    p.add_argument("--scenario", default="all", help="near_far / same_depth_scan / blink_loss / threshold_jitter / all")
    p.add_argument("--preset", default="mid", choices=list(PRESETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, help="覆寫每眼雜訊標準差（度）")
    p.add_argument("--blink-rate", type=float, help="隨機眨眼次數／分鐘")
    p.add_argument("--dropout", type=float, help="隨機遺失比例，例如 0.05")
    p.add_argument("--fs", type=float, help="取樣率（Hz）")
    p.add_argument("--out-dir", default="out")
    a = p.parse_args(argv)

    over = {"seed": a.seed}
    if a.sigma is not None:
        over["sigma_deg"] = a.sigma
    if a.blink_rate is not None:
        over["blink_rate_per_min"] = a.blink_rate
    if a.dropout is not None:
        over["dropout_rate"] = a.dropout
    if a.fs is not None:
        over["fs_hz"] = a.fs
    cfg = make_config(a.preset, **over)
    names = REQUIRED if a.scenario == "all" else [a.scenario]
    for name in names:
        base = save(get_scenario(name), cfg, a.out_dir, f"{a.preset}_s{a.seed}")
        print("寫出", base + ".csv", "與", base + "_truth.csv")


if __name__ == "__main__":
    main()
