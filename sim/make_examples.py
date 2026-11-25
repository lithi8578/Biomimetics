"""產生範例資料與檢查圖。

在「仿生科技」資料夾執行：
    python -m sim.make_examples
輸出到 sim/examples/：
- ideal/、mid/：4 支必做腳本的量測檔與真值檔（種子 0）
- fig_<腳本>.png：mid 雜訊下的真值與量測對照圖
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import geometry  # noqa: E402
from .generator import generate, make_config, save, _runs  # noqa: E402
from .scenarios import REQUIRED, get_scenario  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "examples")

ZONE_D = {"FAR": 0.5, "MID": 1 / 0.70, "NEAR": 1 / 0.35}


def plot(name, cfg, path):
    scn = get_scenario(name)
    meas, tr = generate(scn, cfg)
    t = meas["t_ms"] / 1000
    q_meas = geometry.demand_from_angles(meas["theta_L_deg"], meas["theta_R_deg"], cfg.ipd_m)

    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 1]})
    for kind, color in [("blink", "#f2c14e"), ("loss", "#e07a5f")]:
        for s, e in _runs(tr["eye_event"] == kind):
            for a in ax:
                a.axvspan(t[s], t[min(e, len(t) - 1)], color=color, alpha=0.25, lw=0)

    a = ax[0]
    a.plot(t, q_meas, ".", ms=1.5, color="#8d99ae", label="由量測視線角換算（每筆樣本）")
    a.plot(t, tr["demand_true_D"], color="#d62828", lw=1.8, label="真實屈光需求")
    for z, d in ZONE_D.items():
        a.axhline(d, color="#adb5bd", lw=0.6, ls="--")
        a.text(t[-1], d, f" {z}", va="center", fontsize=8, color="#6c757d")
    a.set_ylabel("屈光需求（D）")
    a.set_ylim(-1.5, 4.5)
    a.legend(loc="upper left", fontsize=8)
    a.set_title(f"{scn.title}（σ = {cfg.sigma_deg}°/眼，種子 {cfg.seed}）—— {scn.purpose}", fontsize=10)

    a = ax[1]
    a.plot(t, meas["theta_L_deg"], ".", ms=1.2, color="#5fa8d3", label="左眼量測")
    a.plot(t, meas["theta_R_deg"], ".", ms=1.2, color="#90be6d", label="右眼量測")
    a.plot(t, tr["theta_L_true_deg"], color="#1d3557", lw=1, label="左眼真值")
    a.plot(t, tr["theta_R_true_deg"], color="#386641", lw=1, label="右眼真值")
    a.set_ylabel("視線角（度）")
    a.legend(loc="upper left", fontsize=7, ncol=4)

    a = ax[2]
    a.plot(t, meas["confidence"], color="#495057", lw=0.8, label="可信度")
    a.plot(t, meas["valid_L"] * 0.5 - 0.7, color="#5fa8d3", lw=1, label="左眼有效")
    a.plot(t, meas["valid_R"] * 0.5 - 0.75, color="#90be6d", lw=1, label="右眼有效")
    a.set_ylim(-0.85, 1.1)
    a.set_yticks([])
    a.set_xlabel("時間（秒）　黃底：眨眼　紅底：追蹤遺失")
    a.legend(loc="lower left", fontsize=7, ncol=3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    for preset in ["ideal", "mid"]:
        cfg = make_config(preset, seed=0)
        for name in REQUIRED:
            save(get_scenario(name), cfg, os.path.join(OUT, preset), f"{preset}_s0")
    cfg = make_config("mid", seed=0)
    for name in REQUIRED:
        plot(name, cfg, os.path.join(OUT, f"fig_{name}.png"))
    print("範例輸出在", OUT)

    # 給下游的提醒數字：遠距時雜訊造成輻輳角 ≤ 0 的比例
    from .scenarios import stationary
    for sigma in [0.5, 1.0, 2.0]:
        m, _ = generate(stationary(600.0, 2.0, zone="FAR"), make_config("mid", sigma_deg=sigma, seed=0))
        v = m["theta_L_deg"] - m["theta_R_deg"]
        q = geometry.demand_from_angles(m["theta_L_deg"], m["theta_R_deg"])
        print(f"看 2 m、sigma = {sigma} 度：輻輳角不大於 0 的樣本 {np.mean(v <= 0) * 100:.1f}%，"
              f"單筆屈光需求標準差 {np.std(q):.2f} D")


if __name__ == "__main__":
    main()
