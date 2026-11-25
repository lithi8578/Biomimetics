"""測試情境腳本（0-2）：「什麼時間、使用者在看多遠的東西」。

每支腳本由兩種元素組成：
- Target：從 t_s 秒開始，使用者看距離 dist_m、方向 version_deg 的目標，直到下一個 Target。
  zone 是這段時間「預期」的焦度區段，用來判斷切換是否必要。
- Event：眨眼（blink，兩眼同時無效）或追蹤遺失（loss，可指定單眼）。

焦度區段的參考值（MVP 三段式）：FAR ≈ 0.5 D（2 m）、MID ≈ 1.43 D（70 cm）、NEAR ≈ 2.86 D（35 cm）。
"""
from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np

Distance = Union[float, Callable[[np.ndarray], np.ndarray]]


@dataclass
class Target:
    t_s: float
    dist_m: Distance  # 定值，或「時間（秒）→ 距離」的函式
    version_deg: float = 0.0
    zone: str = "FAR"

    def distance_at(self, t):
        t = np.asarray(t, dtype=float)
        if callable(self.dist_m):
            return np.asarray(self.dist_m(t), dtype=float)
        return np.full(t.shape, float(self.dist_m))


@dataclass
class Event:
    t_s: float
    dur_ms: float
    kind: str = "blink"  # "blink" 或 "loss"
    eyes: str = "LR"  # loss 才用得到："L"、"R" 或 "LR"


@dataclass
class Scenario:
    name: str
    title: str
    purpose: str
    duration_s: float
    targets: list
    events: list = field(default_factory=list)


def near_far():
    """① 近遠切換：35 / 70 / 200 cm 之間切換，停留 5 秒與 3 秒兩種。"""
    plan = [
        (0, 2.00, "FAR"),
        (5, 0.35, "NEAR"),
        (10, 0.70, "MID"),
        (15, 2.00, "FAR"),
        (20, 0.70, "MID"),
        (23, 0.35, "NEAR"),
        (26, 2.00, "FAR"),
    ]
    return Scenario(
        name="near_far",
        title="① 近遠切換",
        purpose="該切的時候切得夠不夠快、夠不夠準（延遲、殘餘離焦）",
        duration_s=30.0,
        targets=[Target(t, d, 0.0, z) for t, d, z in plan],
    )


def same_depth_scan():
    """② 同深度左右掃視：70 cm 看螢幕四處掃，之後 35 cm 逐行閱讀。全程只有一次合理切換。"""
    targets = [Target(0.0, 0.70, 0.0, "MID")]
    versions = [-12, 4, 15, -6, 9, -15, 2, 12, -9, 6, -3, 14, -13, 0,
                8, -10, 11, -4, 3, -14, 7, 13, -7, 1, 10, -11, 5, -1]
    durs = [0.45, 0.30, 0.60, 0.35, 0.50, 0.25, 0.40]
    t, i = 2.0, 0
    while t < 16.0:
        targets.append(Target(round(t, 3), 0.70, versions[i % len(versions)], "MID"))
        t += durs[i % len(durs)]
        i += 1
    # 16 秒起改看 35 cm 的書：每行從 -8° 往右 2° 一跳，跳到 +8° 後回到行首
    line = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    read_durs = [0.25, 0.22, 0.28, 0.24, 0.30, 0.21, 0.26, 0.23, 0.27]
    t = 16.0
    while t < 30.0:
        for v, dd in zip(line, read_durs):
            if t >= 30.0:
                break
            targets.append(Target(round(t, 3), 0.35, v, "NEAR"))
            t += dd
    return Scenario(
        name="same_depth_scan",
        title="② 同深度左右掃視",
        purpose="距離沒變時會不會亂切（不必要切換）",
        duration_s=30.0,
        targets=targets,
    )


def blink_loss():
    """③ 眨眼與資料遺失：各種長度的眨眼、單眼與雙眼遺失，最後追蹤全失效 3 秒。"""
    targets = [
        Target(0, 0.70, 0.0, "MID"),
        Target(8, 2.00, 0.0, "FAR"),
        Target(16, 0.35, 0.0, "NEAR"),
        Target(24, 0.70, 0.0, "MID"),
    ]
    events = [
        Event(2.0, 100), Event(3.5, 150), Event(5.0, 200), Event(6.5, 300),  # 正常眨眼
        Event(10.0, 500),  # 長眨眼，超過 300 ms
        Event(12.0, 1200, "loss", "LR"),  # 兩眼追蹤遺失 1.2 秒
        Event(15.9, 250),  # 換目標的同時眨眼（16 秒換到近距）
        Event(19.0, 400, "loss", "L"),  # 只有左眼遺失
        Event(21.0, 400, "loss", "R"),  # 只有右眼遺失
        Event(27.0, 3000, "loss", "LR"),  # 追蹤全失效 3 秒（給 1-5 HOLD 逾時與 1-9 故障案例）
    ]
    return Scenario(
        name="blink_loss",
        title="③ 眨眼與資料遺失",
        purpose="沒資料時會不會保持原狀；逾時後會不會退回預設焦度",
        duration_s=34.0,
        targets=targets,
        events=events,
    )


def threshold_jitter(center_D=1.0, amp_D=0.2):
    """④ 門檻附近抖動：5–25 秒目標在 center_D ± amp_D 之間緩慢晃動。

    center_D 預設 1.0 D（約 1 m，介於 FAR 0.5 D 與 MID 1.43 D 之間）。
    請和工作包 D（1-5）訂出的切換門檻對齊後再調整。
    """
    t0 = 5.0
    a = amp_D / 0.2

    def dist(t):
        tt = np.asarray(t, dtype=float) - t0
        q = center_D + a * (0.12 * np.sin(2 * np.pi * 0.31 * tt)
                            + 0.08 * np.sin(2 * np.pi * 0.73 * tt + 1.0))
        return 1.0 / q

    return Scenario(
        name="threshold_jitter",
        title="④ 門檻附近抖動",
        purpose="遲滯有沒有擋住來回跳動；中段 FAR 或 MID 都可以，但段內最多切換 1 次",
        duration_s=30.0,
        targets=[
            Target(0, 2.00, 0.0, "FAR"),
            Target(t0, dist, 0.0, "FAR/MID"),
            Target(25, 0.35, 0.0, "NEAR"),
        ],
    )


def stationary(duration_s=60.0, dist_m=0.70, version_deg=0.0, zone="MID"):
    """輔助腳本：一直看同一個目標。用於驗證雜訊統計、1-3 蒙地卡羅。"""
    return Scenario(
        name="stationary",
        title="固定注視",
        purpose="雜訊統計與誤差分析",
        duration_s=duration_s,
        targets=[Target(0.0, dist_m, version_deg, zone)],
    )


SCENARIOS = {
    "near_far": near_far,
    "same_depth_scan": same_depth_scan,
    "blink_loss": blink_loss,
    "threshold_jitter": threshold_jitter,
}
REQUIRED = list(SCENARIOS)


def get_scenario(name, **kwargs):
    if name == "stationary":
        return stationary(**kwargs)
    return SCENARIOS[name](**kwargs)
