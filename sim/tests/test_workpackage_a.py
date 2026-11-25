"""工作包 A 的驗收測試（0-2 與 1-1）。

在「仿生科技」資料夾執行：
    python -m unittest sim.tests.test_workpackage_a -v
"""
import os
import tempfile
import unittest

import numpy as np

from sim import geometry, interface
from sim.generator import generate, make_config, PRESETS
from sim.scenarios import REQUIRED, get_scenario, stationary, Event


def _run_lengths(mask):
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    d = np.diff(m.astype(int))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


class TestGeometry(unittest.TestCase):
    """1-1 驗收：報告的兩個例子，正算與反算。"""

    def test_report_examples(self):
        for dist, verg in [(0.50, 7.32), (2.00, 1.83)]:
            l, r = geometry.eye_angles(dist)
            self.assertAlmostEqual(geometry.vergence_deg(l, r), verg, delta=0.005)
            self.assertAlmostEqual(geometry.midline_distance(verg), dist, delta=dist * 0.01)
            self.assertAlmostEqual(geometry.distance_from_angles(l, r), dist, places=9)

    def test_roundtrip_with_lateral_gaze(self):
        d = np.array([0.25, 0.35, 0.7, 1.0, 2.0, 5.0])
        for ver in [-20, -8, 0, 8, 20]:
            l, r = geometry.eye_angles(d, ver)
            np.testing.assert_allclose(geometry.distance_from_angles(l, r), d, rtol=1e-9)
            np.testing.assert_allclose(geometry.version_from_angles(l, r), ver, atol=1e-9)
            np.testing.assert_allclose(geometry.demand_from_angles(l, r), 1 / d, rtol=1e-9)

    def test_negative_vergence_gives_nan_distance(self):
        self.assertTrue(np.isnan(geometry.distance_from_angles(-1.0, 1.0)))
        self.assertLess(geometry.demand_from_angles(-1.0, 1.0), 0)


class TestScenarios(unittest.TestCase):
    """0-2 驗收：4 支必做腳本都有真值時間序列。"""

    def test_all_required_generate_with_every_preset(self):
        self.assertEqual(len(REQUIRED), 4)
        for name in REQUIRED:
            for preset in PRESETS:
                scn = get_scenario(name)
                meas, truth = generate(scn, make_config(preset))
                n = int(round(scn.duration_s * 100))
                for col in ["t_ms", "theta_L_deg", "theta_R_deg", "valid_L", "valid_R", "confidence"]:
                    self.assertEqual(len(meas[col]), n, (name, preset, col))
                for col in interface.TRUTH_NAMES:
                    self.assertEqual(len(truth[col]), n, (name, preset, col))
                self.assertTrue(np.all(np.diff(meas["t_ms"]) > 0))
                self.assertTrue(np.all(np.isfinite(truth["demand_true_D"])))
                np.testing.assert_allclose(truth["demand_true_D"], 1 / truth["target_dist_m"])

    def test_sections(self):
        # ① 7 段距離、② 只有一次合理切換、④ 中段可 FAR 或 MID
        self.assertEqual(generate(get_scenario("near_far"))[1]["section_id"].max(), 6)
        self.assertEqual(generate(get_scenario("same_depth_scan"))[1]["section_id"].max(), 1)
        _, tr = generate(get_scenario("threshold_jitter"))
        mid = tr["zone_intent"] == "FAR/MID"
        self.assertTrue(np.all(tr["demand_true_D"][mid] >= 0.8 - 1e-9))
        self.assertTrue(np.all(tr["demand_true_D"][mid] <= 1.2 + 1e-9))

    def test_same_depth_scan_keeps_demand_constant(self):
        _, tr = generate(get_scenario("same_depth_scan"))
        for sec, d in [(0, 1 / 0.70), (1, 1 / 0.35)]:
            np.testing.assert_allclose(tr["demand_true_D"][tr["section_id"] == sec], d)
        self.assertGreater(tr["target_id"].max(), 60)  # 確實有很多次掃視


class TestGenerator(unittest.TestCase):
    """1-1 驗收：無雜訊誤差 < 1%、雜訊統計與設定值相差 < 10%。"""

    def test_noise_free_distance_error_below_1_percent(self):
        for name in REQUIRED:
            meas, tr = generate(get_scenario(name), make_config("ideal"))
            fix = (tr["eye_event"] == "fix") & (meas["valid_L"] == 1) & (meas["valid_R"] == 1)
            d = geometry.distance_from_angles(meas["theta_L_deg"][fix], meas["theta_R_deg"][fix])
            err = np.abs(d / tr["target_dist_m"][fix] - 1)
            self.assertLess(err.max(), 0.01, name)

    def test_noise_sigma_matches_setting(self):
        scn = stationary(600.0)
        for sigma in [0.5, 1.0, 2.0]:
            meas, tr = generate(scn, make_config("mid", sigma_deg=sigma, seed=1))
            for eye in "LR":
                e = meas[f"theta_{eye}_deg"] - tr[f"theta_{eye}_true_deg"]
                s = np.nanstd(e)
                self.assertLess(abs(s / sigma - 1), 0.10, (sigma, eye, s))
                self.assertLess(abs(np.nanmean(e)), 0.1 * sigma)

    def test_bias(self):
        meas, tr = generate(stationary(60.0), make_config("ideal", bias_deg=(0.3, -0.2)))
        np.testing.assert_allclose(meas["theta_L_deg"] - tr["theta_L_true_deg"], 0.3, atol=1e-9)
        np.testing.assert_allclose(meas["theta_R_deg"] - tr["theta_R_true_deg"], -0.2, atol=1e-9)

    def test_scripted_blinks_and_losses(self):
        scn = get_scenario("blink_loss")
        meas, tr = generate(scn, make_config("mid"))
        t = meas["t_ms"]
        for ev in scn.events:
            m = (t >= ev.t_s * 1000 - 1e-6) & (t < ev.t_s * 1000 + ev.dur_ms - 1e-6)
            self.assertAlmostEqual(m.sum(), ev.dur_ms / 10, delta=1)
            exp_l = 0 if ev.kind == "blink" or "L" in ev.eyes else 1
            exp_r = 0 if ev.kind == "blink" or "R" in ev.eyes else 1
            self.assertTrue(np.all(meas["valid_L"][m] == exp_l), ev)
            self.assertTrue(np.all(meas["valid_R"][m] == exp_r), ev)
        inv = (meas["valid_L"] == 0)
        self.assertTrue(np.all(np.isnan(meas["theta_L_deg"][inv])))
        self.assertTrue(np.all(meas["confidence"][inv] == 0))

    def test_random_blink_rate_and_duration(self):
        meas, tr = generate(stationary(3600.0), make_config("mid", blink_rate_per_min=15, seed=2))
        lens = _run_lengths(tr["eye_event"] == "blink")
        rate = len(lens) / 60.0
        self.assertLess(abs(rate / 15 - 1), 0.10, rate)
        self.assertTrue(np.all((lens >= 10 - 1) & (lens <= 30 + 1)))  # 100–300 ms

    def test_dropout_rate(self):
        meas, _ = generate(stationary(600.0), make_config("mid", dropout_rate=0.05, seed=3))
        for eye in "LR":
            frac = 1 - meas[f"valid_{eye}"].mean()
            self.assertLess(abs(frac / 0.05 - 1), 0.10, (eye, frac))

    def test_saccade_duration_20_to_80_ms(self):
        _, tr = generate(get_scenario("same_depth_scan"), make_config("mid"))
        lens = _run_lengths(tr["eye_event"] == "sac")
        self.assertGreater(len(lens), 60)
        self.assertTrue(np.all((lens >= 2) & (lens <= 8)), lens)

    def test_saccade_vergence_transient(self):
        # 同深度掃視時，掃視中的真實輻輳角會短暫變小，注視時回到幾何值
        _, tr = generate(get_scenario("same_depth_scan"), make_config("mid"))
        verg = tr["theta_L_true_deg"] - tr["theta_R_true_deg"]
        l, r = geometry.eye_angles(tr["target_dist_m"], tr["version_true_deg"])
        geo = l - r
        sac = tr["eye_event"] == "sac"
        fix = tr["eye_event"] == "fix"
        self.assertLess((verg - geo)[sac].min(), -0.5)
        np.testing.assert_allclose(verg[fix], geo[fix], atol=1e-9)

    def test_reproducible(self):
        scn = get_scenario("near_far")
        a, _ = generate(scn, make_config("mid", seed=7))
        b, _ = generate(scn, make_config("mid", seed=7))
        c, _ = generate(scn, make_config("mid", seed=8))
        np.testing.assert_array_equal(a["theta_L_deg"], b["theta_L_deg"])
        self.assertFalse(np.allclose(a["theta_L_deg"], c["theta_L_deg"]))

    def test_csv_roundtrip(self):
        meas, tr = generate(get_scenario("blink_loss"), make_config("mid", dropout_rate=0.02))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.csv")
            interface.write_csv(p, meas, interface.MEASURED_NAMES)
            back = interface.read_csv(p)
            self.assertEqual(list(back), interface.MEASURED_NAMES)
            np.testing.assert_allclose(back["theta_L_deg"], meas["theta_L_deg"], atol=1e-6)
            np.testing.assert_array_equal(back["valid_R"], meas["valid_R"])
            self.assertTrue(np.all(np.isnan(back["demand_D"])))  # 下游欄位先空白
            pt = os.path.join(d, "t.csv")
            interface.write_csv(pt, tr, interface.TRUTH_NAMES)
            bt = interface.read_csv(pt)
            np.testing.assert_array_equal(bt["eye_event"], tr["eye_event"])
            np.testing.assert_array_equal(bt["section_id"], tr["section_id"])


if __name__ == "__main__":
    unittest.main()
