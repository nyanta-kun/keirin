"""roi_summary（H-4: ブートストラップCI＋最大払戻除去）の純粋テスト。"""
import importlib

# scripts/ は conftest で path 追加済
roi = importlib.import_module("roi_robustness_wt")


def test_roi_summary_basic():
    # 投資一定（各300円）、払戻 [0,600,0,1200]
    s = roi.roi_summary([0, 600, 0, 1200], [300, 300, 300, 300], seed=1)
    assert s["n"] == 4
    assert s["hits"] == 2
    assert abs(s["hit_rate"] - 0.5) < 1e-9
    assert abs(s["roi"] - 1.5) < 1e-9            # 1800 / 1200
    assert abs(s["roi_ex_max"] - (600 / 900)) < 1e-9   # 1200除去
    assert abs(s["roi_ex_top2"] - 0.0) < 1e-9          # 1200,600除去 → 0/600
    assert abs(s["median_hit"] - 900.0) < 1e-9         # median(600,1200)
    # CI は点推定を含む向き（lo <= roi <= hi）
    assert s["ci_lo"] <= s["roi"] <= s["ci_hi"]


def test_roi_summary_variable_bets():
    # 小フィールドで投資可変（点数<3）
    s = roi.roi_summary([0, 500], [100, 300], seed=1)
    assert abs(s["roi"] - (500 / 400)) < 1e-9


def test_roi_summary_empty():
    s = roi.roi_summary([], [])
    assert s["n"] == 0 and s["roi"] == 0.0


def test_roi_summary_all_miss():
    s = roi.roi_summary([0, 0, 0], [300, 300, 300], seed=1)
    assert s["roi"] == 0.0
    assert s["hits"] == 0
    assert s["median_hit"] == 0.0
