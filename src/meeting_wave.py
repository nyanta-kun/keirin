"""開催（会場×日）を netkeirin 入稿の「波」へ振り分ける（2026-08-07 新設）。

## なぜ分けるのか

三連複の板は**時計時刻ではなく発走までの近さ**で埋まる。朝8時台の未確定率（実測・
2026-07-16以降・7車）:

| 開催種別（第1R発走） | 朝8:12 | 10:00 | 12:00 | 14:00 | 18:00 | 20:00 |
|---|---|---|---|---|---|---|
| モーニング（8時）    | **0.7%** | 0.0% | — | — | — | — |
| デイ（10-11時）      | 8.6% | 1.7% | 0.2% | 0.0% | — | — |
| ナイター（15-16時）  | 30.8% | 13.4% | **5.3%** | 2.0% | 0.0% | 0.0% |
| ミッドナイト（20時） | **63.4%** | 50.1% | 41.5% | 31.5% | **10.8%** | 2.4% |

推奨レースで「買う点すべてにオッズがある」割合は モーニング 100% / デイ 91.1% /
ナイター 65.2% / **ミッドナイト 17.5%**。夜の開催を朝に入稿すると、
[[keirin_netkeirin_gami_allocation]] の傾斜配分がほぼ効かない。

netkeirin は**公開後の差し替えができない**ので、板が育ってから入稿するしかない。

## 波の決め方

**開催 = (開催日, 会場)**。その開催の**第1レースの発走時刻**だけで決まる。
レース個別の発走時刻では分けない（同じ開催の中で商品の出るタイミングが
バラバラになると分かりにくい、というユーザー判断・2026-08-07）。

実測では第1R発走は 8 / 10 / 11 / 15 / 16 / 20 時の6通りしか無く、
競輪の モーニング / デイ / ナイター / ミッドナイト にそのまま対応する。
閾値は境界（9時台・12-14時台・17-19時台）が空いているので余裕がある。

⚠️ **発走時刻が取れない開催は WAVE_MORNING**（安全側）。
   分からないことを理由に入稿を落とすと、商品が黙って消える。
"""
from __future__ import annotations

WAVE_MORNING = "morning"   # モーニング・デイ（第1R < 12時）→ 朝の日次バッチで入稿
WAVE_NOON = "noon"         # ナイター（12時 <= 第1R < 18時）→ 昼に入稿
WAVE_NIGHT = "night"       # ミッドナイト（第1R >= 18時）→ 夕方に入稿

WAVES = (WAVE_MORNING, WAVE_NOON, WAVE_NIGHT)

# 第1R発走時刻（時）の境界。実測の第1R発走は 8/10/11/15/16/20 時のみで、
# 9時台・12〜14時台・17〜19時台は空いている＝境界に余裕がある。
NOON_FROM_HOUR = 12
NIGHT_FROM_HOUR = 18

WAVE_LABEL_JP = {
    WAVE_MORNING: "朝（モーニング・デイ）",
    WAVE_NOON: "昼（ナイター）",
    WAVE_NIGHT: "夕方（ミッドナイト）",
}


def wave_of_first_hour(first_race_hour: int | float | None) -> str:
    """開催の第1R発走時刻（時）から入稿の波を返す。

    first_race_hour が None（発走時刻不明）なら WAVE_MORNING を返す。
    """
    if first_race_hour is None:
        return WAVE_MORNING
    h = float(first_race_hour)
    if h >= NIGHT_FROM_HOUR:
        return WAVE_NIGHT
    if h >= NOON_FROM_HOUR:
        return WAVE_NOON
    return WAVE_MORNING


def waves_due_by(session_wave: str) -> tuple[str, ...]:
    """その回の入稿で**対象にすべき**波（自分の波 + 取りこぼした過去の波）。

    背景（2026-08-08 レビュー指摘 M-6）:
      波は毎回 `wt_races.start_at` から**都度**再計算される。そのため発走時刻が
      **前倒しに訂正**されると、開催が通過済みの波へ移ってしまい、
      「自分の波と一致するもの」だけを見る実装では**その日どの回からも
      入稿されずに終わる**。

      例: 当初12:30発走（朝は noon 判定で対象外）→ 昼までに10:30へ訂正
          → 昼の時点で morning と再判定 → `== "noon"` のフィルタから外れる。

      逆（後ろへ動く）は後続の回が拾うので穴にならない。前倒しだけが落ちる。

    そこで自分の波より**前の波も対象に含める**。二重入稿は
    `_already_submitted()` が、終わったレースへの入稿は `_load_started_races()` が
    それぞれ止めるので、拾い直しても副作用は無い。

    ⚠️ 逆に「後の波」を含めてはいけない。板が育つのを待つという波の目的そのもの
       （ミッドナイトを朝に出すと傾斜配分がほぼ効かない）が壊れる。
    """
    if session_wave not in WAVES:
        return (session_wave,)
    return WAVES[: WAVES.index(session_wave) + 1]


def parse_wave(value: str | None) -> str | None:
    """CLI 引数などの文字列を波名へ正規化する。未知なら None。"""
    if not value:
        return None
    v = str(value).strip().lower()
    return v if v in WAVES else None
