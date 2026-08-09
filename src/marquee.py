"""看板レース（売上が集まりやすいレース）の検出（2026-08-09 新設）。

## なぜ要るのか

2026-08-08 のレース単位分析で、**当日売上 5,060pt の 84% が「外れたレース」に
集中**していた。売れたのは全て看板レース（決勝・特選クラス）で、逆に高配当を
返した準決勝・予選は買い手0。「売れた時に当たらず、当たり出した時に売れていない」。

ユーザー決定（2026-08-09）: **看板レースとその前後には必ず推奨を出す**。
目的関数は売上加重の的中率（ROI悪化は許容）。

⚠️ 2026-08-09 時点では検出が実装されておらず、当日の看板レース11件は**手作業**で
   入稿した。実際に和歌山GIII S級決勝・佐世保GI ガールズ決勝ともに、
   朝の波と昼の波を消化しても**商品がゼロ**だった
   （和歌山12Rは軸1の3着内率 95.6% と指数が最も自信を持っていたのに、
     9車ランクのゲートで落ちていた）。

## 判定

    看板   : race_type に 決勝 / 特選 / 選抜 / 特秀 のいずれかを含む
    前後   : 看板レースの前後1レース（同一開催）

⚠️ **レース番号ではなく race_type で判定する**。最終Rが決勝とは限らず
   （ガールズ決勝が6Rと12Rの両方に置かれる開催がある・2026-08-09 佐世保）、
   逆に最終Rが一般戦のこともある。
"""

from __future__ import annotations

MARQUEE_KEYWORDS = ("決勝", "特選", "選抜", "特秀")

# 🔴 部分一致で拾ってしまう非・看板。**「準決勝」は「決勝」を含む**ので
#    除外しないと対象が跳ねる（実測 準決勝は全体の 14.5%）。
#    ユーザーが挙げた看板は「決勝・特選・チャレンジ決勝」で準決勝は含まない。
MARQUEE_EXCLUDE = ("準決勝",)


def is_marquee_type(race_type: str | None) -> bool:
    """race_type が看板レース（決勝・特選クラス）か。

    ⚠️ 除外を先に見る。「準決勝」は「決勝」を部分一致で拾うため。
    """
    if not race_type:
        return False
    if any(k in race_type for k in MARQUEE_EXCLUDE):
        return False
    return any(k in race_type for k in MARQUEE_KEYWORDS)


def marquee_race_nos(races: list[dict]) -> set[int]:
    """同一開催のレース一覧から、看板レースとその前後のレース番号を返す。

    races: [{"race_no": int, "race_type": str|None}, …]（同一開催ぶん）

    ⚠️ 「前後」は**レース番号の±1**。実際に隣接するレースが存在するかは
       呼び出し側が `races` に含まれるかで判断する（欠番があっても
       存在しない番号を返さない）。
    """
    present = {int(r["race_no"]) for r in races if r.get("race_no") is not None}
    marquee = {int(r["race_no"]) for r in races
               if r.get("race_no") is not None and is_marquee_type(r.get("race_type"))}
    out = set(marquee)
    for n in marquee:
        out |= {n - 1, n + 1}
    return out & present
