"""凍結vintageモデルのマニフェスト（`data/models/vintage_manifest.json`）管理CLI。

2026-07-31・D-4（`rm` 耐性のある vintage 凍結保護強化）で追加。
`src/models/vintage_manifest.py` のdocstringに設計意図・限界を記載。

サブコマンド:
  build   - `data/models/` を走査し、vintage命名規則（`_VINTAGE_NAME_RE`）に
            一致する `.pkl` のうち、まだマニフェスト未登録のものを登録する
            （既存エントリは上書きしない。上書きしたい場合は `--force`）。
            **ファイルの読み取り（SHA256計算）のみを行い、モデルファイル自体を
            書き換えたり削除したりしない。**
  verify  - マニフェストと実ファイルの整合性を検証し、欠落・ハッシュ不一致を
            報告する（終了コード: 問題があれば1、無ければ0）。

実行例:
    .venv/bin/python scripts/build_vintage_manifest.py build
    .venv/bin/python scripts/build_vintage_manifest.py verify
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.vintage_manifest import (  # noqa: E402
    MODEL_DIR,
    VINTAGE_NAME_RE,
    is_registered,
    register,
    verify_manifest,
)


def cmd_build(args: argparse.Namespace) -> int:
    pkl_files = sorted(MODEL_DIR.glob("*.pkl"))
    vintage_files = [p for p in pkl_files if VINTAGE_NAME_RE.search(p.stem)]
    print(f"vintage命名規則に一致する .pkl: {len(vintage_files)}件 "
          f"(全 {len(pkl_files)}件中)")

    registered, skipped = 0, 0
    for path in vintage_files:
        name = path.stem
        if is_registered(name) and not args.force:
            skipped += 1
            continue
        entry = register(name, path)
        print(f"[registered] {name}  sha256={entry['sha256'][:12]}...  "
              f"size={entry['size_bytes']:,}B")
        registered += 1

    print(f"\n完了: registered={registered} skipped(既存)={skipped}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify_manifest()
    n_ok = len(result["ok"])
    n_missing = len(result["missing"])
    n_mismatch = len(result["hash_mismatch"])
    print(f"ok={n_ok}  missing={n_missing}  hash_mismatch={n_mismatch}")

    if result["missing"]:
        print("\n[missing] マニフェストに登録済みだがファイル実体が無い"
              "（rm等で削除された可能性）:")
        for name in result["missing"]:
            print(f"  - {name}")

    if result["hash_mismatch"]:
        print("\n[hash_mismatch] ファイルは存在するがSHA256が一致しない"
              "（内容が書き換えられた可能性）:")
        for name in result["hash_mismatch"]:
            print(f"  - {name}")

    if result["missing"] or result["hash_mismatch"]:
        print("\n[NG] 整合性の問題を検出しました。")
        return 1
    print("\n[OK] マニフェストと実ファイルは全て整合しています。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="未登録のvintageモデルをマニフェストへ登録")
    p_build.add_argument("--force", action="store_true",
                          help="既存エントリもハッシュを再計算して上書き登録する")
    p_build.set_defaults(func=cmd_build)

    p_verify = sub.add_parser("verify", help="マニフェストと実ファイルの整合性を検証")
    p_verify.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
