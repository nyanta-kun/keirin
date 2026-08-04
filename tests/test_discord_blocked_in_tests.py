"""テストから実際の Discord 送信が飛ばないことを固定する（2026-08-04）。

2026-08-04、`test_three_head_rebuild_guard.py` の1ケースが `rebuild_pg_atomic` の
「挿入対象0件」警告経路を monkeypatch なしで通り、**本番の #システム障害
チャンネルへ警告が5通投稿された**（ローカルの .env に本番webhookが入っているため。
pytest を回すたびに1通増えていた）。

テストが本番の通知先を汚すのは検査として明確に誤りで、かつ個別テストの
monkeypatch 漏れは今後も必ず起きる。`tests/conftest.py` の autouse fixture
`_block_discord` で webhook URL の解決自体を潰しているので、その効き目を固定する。
"""
from src.notify.discord import send, send_file


def test_send_is_blocked():
    """webhook URL が解決できないため送信前に False で返る。"""
    assert send("これはテストです。実送信されてはいけません。", channel="system") is False


def test_send_file_is_blocked(tmp_path):
    f = tmp_path / "dummy.txt"
    f.write_text("x", encoding="utf-8")
    assert send_file(str(f), channel="system") is False


def test_notify_discord_warning_does_not_raise():
    """通知経路の失敗が処理本体を巻き込まない設計であることも併せて固定する。"""
    from src.wt_rebuild_common import notify_discord_warning
    notify_discord_warning("テスト（実送信されてはいけません）")
