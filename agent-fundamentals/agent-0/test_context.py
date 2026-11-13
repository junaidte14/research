from types import SimpleNamespace

import context
from loopguard import LoopGuard


def make_history():
    """system + 3 turns. Turn 2 contains a tool call with a big result that starts with a header line."""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "turn 2: list a folder"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "list_files", "arguments": '{"directory": "X"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Contents of X (3 items):\n" + "x" * 5000},
        {"role": "assistant", "content": "answer 2"},
        {"role": "user", "content": "turn 3"},
        {"role": "assistant", "content": "answer 3"},
    ]


def assert_tool_pairs_intact(messages):
    open_ids = set()
    for m in messages:
        if m["role"] == "assistant":
            open_ids = {c["id"] for c in m.get("tool_calls") or []}
        elif m["role"] == "tool":
            assert m["tool_call_id"] in open_ids


# ---------- pruning ----------
def test_prune_keeps_the_head_and_marks_the_rest():
    h = make_history()
    assert context.prune_old_tool_results(h, keep_recent=2) == 1
    content = h[5]["content"]
    assert content.startswith("Contents of X (3 items):")
    assert context.PRUNE_MARK in content
    assert len(content) < 500


def test_prune_is_idempotent():
    h = make_history()
    context.prune_old_tool_results(h, keep_recent=2)
    assert context.prune_old_tool_results(h, keep_recent=2) == 0


def test_prune_keeps_recent_results():
    h = make_history()
    assert context.prune_old_tool_results(h, keep_recent=len(h)) == 0


def test_prune_skips_small_results():
    h = make_history()
    h[5]["content"] = "small"
    assert context.prune_old_tool_results(h, keep_recent=2) == 0


# ---------- manage_context ----------
def test_manage_context_does_nothing_below_the_thresholds():
    h = make_history()
    assert context.manage_context(h, 100, 1000, 2000, lambda t: "S") is False


def test_manage_context_prunes_between_the_thresholds():
    h = make_history()
    assert context.manage_context(h, 1500, 1000, 2000, lambda t: "S", keep_recent=2) is True
    assert context.PRUNE_MARK in h[5]["content"]


def test_manage_context_summarizes_above_the_compact_threshold():
    h = make_history()
    assert context.manage_context(h, 5000, 1000, 2000, lambda t: "FAKE-SUMMARY", keep_recent=3) is True
    assert "FAKE-SUMMARY" in h[1]["content"]
    assert_tool_pairs_intact(h)


def test_repeats_after_pruning_are_legitimate():
    g = LoopGuard()
    args = '{"directory": "."}'
    g.check("list_files", args)
    assert g.check("list_files", args) == "warn"
    g.reset()          # agent.py does this whenever manage_context changed the history
    assert g.check("list_files", args) == "ok"


# ---------- summarizing ----------
def test_cut_point_is_a_user_message():
    h = make_history()
    cut = context._safe_cut(h, keep_recent=3)
    assert h[cut]["role"] == "user"


def test_compact_keeps_tool_pairs_and_inserts_summary():
    h = make_history()
    assert context.compact(h, lambda text: "SUMMARY", keep_recent=3)
    assert h[0]["role"] == "system"
    assert "SUMMARY" in h[1]["content"] and h[1]["role"] == "user"
    assert h[2]["role"] == "assistant"
    assert_tool_pairs_intact(h)


def test_compact_does_nothing_when_there_is_nothing_old():
    h = [{"role": "system", "content": "s"}, {"role": "user", "content": "only turn"}]
    assert context.compact(h, lambda text: "SUMMARY") is False
    assert len(h) == 2


def test_failed_summary_keeps_history():
    def boom(text):
        raise RuntimeError("api down")
    h = make_history()
    before = list(h)
    assert context.compact(h, boom, keep_recent=3) is False
    assert h == before


# ---------- sessions and messages ----------
def test_session_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "SESSIONS_DIR", tmp_path)
    h = make_history()
    context.save_session(h, "t")
    assert context.load_session("t") == h


def test_message_to_dict_with_tool_call():
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="read_file", arguments='{"path": "a"}'))
    d = context.message_to_dict(SimpleNamespace(content=None, tool_calls=[call]))
    assert d["content"] is None
    assert d["tool_calls"][0]["function"]["name"] == "read_file"


def test_message_to_dict_plain_answer():
    d = context.message_to_dict(SimpleNamespace(content=None, tool_calls=None))
    assert d["content"] == ""


def test_estimate_tokens_is_roughly_chars_over_four():
    assert context.estimate_tokens([{"role": "user", "content": "a" * 400}]) >= 100