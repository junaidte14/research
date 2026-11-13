import json

import safety
import tools
from loopguard import LoopGuard, normalize


def test_equivalent_paths_normalize_the_same():
    assert normalize('{"directory": "../.."}') == normalize('{"directory": "../../"}')


def test_third_identical_call_stops():
    g = LoopGuard()
    args = '{"directory": "."}'
    assert g.check("list_files", args) == "ok"
    assert g.check("list_files", args) == "warn"
    assert g.check("list_files", args) == "stop"


def test_equivalent_spellings_count_together():
    g = LoopGuard()
    g.check("list_files", '{"directory": "../.."}')
    assert g.check("list_files", '{"directory": "../../"}') == "warn"


def test_different_calls_are_fine():
    g = LoopGuard()
    for d in [".", "..", "../..", "sub"]:
        assert g.check("list_files", json.dumps({"directory": d})) == "ok"


def test_state_change_resets_counting():
    g = LoopGuard()
    args = '{"path": "a.py"}'
    g.check("read_file", args)
    g.check("read_file", args)                                        # warn
    g.check("edit_file", '{"path": "a.py", "old_text": "x", "new_text": "y"}')
    assert g.check("read_file", args) == "ok"                         # re-reading after an edit is legitimate


def test_list_files_header_names_the_folder():
    result = tools.list_files("..")
    assert result.startswith("Contents of ")
    assert str(safety.WORKSPACE.parent) in result.splitlines()[0]


def test_list_files_error_shows_the_resolved_path():
    result = tools.list_files("no-such-folder")
    assert result.startswith("Error") and "no-such-folder" in result


def test_plan_renders_progress():
    result = tools.update_plan(["a", "b", "c"], done=1)
    assert "[x] 1. a" in result and "[>] 2. b" in result and "[ ] 3. c" in result


def test_plan_accepts_a_json_string():
    assert "[>] 1. a" in tools.update_plan('["a", "b"]')


def test_plan_schema_declares_an_array():
    schema = next(t for t in tools.TOOLS if t["function"]["name"] == "update_plan")
    params = schema["function"]["parameters"]
    assert params["properties"]["steps"]["type"] == "array"
    assert "steps" in params["required"] and "done" not in params["required"]