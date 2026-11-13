import os
import shutil

import pytest

import safety
import tools


@pytest.fixture
def scratch():
    """A throwaway folder inside the workspace, removed after each test."""
    d = safety.WORKSPACE / "tmp_scratch"
    d.mkdir(exist_ok=True)
    (d / "a.txt").write_text("hello world", encoding="utf-8")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_list_files_shows_full_paths(scratch):
    result = tools.list_files("tmp_scratch")
    assert result.startswith("Contents of ")
    assert str(scratch / "a.txt") in result


def test_list_files_marks_folders_with_a_trailing_separator(scratch):
    (scratch / "sub").mkdir()
    assert str(scratch / "sub") + os.sep in tools.list_files("tmp_scratch")


def test_list_files_caps_long_listings(scratch):
    for i in range(tools.MAX_LISTED + 10):
        (scratch / f"f{i}.txt").write_text("", encoding="utf-8")
    assert "more not shown" in tools.list_files("tmp_scratch")


def test_list_files_on_a_file_says_what_to_do(scratch):
    result = tools.list_files("tmp_scratch/a.txt")
    assert "is a file" in result and "read_file" in result


def test_read_file_on_a_folder_says_what_to_do(scratch):
    result = tools.read_file("tmp_scratch")
    assert "is a folder" in result and "list_files" in result


def test_read_file_works_with_an_absolute_path(scratch):
    assert tools.read_file(str(scratch / "a.txt")) == "hello world"


def test_search_text_on_a_folder_says_what_to_do(scratch):
    assert "is a folder" in tools.search_text("tmp_scratch", "x")


def test_search_text_finds_lines(scratch):
    assert "1: hello world" in tools.search_text("tmp_scratch/a.txt", "WORLD")


def test_errors_show_the_resolved_path():
    wanted = str((safety.WORKSPACE / "no" / "such").resolve())
    assert wanted in tools.list_files("no/such")


def test_env_files_are_still_blocked():
    assert "blocked" in tools.read_file("../.env")