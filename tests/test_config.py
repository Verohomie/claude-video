"""WATCH_DETAIL resolution and frame_cap mapping."""
from __future__ import annotations

import config


def test_default_detail_is_balanced(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "balanced"


def test_env_overrides_detail(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_DETAIL", "efficient")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "efficient"


def test_invalid_detail_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_DETAIL", "bogus")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "balanced"


def test_get_config_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    cfg = config.get_config()
    assert set(cfg) == {"detail", "quality", "max_fps", "engine", "gemini_model",
                        "gemini_timeout", "gemini_confirm_minutes", "config_file"}


# --- WATCH_MAX_FPS ----------------------------------------------------------
#
# The fps ceiling only ever binds an explicit --fps; auto-fps targets a frame
# budget. It exists for sub-second and fast-action clips, where a budget spread
# over the whole video yields one or two frames of the moment that mattered.


def test_default_max_fps(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_MAX_FPS", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["max_fps"] == 2.0


def test_env_overrides_max_fps(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_MAX_FPS", "24")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["max_fps"] == 24.0


def test_max_fps_can_live_in_the_config_file(monkeypatch, tmp_path):
    """The whole point of routing it through get_config: the PR's version read
    os.environ at import time, so this was the one setting .env could not hold."""
    monkeypatch.delenv("WATCH_MAX_FPS", raising=False)
    env = tmp_path / ".env"
    env.write_text("WATCH_MAX_FPS=12\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", env)
    assert config.get_config()["max_fps"] == 12.0


def test_env_beats_the_config_file(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_MAX_FPS", "6")
    env = tmp_path / ".env"
    env.write_text("WATCH_MAX_FPS=12\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", env)
    assert config.get_config()["max_fps"] == 6.0


def test_a_trailing_comment_does_not_break_it(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_MAX_FPS", raising=False)
    env = tmp_path / ".env"
    env.write_text("WATCH_MAX_FPS=8  # fast action\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", env)
    assert config.get_config()["max_fps"] == 8.0


def test_unusable_max_fps_falls_back_rather_than_raising(monkeypatch, tmp_path):
    """A typo in .env must not take down a run that already paid for a download."""
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    # "nan" parses as a float but fails the > 0 check, so it falls back too.
    for bad in ("bogus", "0", "-4", "nan", "1e400"):
        monkeypatch.setenv("WATCH_MAX_FPS", bad)
        assert config.get_config()["max_fps"] == 2.0, f"{bad!r} should fall back"


def test_frame_cap_mapping():
    assert config.frame_cap("efficient") == 50
    assert config.frame_cap("balanced") == 100
    assert config.frame_cap("token-burner") is None
    assert config.frame_cap("transcript") is None
    assert config.frame_cap("anything-else") == 100
