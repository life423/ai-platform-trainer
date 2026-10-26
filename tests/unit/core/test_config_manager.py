"""The settings file: fullscreen only, read strictly and saved back as JSON."""
import json

from ai_platform_trainer.core.config_manager import ConfigManager


def test_defaults_when_there_is_no_file(tmp_path):
    manager = ConfigManager(str(tmp_path / "missing.json"))

    assert manager.get("display.fullscreen") is False
    assert manager.get("display.width") is None


def test_reads_a_saved_fullscreen_value(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"display": {"fullscreen": True}}), encoding="utf-8")

    assert ConfigManager(str(path)).get("display.fullscreen") is True


def test_a_string_false_is_false_not_truthy(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"display": {"fullscreen": "false"}}), encoding="utf-8")

    assert ConfigManager(str(path)).get("display.fullscreen") is False


def test_an_unreadable_value_falls_back_to_the_default(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"display": {"fullscreen": "maybe"}}), encoding="utf-8")

    assert ConfigManager(str(path)).get("display.fullscreen") is False


def test_a_broken_file_is_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")

    assert ConfigManager(str(path)).get("display.fullscreen") is False


def test_set_creates_sections_and_get_reads_them_back(tmp_path):
    manager = ConfigManager(str(tmp_path / "config.json"))

    manager.set("display.fullscreen", True)
    manager.set("new_section.value", 42)

    assert manager.get("display.fullscreen") is True
    assert manager.get("new_section.value") == 42
    assert manager.get("new_section.missing", "fallback") == "fallback"


def test_save_writes_the_setting_and_a_new_manager_reads_it(tmp_path):
    path = str(tmp_path / "config.json")
    manager = ConfigManager(path)
    manager.set("display.fullscreen", True)

    manager.save()

    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == {
        "display": {"fullscreen": True}
    }
    assert ConfigManager(path).get("display.fullscreen") is True
