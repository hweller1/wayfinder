"""release/stage.py: card gate, staged-folder gate, adapter key parsing."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "release"))
import stage  # noqa: E402

CARD = (ROOT / "release" / "MODEL_CARD.md").read_text()


def test_template_card_renders_and_passes():
    out = stage.render_card(CARD, repo_id="u/x", variant="v1")
    assert stage.check_card(out) == []
    assert stage.count_todos(out) == 0  # the card is final: no unresolved TODO(final) markers
    assert "{{" not in out


def test_unresolved_placeholder_rejected():
    with pytest.raises(ValueError):
        stage.render_card("{{NOPE}}", repo_id="u/x", variant="v1")


def test_card_without_attribution_fails():
    bad = stage.render_card(CARD, repo_id="u/x", variant="v1").replace(stage.ATTRIBUTION, "Built with something")
    assert any("Muse Spark Open" in e for e in stage.check_card(bad))


def test_card_attribution_must_be_near_top():
    body = stage.render_card(CARD, repo_id="u/x", variant="v1")
    front, rest = body.split("\n---\n", 1)
    moved = front + "\n---\n" + "\n".join(["filler"] * 30) + "\n" + rest.replace(f"**{stage.ATTRIBUTION}.**", "")
    assert any("first lines" in e for e in stage.check_card(moved))


def test_check_staged(tmp_path):
    for n in stage.REQUIRED:
        (tmp_path / n).write_text("x")
    (tmp_path / "NOTICE").write_text(stage.NOTICE)
    (tmp_path / "model-00001-of-00002.safetensors").write_text("x")
    assert stage.check_staged(tmp_path) == []
    (tmp_path / "adapter_model.safetensors").write_text("x")
    (tmp_path / ".env").write_text("HF_TOKEN=x")
    errs = stage.check_staged(tmp_path)
    assert any("adapter_model" in e for e in errs) and any(".env" in e for e in errs)


def test_base_files_to_copy():
    names = ["config.json", "README.md", "LICENSE", "model-00001-of-00002.safetensors", ".cache", ".eval_results/x.yaml",
             "tokenizer.json", "chat_template.jinja", ".download_complete"]
    assert stage.base_files_to_copy(names) == ["chat_template.jinja", "config.json", "tokenizer.json"]


def test_lora_module_name():
    k = "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight"
    assert stage.lora_module_name(k) == ("model.language_model.layers.0.mlp.down_proj.weight", "B")
    with pytest.raises(ValueError):
        stage.lora_module_name("model.x.lora_A.weight")


def test_vendored_licences_present_and_attribution_text():
    for vendored in stage.LICENSE_FILES.values():
        assert (ROOT / "release" / "licenses" / vendored).stat().st_size > 1000
    assert '"Built with Muse Spark Open"' in (ROOT / "release/licenses/LICENSE-MUSE-SPARK-OPEN.md").read_text()


def test_public_refused_while_todo_markers_remain():
    card = stage.render_card(CARD, repo_id="u/x", variant="v3")
    assert stage.public_refusal(card, public=True) is None  # the shipped card is final
    assert stage.public_refusal(card + "\nTODO(final): x", public=True) is not None
    assert stage.public_refusal(card + "\nTODO(final): x", public=False) is None  # private upload stays allowed
