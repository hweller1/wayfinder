# ruff: noqa: E501  (long fixture lines)
"""Static checks of integrations/claude-code: manifests, agents, commands, skill, hooks wiring, MCP config,
launcher, installer. AC-7.6, FR-I1, FR-I2, FR-I7."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "integrations" / "claude-code"
PREFIX = "mcp__plugin_scout_scout__"
SCOUT_TOOLS = {f"{PREFIX}{t}" for t in ("explore", "semantic_search", "read_range", "index_status")}


def _json(rel: str) -> dict:
    return json.loads((PLUGIN / rel).read_text(encoding="utf-8"))


def frontmatter(path: Path) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", path.read_text(encoding="utf-8"), re.S)
    assert m, f"{path.name} has no frontmatter"
    return dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line), m.group(2)


def tools(fm: dict[str, str]) -> set[str]:
    return {t.strip() for t in fm["tools"].split(",")}


def test_json_files_parse() -> None:
    for rel in (".claude-plugin/plugin.json", ".claude-plugin/marketplace.json", ".mcp.json", "scout-host.json",
                "hooks/hooks.json"):  # fmt: skip
        assert isinstance(_json(rel), dict), rel


def test_plugin_name_version_and_marketplace_agree() -> None:
    plugin = _json(".claude-plugin/plugin.json")
    assert plugin["name"] == "scout"
    assert re.fullmatch(r"\d+\.\d+\.\d+", plugin["version"])
    assert tuple(map(int, plugin["version"].split("."))) >= (0, 2, 0)
    market = _json(".claude-plugin/marketplace.json")
    assert market["name"] == "voyage-scout"
    [entry] = market["plugins"]
    assert (entry["name"], entry["version"], entry["source"]) == ("scout", plugin["version"], "./")
    host = _json("scout-host.json")
    assert host["plugin"] == "scout"
    assert "claude plugin install scout@voyage-scout" in host["install_commands"]


def test_hybrid_agent_frontmatter_and_protocol() -> None:
    fm, body = frontmatter(PLUGIN / "agents" / "scout-explorer.md")
    assert fm["name"] == "scout-explorer"
    assert fm["model"] in ("sonnet", "inherit")
    assert tools(fm) == SCOUT_TOOLS | {"Read", "Grep", "Glob"}
    assert not {"Bash", "Edit", "Write"} & tools(fm)  # read-only
    for field in ("hooks", "mcpServers", "permissionMode"):
        assert field not in fm  # ignored for plugin agents
    # the protocol: explore first, then verify by reading, then fill gaps; at most 12 ranked ranges
    first, verify, gaps = (
        body.index(s)
        for s in ("**Wide sweep first.**", "**Verify the top ranges.**", "**Fill gaps, briefly.**")
    )
    assert first < verify < gaps
    assert "up to 12 files" in body
    assert "(unverified)" in body
    assert "about 8 tool calls" in body
    assert "path/to/file.ext:START-END" in body
    assert "[scout repo: NAME]" in body  # demo lane
    assert len(fm["description"]) < 600


def test_fast_shim_is_kept_and_uses_only_explore() -> None:
    fm, _ = frontmatter(PLUGIN / "agents" / "super-explore.md")
    assert (fm["name"], fm["model"]) == ("super-explore", "haiku")
    assert tools(fm) == {f"{PREFIX}explore"}


def test_mcp_config_exposes_the_hybrid_tools_and_render_budget() -> None:
    """AC-7.6: per-call render budget is 800-1,500 tokens and the ceiling is 1,500."""
    server = _json(".mcp.json")["mcpServers"]["scout"]
    assert server["command"] == "${CLAUDE_PLUGIN_ROOT}/scripts/launch-scout.sh"
    assert 800 <= int(server["env"]["SCOUT_RENDER_MAX_TOKENS"]) <= 1500
    args = server["args"]
    assert int(args[args.index("--render-ceiling") + 1]) == 1500
    exposed = {PREFIX + t for t in args[args.index("--tools") + 1].split(",")}
    agent_tools = set()
    for agent in (PLUGIN / "agents").glob("*.md"):
        agent_tools |= {t for t in tools(frontmatter(agent)[0]) if t.startswith("mcp__")}
    assert agent_tools <= exposed  # every MCP tool an agent names is served
    assert f"{PREFIX}grep" not in exposed  # the hybrid agent uses Claude Code's own Grep


def test_hooks_wiring() -> None:
    hooks = _json("hooks/hooks.json")["hooks"]
    pre = {h["matcher"]: h["hooks"][0]["command"] for h in hooks["PreToolUse"]}
    assert "rewrite_explore.py" in pre["Agent|Task"]
    allow_matcher = next(m for m in pre if "mcp__" in m)
    assert all(re.fullmatch(allow_matcher, t) for t in SCOUT_TOOLS)
    assert not re.fullmatch(allow_matcher, "mcp__plugin_other_scout__explore")
    [ss] = hooks["SessionStart"]
    assert "session_start.py" in ss["hooks"][0]["command"]
    for group in (*hooks["PreToolUse"], *hooks["SessionStart"]):
        for h in group["hooks"]:
            assert h["command"].startswith('python3 "${CLAUDE_PLUGIN_ROOT}/')
            assert h["timeout"] <= 30
            script = PLUGIN / h["command"].split("${CLAUDE_PLUGIN_ROOT}/")[1].split('"')[0]
            assert script.is_file(), script


COMMANDS = {
    "index": None,
    "status": None,
    "find": "<question about the code>",
    "model": "glimmer|v3|a1",
    "mode": "hybrid|fast|off",
}


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_slash_command_frontmatter(name: str) -> None:
    fm, body = frontmatter(PLUGIN / "commands" / f"{name}.md")
    assert fm["description"]
    if COMMANDS[name]:
        assert fm["argument-hint"] == COMMANDS[name]
        assert "$ARGUMENTS" in body
    if name != "find":
        assert fm["allowed-tools"] == "Bash(python3:*)"
        bang = re.findall(r"^!`(.*)`$", body, re.M)
        assert len(bang) == 1
        assert bang[0].startswith('python3 "${CLAUDE_PLUGIN_ROOT}/lib/scout_cc.py" ' + name)
        assert '--data "${CLAUDE_PLUGIN_DATA}"' in bang[0]
    else:
        assert "scout:scout-explorer" in body
        assert "allowed-tools" not in fm


def test_skill_frontmatter_and_guidance() -> None:
    [skill] = list((PLUGIN / "skills").glob("*/SKILL.md"))
    fm, body = frontmatter(skill)
    assert fm["name"] == skill.parent.name == "scout-search"
    assert 50 < len(fm["description"]) < 1024
    assert "scout-explorer" in fm["description"]
    assert "## Delegate to scout-explorer when" in body
    assert "## Do not delegate when" in body
    assert "already know the file" in body
    assert "single exact identifier" in body
    assert 'subagent_type: "scout:scout-explorer"' in body


def test_python_files_compile_and_run_on_old_python() -> None:
    """Hooks run under the user's `python3`, which may be 3.9 (macOS system Python): no 3.10+ syntax at runtime."""
    for path in [*PLUGIN.glob("hooks/*.py"), *PLUGIN.glob("lib/*.py"), PLUGIN / "install.py"]:
        text = path.read_text(encoding="utf-8")
        compile(text, str(path), "exec")
        assert "datetime.UTC)" not in text.replace("no datetime.UTC)", ""), path
        assert "from datetime import UTC" not in text, path
        assert not re.search(r"^\s*(import scout|from scout)\b(?!_cc)", text, re.M), path  # never the package
    old = Path("/usr/bin/python3")
    if old.exists():
        for path in [*PLUGIN.glob("hooks/*.py"), *PLUGIN.glob("lib/*.py"), PLUGIN / "install.py"]:
            proc = subprocess.run([str(old), "-m", "py_compile", str(path)], capture_output=True, check=False)  # noqa: S603
            assert proc.returncode == 0, (path, proc.stderr)


def test_launcher_is_valid_posix_sh_and_executable() -> None:
    script = PLUGIN / "scripts" / "launch-scout.sh"
    assert subprocess.run(["sh", "-n", str(script)], check=False).returncode == 0  # noqa: S603, S607
    assert script.stat().st_mode & 0o100
    assert not re.search(r"\b(curl|wget|nc)\b", script.read_text(encoding="utf-8"))


def fake_claude(tmp_path: Path) -> Path:
    """A `claude` stub that records argv and answers the plugin subcommands the installer uses."""
    log = tmp_path / "claude.log"
    stub = tmp_path / "claude"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"open({str(log)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "a = sys.argv[1:]\n"
        "if a[:3] == ['plugin', 'marketplace', 'list']: print('[]')\n"
        "elif a[:2] == ['plugin', 'list']: print(json.dumps([{'id': 'scout@voyage-scout', 'enabled': True, 'version': '0.2.0'}]))\n"
        "elif a[:2] == ['plugin', 'install']: print(json.dumps({'outcome': 'ok'}))\n"
        "else: print('ok')\n"
    )
    stub.chmod(0o755)
    return stub


def test_installer_runs_validate_add_install_and_verifies(tmp_path: Path) -> None:
    stub = fake_claude(tmp_path)
    scout = tmp_path / "scout"
    scout.write_text("#!/bin/sh\n")
    scout.chmod(0o755)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "SCOUT_KEYSTORE": "file", "SCOUT_BIN": str(scout),
           "SCOUT_GATEWAY_URL": "http://127.0.0.1:9", "CLAUDE_CONFIG_DIR": str(tmp_path / "cfg")}  # fmt: skip
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(PLUGIN / "install.py"), "--claude", str(stub)],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    report = json.loads(proc.stdout.splitlines()[-1])
    assert [s["step"] for s in report["steps"]] == ["validate", "marketplace add", "install"]
    assert all(s["rc"] == 0 for s in report["steps"])
    assert report["installed"] is True
    assert report["config_dir"] == str(tmp_path / "cfg")
    assert report["doctor"]["scout_binary"] == str(scout)
    assert report["ok"] is True
    assert proc.returncode == 0
    calls = (tmp_path / "claude.log").read_text().splitlines()
    assert f"plugin marketplace add {PLUGIN}" in calls
    assert "plugin install scout@voyage-scout --json" in calls


def test_installer_dry_run_changes_nothing(tmp_path: Path) -> None:
    stub = fake_claude(tmp_path)
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(PLUGIN / "install.py"), "--claude", str(stub), "--dry-run"],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}, capture_output=True, text=True, check=True,
    )  # fmt: skip
    report = json.loads(proc.stdout)
    assert [s["step"] for s in report["steps"]] == ["validate", "marketplace add", "install"]
    calls = (tmp_path / "claude.log").read_text().splitlines()
    assert calls == ["plugin marketplace list --json"]  # the only call a dry run makes is read-only


def test_real_host_file_runs_the_installer_through_scout_init(tmp_path: Path) -> None:
    """`scout init --claude-code --dry-run` lists the installer step from scout-host.json (no host run)."""
    host = _json("scout-host.json")
    assert host["install_steps"] == [["{python}", "{integrations}/claude-code/install.py"]]
    env = {**os.environ, "HOME": str(tmp_path), "SCOUT_INTEGRATIONS_DIR": str(REPO / "integrations")}
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "scout", "init", "--claude-code", "--dry-run"],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    out = json.loads(proc.stdout)
    assert out["install"]["ran"] is False
    [step] = out["install"]["steps"]
    assert step["argv"][1] == str((REPO / "integrations" / "claude-code" / "install.py").resolve())
