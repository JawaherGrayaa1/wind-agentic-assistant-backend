from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import os
from pathlib import Path
import re
from typing import Any, Callable


@dataclass
class SkillTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]
    requires_confirmation: bool = False


@dataclass
class Skill:
    name: str
    description: str
    directory: Path
    instructions: str = ""
    tools: list[SkillTool] = field(default_factory=list)


class SkillLoader:
    """Discovers and loads skills from the skills directory."""

    def __init__(self, skills_dir: Path | str | None = None):
        if skills_dir is None:
            self.skills_dir = Path(__file__).resolve().parent
        else:
            self.skills_dir = Path(skills_dir)
        self.skills: dict[str, Skill] = {}

    def parse_skill_markdown(self, skill_md_path: Path) -> tuple[dict[str, str], str]:
        """Parses YAML frontmatter and markdown body from SKILL.md."""
        content = skill_md_path.read_text(encoding="utf-8")
        frontmatter = {}
        body = content

        # Check for YAML frontmatter between --- and ---
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if match:
            fm_text, body = match.groups()
            for line in fm_text.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    frontmatter[key.strip()] = val.strip()

        return frontmatter, body

    def load_all(self, db_instance: Any = None) -> dict[str, Skill]:
        """Loads all skills present in subdirectories."""
        self.skills.clear()
        if not self.skills_dir.is_dir():
            return self.skills

        for item in self.skills_dir.iterdir():
            if item.is_dir() and (item / "SKILL.md").exists():
                skill = self._load_skill_dir(item, db_instance=db_instance)
                if skill:
                    self.skills[skill.name] = skill

        return self.skills

    def _load_skill_dir(self, skill_dir: Path, db_instance: Any = None) -> Skill | None:
        skill_md = skill_dir / "SKILL.md"
        frontmatter, instructions = self.parse_skill_markdown(skill_md)
        name = frontmatter.get("name", skill_dir.name)
        description = frontmatter.get("description", f"Skill for {name}")

        tools: list[SkillTool] = []

        # Check if skill directory has a tools.py implementation
        tools_module_path = skill_dir / "tools.py"
        if tools_module_path.exists():
            module_name = f"erp_agent.skills.{skill_dir.name}.tools"
            spec = importlib.util.spec_from_file_location(module_name, tools_module_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                # If module has a register_tools function
                if hasattr(mod, "register_tools"):
                    registered = mod.register_tools(db=db_instance, skill_dir=skill_dir)
                    tools.extend(registered)

        return Skill(
            name=name,
            description=description,
            directory=skill_dir,
            instructions=instructions,
            tools=tools,
        )
