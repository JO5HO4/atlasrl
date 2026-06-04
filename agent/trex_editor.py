from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import re


@dataclass
class Block:
    kind: str
    name: str
    start: int
    end: int
    lines: list[str]


_BLOCK_RE = re.compile(r'^(Job|Fit|Region|Sample|NormFactor|Systematic):\s*"([^"]+)"')


class TrexConfigEditor:
    """
    Minimal block-level TRExFitter editor.

    TRExFitter config blocks look like:

        Region: "cat_2jet"
          Type: SIGNAL
          Variable: "...",30,100,160
          Selection: "..."

    This editor preserves text and comments as much as possible,
    while allowing semantic edits to Region/Sample/NormFactor blocks.
    """

    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.blocks = self._parse_blocks()

    @classmethod
    def from_file(cls, path: str | Path) -> "TrexConfigEditor":
        return cls(Path(path).read_text())

    def to_text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_text())

    def _parse_blocks(self) -> list[Block]:
        starts = []

        for i, line in enumerate(self.lines):
            match = _BLOCK_RE.match(line)
            if match:
                starts.append((i, match.group(1), match.group(2)))

        blocks = []

        for idx, (start, kind, name) in enumerate(starts):
            if idx + 1 < len(starts):
                end = starts[idx + 1][0]
            else:
                end = len(self.lines)

            blocks.append(
                Block(
                    kind=kind,
                    name=name,
                    start=start,
                    end=end,
                    lines=self.lines[start:end],
                )
            )

        return blocks

    def refresh(self) -> None:
        self.blocks = self._parse_blocks()

    def get_blocks(self, kind: str) -> list[Block]:
        return [b for b in self.blocks if b.kind == kind]

    def get_block(self, kind: str, name: str) -> Block:
        for block in self.blocks:
            if block.kind == kind and block.name == name:
                return block
        raise KeyError(f"No {kind} block named {name!r}")

    def get_block_by_index(self, kind: str, index: int) -> Block:
        blocks = self.get_blocks(kind)
        if not blocks:
            raise KeyError(f"No {kind} blocks found")
        return blocks[index % len(blocks)]

    def get_regions(self) -> list[Block]:
        return self.get_blocks("Region")

    def get_region(self, name_or_index: str | int) -> Block:
        if isinstance(name_or_index, int):
            return self.get_block_by_index("Region", name_or_index)
        return self.get_block("Region", name_or_index)

    def get_region_names(self) -> list[str]:
        return [b.name for b in self.get_regions()]

    def get_field(self, block: Block, key: str) -> str | None:
        pattern = re.compile(rf"^\s*{re.escape(key)}:\s*(.*)$")

        for line in block.lines:
            match = pattern.match(line)
            if match:
                return match.group(1).strip()

        return None

    def get_field_line_index(self, block: Block, key: str) -> int | None:
        pattern = re.compile(rf"^\s*{re.escape(key)}:\s*")

        for line_index in range(block.start, block.end):
            if pattern.match(self.lines[line_index]):
                return line_index

        return None

    def set_field(self, block: Block, key: str, value: str) -> None:
        """
        value should include quotes if TRExFitter expects quoted value.
        Example:
            set_field(region, "Selection", '"photon_n>=2"')
        """
        pattern = re.compile(rf"^(\s*){re.escape(key)}:\s*")

        new_block_lines = []
        replaced = False

        for line in block.lines:
            if pattern.match(line):
                indent = pattern.match(line).group(1)
                new_block_lines.append(f"{indent}{key}: {value}")
                replaced = True
            else:
                new_block_lines.append(line)

        if not replaced:
            new_block_lines.append(f"  {key}: {value}")

        self._replace_block_lines(block, new_block_lines)

    def get_selection(self, region: Block | str | int) -> str:
        region_block = self._coerce_region(region)
        raw = self.get_field(region_block, "Selection")
        if raw is None:
            raise KeyError(f"Region {region_block.name!r} has no Selection field")
        return unquote_trex_value(raw)

    def set_selection(self, region: Block | str | int, selection: str) -> None:
        region_block = self._coerce_region(region)
        self.set_field(region_block, "Selection", quote_trex_value(selection))

    def replace_selection_fragment(
        self,
        region: Block | str | int,
        old: str,
        new: str,
        *,
        count: int = 1,
    ) -> int:
        selection = self.get_selection(region)
        updated, replaced = selection.replace(old, new, count), selection.count(old)
        if count >= 0:
            replaced = min(replaced, count)
        if replaced:
            self.set_selection(region, updated)
        return replaced

    def append_selection_cut(self, region: Block | str | int, cut: str) -> None:
        selection = self.get_selection(region)
        self.set_selection(region, f"({selection}) && ({cut})")

    def remove_selection_fragment(self, region: Block | str | int, pattern: str) -> int:
        selection = self.get_selection(region)
        updated = re.sub(rf"\s*&&\s*\({pattern}\)", "", selection, count=1)
        if updated == selection:
            updated = re.sub(rf"\({pattern}\)\s*&&\s*", "", selection, count=1)
        if updated == selection:
            updated = re.sub(pattern, "1", selection, count=1)
        if updated == selection:
            return 0
        self.set_selection(region, updated)
        return 1

    def selection_line_action(self, region: Block | str | int) -> dict[str, object]:
        region_block = self._coerce_region(region)
        line_index = self.get_field_line_index(region_block, "Selection")
        if line_index is None:
            raise KeyError(f"Region {region_block.name!r} has no Selection field")
        return {
            "op": 2,
            "line": line_index,
            "text": self.lines[line_index],
            "n": 1,
        }

    def delete_block(self, block: Block) -> None:
        del self.lines[block.start:block.end]
        self.refresh()

    def insert_block_after(self, after_block: Block, new_block_lines: list[str]) -> None:
        insert_at = after_block.end
        self.lines[insert_at:insert_at] = [""] + new_block_lines
        self.refresh()

    def replace_block(self, block: Block, new_block_lines: list[str]) -> None:
        self._replace_block_lines(block, new_block_lines)

    def _replace_block_lines(self, block: Block, new_block_lines: list[str]) -> None:
        self.lines[block.start:block.end] = new_block_lines
        self.refresh()

    def _coerce_region(self, region: Block | str | int) -> Block:
        if isinstance(region, Block):
            return self.get_block(region.kind, region.name)
        return self.get_region(region)

    def clone_region(self, old_region: Block, new_name: str) -> Block:
        new_lines = copy.deepcopy(old_region.lines)
        new_lines[0] = f'Region: "{new_name}"'

        last_region = self.get_regions()[-1]
        self.insert_block_after(last_region, new_lines)

        return self.get_block("Region", new_name)

    def clone_region_after(self, old_region: Block | str | int, new_name: str) -> Block:
        old_region = self._coerce_region(old_region)
        new_lines = copy.deepcopy(old_region.lines)
        new_lines[0] = f'Region: "{new_name}"'
        for index, line in enumerate(new_lines):
            if re.match(r"^\s*ShortLabel:\s*", line):
                indent = re.match(r"^(\s*)", line).group(1)
                new_lines[index] = f'{indent}ShortLabel: "{new_name}"'
        self.insert_block_after(old_region, new_lines)
        return self.get_block("Region", new_name)

    def unique_region_name(self, base_name: str) -> str:
        names = set(self.get_region_names())
        index = 1
        while True:
            candidate = f"{base_name}_rl{index}"
            if candidate not in names:
                return candidate
            index += 1


def unquote_trex_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def quote_trex_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
