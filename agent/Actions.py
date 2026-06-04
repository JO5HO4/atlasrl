from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable
from typing import Any

from agent.trex_editor import Block, TrexConfigEditor


ACTION_NAMES = [
    "modify_category_cut",
    "add_category_cut",
    "remove_category_cut",
    "add_category",
    "remove_category",
]

CUT_NAMES = [
    "lead_photon_pt",
    "sublead_photon_pt",
    "photon_eta_max",
    "ptt",
    "jet_pt",
    "deta_jj",
    "mjj",
    "dphi_gamgam_jj",
]

PERCENT_VALUES = [-50, -30, -20, -10, -5, -2, 2, 5, 10, 20, 30, 50]

PTT_EXPR = (
    "fabs(2*photon_pt[0]*photon_pt[1]*sin(photon_phi[0]-photon_phi[1]))/"
    "sqrt(pow(photon_pt[0],2)+pow(photon_pt[1],2)-2*photon_pt[0]*photon_pt[1]*"
    "cos(photon_phi[0]-photon_phi[1]))"
)
MJJ_EXPR = (
    "sqrt(2*jet_pt[0]*jet_pt[1]*(cosh(jet_eta[0]-jet_eta[1])-"
    "cos(jet_phi[0]-jet_phi[1])))"
)
DPHI_GAMGAM_JJ_EXPR = (
    "acos(cos((atan2(photon_pt[0]*sin(photon_phi[0])+photon_pt[1]*sin(photon_phi[1]),"
    "photon_pt[0]*cos(photon_phi[0])+photon_pt[1]*cos(photon_phi[1])))-"
    "(atan2(jet_pt[0]*sin(jet_phi[0])+jet_pt[1]*sin(jet_phi[1]),"
    "jet_pt[0]*cos(jet_phi[0])+jet_pt[1]*cos(jet_phi[1]))))))"
)


@dataclass(frozen=True)
class SelectionEditResult:
    region: str
    cut_name: str
    replacements: int
    before: str
    after: str

@dataclass(frozen=True)
class Action:
    action_type: int
    region_index: int
    cut_type: int
    percent_index: int

    @classmethod
    def from_dict(cls, action: dict[str, Any]) -> "Action":
        return cls(
            action_type=int(action["action_type"]),
            region_index=int(action["region_index"]),
            cut_type=int(action["cut_type"]),
            percent_index=int(action["percent_index"]),
        )

    @classmethod
    def category_cut(
        cls,
        action_name: str,
        *,
        region_index: int,
        cut_name: str,
        percent: int | float = 2,
    ) -> "Action":
        return cls(
            action_type=ACTION_NAMES.index(action_name),
            region_index=int(region_index),
            cut_type=CUT_NAMES.index(cut_name),
            percent_index=PERCENT_VALUES.index(int(percent)),
        )

    @classmethod
    def modify_category_cut(
        cls,
        *,
        region_index: int,
        cut_name: str,
        percent: int | float,
    ) -> "Action":
        return cls.category_cut(
            "modify_category_cut",
            region_index=region_index,
            cut_name=cut_name,
            percent=percent,
        )

    @classmethod
    def add_category_cut(
        cls,
        *,
        region_index: int,
        cut_name: str,
        percent: int | float,
    ) -> "Action":
        return cls.category_cut(
            "add_category_cut",
            region_index=region_index,
            cut_name=cut_name,
            percent=percent,
        )

    @classmethod
    def remove_category_cut(
        cls,
        *,
        region_index: int,
        cut_name: str,
    ) -> "Action":
        return cls.category_cut(
            "remove_category_cut",
            region_index=region_index,
            cut_name=cut_name,
        )

    @classmethod
    def add_category(cls, *, region_index: int) -> "Action":
        return cls(
            action_type=ACTION_NAMES.index("add_category"),
            region_index=int(region_index),
            cut_type=0,
            percent_index=PERCENT_VALUES.index(2),
        )

    @classmethod
    def remove_category(cls, *, region_index: int) -> "Action":
        return cls(
            action_type=ACTION_NAMES.index("remove_category"),
            region_index=int(region_index),
            cut_type=0,
            percent_index=PERCENT_VALUES.index(2),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "action_type": self.action_type,
            "region_index": self.region_index,
            "cut_type": self.cut_type,
            "percent_index": self.percent_index,
        }

    @property
    def action_name(self) -> str:
        return ACTION_NAMES[self.action_type]

    @property
    def cut_name(self) -> str:
        return CUT_NAMES[self.cut_type]

    @property
    def percent(self) -> int:
        return PERCENT_VALUES[self.percent_index]

    def apply(self, cfg: TrexConfigEditor) -> SelectionEditResult:
        return apply_analysis_action(cfg, self)

    def __str__(self) -> str:
        sign = "+" if self.percent > 0 else ""
        if self.action_name in {"add_category", "remove_category"}:
            return f"{self.action_name}: region_index={self.region_index}"
        if self.action_name == "remove_category_cut":
            return (
                f"{self.action_name}: region_index={self.region_index}, "
                f"cut={self.cut_name}"
            )
        return (
            f"{self.action_name}: region_index={self.region_index}, "
            f"cut={self.cut_name}, scale={sign}{self.percent}%"
        )


def apply_analysis_action(
    cfg: TrexConfigEditor,
    action: Action | dict[str, Any],
) -> SelectionEditResult:
    """
    Apply one action from AnalysisEnv.action_space to a TREx config editor.

    Expected action keys:
      action_type, region_index, cut_type, percent_index
    """
    action_obj = action if isinstance(action, Action) else Action.from_dict(action)

    action_name = action_obj.action_name
    cut_name = action_obj.cut_name
    percent = action_obj.percent

    if action_name == "modify_category_cut":
        return scale_category_cut(cfg, action_obj.region_index, cut_name, percent)
    if action_name == "add_category_cut":
        return add_category_cut(cfg, action_obj.region_index, cut_name, percent)
    if action_name == "remove_category_cut":
        return remove_category_cut(cfg, action_obj.region_index, cut_name)
    if action_name == "add_category":
        return add_category(cfg, action_obj.region_index)
    if action_name == "remove_category":
        return remove_category(cfg, action_obj.region_index)

    raise ValueError(f"Unsupported action_type {action_obj.action_type}: {action_name!r}")


def make_modify_category_cut_action(
    region_index: int,
    cut_name: str,
    percent: int | float,
) -> dict[str, int]:
    """Build a Gym-compatible action dict."""
    return Action.modify_category_cut(
        region_index=region_index,
        cut_name=cut_name,
        percent=percent,
    ).to_dict()


def add_category_cut(
    cfg: TrexConfigEditor,
    region: str | int | Block,
    cut_name: str,
    percent: float,
) -> SelectionEditResult:
    region_block = _coerce_region(cfg, region)
    before = cfg.get_selection(region_block)
    cut = _build_cut(cut_name, percent)
    cfg.append_selection_cut(region_block.name, cut)
    after = cfg.get_selection(region_block.name)
    return SelectionEditResult(region_block.name, cut_name, 1, before, after)


def remove_category_cut(
    cfg: TrexConfigEditor,
    region: str | int | Block,
    cut_name: str,
) -> SelectionEditResult:
    region_block = _coerce_region(cfg, region)
    before = cfg.get_selection(region_block)
    replacements = cfg.remove_selection_fragment(region_block.name, _remove_pattern(cut_name))
    if replacements == 0:
        raise ValueError(f"Cut {cut_name!r} was not found in region {region_block.name!r}")
    after = cfg.get_selection(region_block.name)
    return SelectionEditResult(region_block.name, cut_name, replacements, before, after)


def add_category(cfg: TrexConfigEditor, region: str | int | Block) -> SelectionEditResult:
    region_block = _coerce_region(cfg, region)
    before = ",".join(cfg.get_region_names())
    new_name = cfg.unique_region_name(region_block.name)
    cfg.clone_region_after(region_block, new_name)
    after = ",".join(cfg.get_region_names())
    return SelectionEditResult(new_name, "category", 1, before, after)


def remove_category(cfg: TrexConfigEditor, region: str | int | Block) -> SelectionEditResult:
    if len(cfg.get_regions()) <= 1:
        raise ValueError("Cannot remove the only remaining category")
    region_block = _coerce_region(cfg, region)
    before = ",".join(cfg.get_region_names())
    cfg.delete_block(region_block)
    after = ",".join(cfg.get_region_names())
    return SelectionEditResult(region_block.name, "category", 1, before, after)


def scale_category_cut(
    cfg: TrexConfigEditor,
    region: str | int | Block,
    cut_name: str,
    percent: float,
) -> SelectionEditResult:
    """Scale a cut by percentage inside exactly one Region Selection."""
    region_block = _coerce_region(cfg, region)
    before = cfg.get_selection(region_block)

    editor = _CUT_SCALERS.get(cut_name)
    if editor is None:
        raise ValueError(f"Unknown cut_name {cut_name!r}; choose one of {sorted(_CUT_SCALERS)}")

    after, replacements = editor(before, float(percent))
    if replacements == 0:
        raise ValueError(f"Cut {cut_name!r} was not found in region {region_block.name!r}")

    cfg.set_selection(region_block.name, after)
    return SelectionEditResult(
        region=region_block.name,
        cut_name=cut_name,
        replacements=replacements,
        before=before,
        after=after,
    )

def _coerce_region(cfg: TrexConfigEditor, region: str | int | Block) -> Block:
    if isinstance(region, Block):
        return cfg.get_region(region.name)
    return cfg.get_region(region)


def _shift_photon_pt(selection: str, delta: float) -> tuple[str, int]:
    pattern = re.compile(r"(photon_pt\[(?:0|1)\]\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _shift_regex_group(selection, pattern, delta, min_value=15, max_value=120)


def _shift_lead_photon_pt(selection: str, delta: float) -> tuple[str, int]:
    pattern = re.compile(r"(photon_pt\[0\]\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _shift_regex_group(selection, pattern, delta, min_value=15, max_value=120)


def _shift_sublead_photon_pt(selection: str, delta: float) -> tuple[str, int]:
    pattern = re.compile(r"(photon_pt\[1\]\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _shift_regex_group(selection, pattern, delta, min_value=15, max_value=120)


def _shift_jet_pt(selection: str, delta: float) -> tuple[str, int]:
    pattern = re.compile(r"(jet_pt\[(?:0|1)\]\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _shift_regex_group(selection, pattern, delta, min_value=20, max_value=120)


def _shift_deta_jj(selection: str, delta: float) -> tuple[str, int]:
    pattern = re.compile(r"(fabs\(jet_eta\[0\]-jet_eta\[1\]\)\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _shift_regex_group(selection, pattern, delta, min_value=0, max_value=8)


def _shift_mjj(selection: str, delta: float) -> tuple[str, int]:
    pattern = re.compile(rf"(\({re.escape(MJJ_EXPR)}\)\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _shift_regex_group(selection, pattern, delta, min_value=100, max_value=2000)


def _shift_dphi_gamgam_jj(selection: str, delta: float) -> tuple[str, int]:
    pattern = re.compile(r"(\(acos\(cos\(.+?\)\)\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _shift_regex_group(selection, pattern, delta, min_value=0, max_value=3.14159)


def _shift_ptt(selection: str, delta: float) -> tuple[str, int]:
    pattern = re.compile(rf"(\({re.escape(PTT_EXPR)}\)\s*(?:<=|>=|<|>)\s*)(-?\d+(?:\.\d+)?)")
    return _shift_regex_group(selection, pattern, delta, min_value=0, max_value=250)


def _shift_central_eta(selection: str, delta: float) -> tuple[str, int]:
    pattern = re.compile(
        r"(\(fabs\(photon_eta\[0\]\)\s*<\s*)(-?\d+(?:\.\d+)?)"
        r"(\s*&&\s*fabs\(photon_eta\[1\]\)\s*<\s*)(-?\d+(?:\.\d+)?)(\))"
    )
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 2
        first = _bounded(float(match.group(2)) + delta, 0.1, 2.37)
        second = _bounded(float(match.group(4)) + delta, 0.1, 2.37)
        return (
            f"{match.group(1)}{_format_number(first)}"
            f"{match.group(3)}{_format_number(second)}{match.group(5)}"
        )

    return pattern.sub(replace, selection), replacements


def _shift_transition_eta(selection: str, delta: float) -> tuple[str, int]:
    pattern = re.compile(
        r"(\(\(fabs\(photon_eta\[0\]\)\s*>\s*)(-?\d+(?:\.\d+)?)"
        r"(\s*&&\s*fabs\(photon_eta\[0\]\)\s*<\s*)(-?\d+(?:\.\d+)?)"
        r"(\)\s*\|\|\s*\(fabs\(photon_eta\[1\]\)\s*>\s*)(-?\d+(?:\.\d+)?)"
        r"(\s*&&\s*fabs\(photon_eta\[1\]\)\s*<\s*)(-?\d+(?:\.\d+)?)(\)\))"
    )
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 4
        values = [
            _bounded(float(match.group(2)) + delta, 0.0, 2.37),
            _bounded(float(match.group(4)) + delta, 0.0, 2.37),
            _bounded(float(match.group(6)) + delta, 0.0, 2.37),
            _bounded(float(match.group(8)) + delta, 0.0, 2.37),
        ]
        return (
            f"{match.group(1)}{_format_number(values[0])}"
            f"{match.group(3)}{_format_number(values[1])}"
            f"{match.group(5)}{_format_number(values[2])}"
            f"{match.group(7)}{_format_number(values[3])}{match.group(9)}"
        )

    return pattern.sub(replace, selection), replacements


def _shift_photon_eta_max(selection: str, delta: float) -> tuple[str, int]:
    pattern = re.compile(
        r"(fabs\(photon_eta\[(?:0|1)\]\)\s*<\s*)(2\.37|-?\d+(?:\.\d+)?)"
    )
    return _shift_regex_group(selection, pattern, delta, min_value=1.0, max_value=2.5)


def _shift_regex_group(
    selection: str,
    pattern: re.Pattern[str],
    delta: float,
    *,
    min_value: float,
    max_value: float,
) -> tuple[str, int]:
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        old_value = float(match.group(2))
        new_value = min(max(old_value + delta, min_value), max_value)
        replacements += 1
        return f"{match.group(1)}{_format_number(new_value)}"

    return pattern.sub(replace, selection), replacements


def _scale_regex_group(
    selection: str,
    pattern: re.Pattern[str],
    percent: float,
    *,
    min_value: float,
    max_value: float,
) -> tuple[str, int]:
    replacements = 0
    scale = 1.0 + percent / 100.0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        old_value = float(match.group(2))
        new_value = min(max(old_value * scale, min_value), max_value)
        replacements += 1
        return f"{match.group(1)}{_format_number(new_value)}"

    return pattern.sub(replace, selection), replacements


def _scale_lead_photon_pt(selection: str, percent: float) -> tuple[str, int]:
    pattern = re.compile(r"(photon_pt\[0\]\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _scale_regex_group(selection, pattern, percent, min_value=15, max_value=120)


def _scale_sublead_photon_pt(selection: str, percent: float) -> tuple[str, int]:
    pattern = re.compile(r"(photon_pt\[1\]\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _scale_regex_group(selection, pattern, percent, min_value=15, max_value=120)


def _scale_photon_eta_max(selection: str, percent: float) -> tuple[str, int]:
    pattern = re.compile(r"(fabs\(photon_eta\[(?:0|1)\]\)\s*<\s*)(2\.37|-?\d+(?:\.\d+)?)")
    return _scale_regex_group(selection, pattern, percent, min_value=1.0, max_value=2.5)


def _scale_ptt(selection: str, percent: float) -> tuple[str, int]:
    pattern = re.compile(rf"(\({re.escape(PTT_EXPR)}\)\s*(?:<=|>=|<|>)\s*)(-?\d+(?:\.\d+)?)")
    return _scale_regex_group(selection, pattern, percent, min_value=0, max_value=250)


def _scale_jet_pt(selection: str, percent: float) -> tuple[str, int]:
    pattern = re.compile(r"(jet_pt\[(?:0|1)\]\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _scale_regex_group(selection, pattern, percent, min_value=20, max_value=120)


def _scale_deta_jj(selection: str, percent: float) -> tuple[str, int]:
    pattern = re.compile(r"(fabs\(jet_eta\[0\]-jet_eta\[1\]\)\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _scale_regex_group(selection, pattern, percent, min_value=0, max_value=8)


def _scale_mjj(selection: str, percent: float) -> tuple[str, int]:
    pattern = re.compile(rf"(\({re.escape(MJJ_EXPR)}\)\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _scale_regex_group(selection, pattern, percent, min_value=100, max_value=2000)


def _scale_dphi_gamgam_jj(selection: str, percent: float) -> tuple[str, int]:
    pattern = re.compile(r"(\(acos\(cos\(.+?\)\)\s*>\s*)(-?\d+(?:\.\d+)?)")
    return _scale_regex_group(selection, pattern, percent, min_value=0, max_value=3.14159)


def _build_cut(cut_name: str, percent: float) -> str:
    scale = 1.0 + percent / 100.0
    if cut_name == "lead_photon_pt":
        return f"photon_pt[0]>{_format_number(_bounded(40 * scale, 15, 120))}"
    if cut_name == "sublead_photon_pt":
        return f"photon_pt[1]>{_format_number(_bounded(30 * scale, 15, 120))}"
    if cut_name == "photon_eta_max":
        return (
            f"fabs(photon_eta[0])<{_format_number(_bounded(2.37 * scale, 1.0, 2.5))} && "
            f"fabs(photon_eta[1])<{_format_number(_bounded(2.37 * scale, 1.0, 2.5))}"
        )
    if cut_name == "ptt":
        return f"({PTT_EXPR})>{_format_number(_bounded(40 * scale, 0, 250))}"
    if cut_name == "jet_pt":
        value = _format_number(_bounded(25 * scale, 20, 120))
        return f"jet_pt[0]>{value} && jet_pt[1]>{value}"
    if cut_name == "deta_jj":
        return f"fabs(jet_eta[0]-jet_eta[1])>{_format_number(_bounded(2.8 * scale, 0, 8))}"
    if cut_name == "mjj":
        return f"({MJJ_EXPR})>{_format_number(_bounded(400 * scale, 100, 2000))}"
    if cut_name == "dphi_gamgam_jj":
        return f"({DPHI_GAMGAM_JJ_EXPR})>{_format_number(_bounded(2.6 * scale, 0, 3.14159))}"
    raise ValueError(f"Unknown cut_name {cut_name!r}")


def _remove_pattern(cut_name: str) -> str:
    patterns = {
        "lead_photon_pt": r"photon_pt\[0\]\s*>\s*-?\d+(?:\.\d+)?",
        "sublead_photon_pt": r"photon_pt\[1\]\s*>\s*-?\d+(?:\.\d+)?",
        "photon_eta_max": r"fabs\(photon_eta\[(?:0|1)\]\)\s*<\s*-?\d+(?:\.\d+)?",
        "ptt": rf"\({re.escape(PTT_EXPR)}\)\s*(?:<=|>=|<|>)\s*-?\d+(?:\.\d+)?",
        "jet_pt": r"jet_pt\[(?:0|1)\]\s*>\s*-?\d+(?:\.\d+)?",
        "deta_jj": r"fabs\(jet_eta\[0\]-jet_eta\[1\]\)\s*>\s*-?\d+(?:\.\d+)?",
        "mjj": rf"\({re.escape(MJJ_EXPR)}\)\s*>\s*-?\d+(?:\.\d+)?",
        "dphi_gamgam_jj": r"\(acos\(cos\(.+?\)\)\s*>\s*-?\d+(?:\.\d+)?",
    }
    try:
        return patterns[cut_name]
    except KeyError as exc:
        raise ValueError(f"Unknown cut_name {cut_name!r}") from exc


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def _bounded(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


_CUT_EDITORS: dict[str, Callable[[str, float], tuple[str, int]]] = {
    "photon_pt": _shift_photon_pt,
    "lead_photon_pt": _shift_lead_photon_pt,
    "sublead_photon_pt": _shift_sublead_photon_pt,
    "photon_eta_max": _shift_photon_eta_max,
    "jet_pt": _shift_jet_pt,
    "deta_jj": _shift_deta_jj,
    "mjj": _shift_mjj,
    "dphi_gamgam_jj": _shift_dphi_gamgam_jj,
    "ptt": _shift_ptt,
    "central_eta": _shift_central_eta,
    "transition_eta": _shift_transition_eta,
}

_CUT_SCALERS: dict[str, Callable[[str, float], tuple[str, int]]] = {
    "lead_photon_pt": _scale_lead_photon_pt,
    "sublead_photon_pt": _scale_sublead_photon_pt,
    "photon_eta_max": _scale_photon_eta_max,
    "ptt": _scale_ptt,
    "jet_pt": _scale_jet_pt,
    "deta_jj": _scale_deta_jj,
    "mjj": _scale_mjj,
    "dphi_gamgam_jj": _scale_dphi_gamgam_jj,
}
