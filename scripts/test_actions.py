#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from agent.Actions import Action, apply_analysis_action
from agent.trex_editor import TrexConfigEditor


def main() -> None:
    cfg = TrexConfigEditor((PROJECT_DIR / "configs" / "hyy.config").read_text())
    action = Action.modify_category_cut(
        region_index=0,
        cut_name="jet_pt",
        percent=10,
    )

    assert str(action) == "modify_category_cut: region_index=0, cut=jet_pt, scale=+10%"
    assert str(Action.add_category_cut(region_index=0, cut_name="mjj", percent=20)) == (
        "add_category_cut: region_index=0, cut=mjj, scale=+20%"
    )
    assert str(Action.remove_category_cut(region_index=0, cut_name="mjj")) == (
        "remove_category_cut: region_index=0, cut=mjj"
    )
    assert str(Action.add_category(region_index=0)) == "add_category: region_index=0"
    assert str(Action.remove_category(region_index=0)) == "remove_category: region_index=0"
    assert Action.from_dict(action.to_dict()) == action

    before = cfg.get_selection(0)
    result = action.apply(cfg)
    after = cfg.get_selection(0)
    assert result.replacements == 2
    assert before != after
    assert apply_analysis_action(cfg, action.to_dict()).region == result.region

    before = cfg.get_selection(0)
    add_cut_result = Action.add_category_cut(region_index=0, cut_name="mjj", percent=10).apply(cfg)
    assert add_cut_result.replacements == 1
    assert before != cfg.get_selection(0)

    remove_cut_result = Action.remove_category_cut(region_index=0, cut_name="mjj").apply(cfg)
    assert remove_cut_result.replacements == 1

    region_count = len(cfg.get_regions())
    add_result = Action.add_category(region_index=0).apply(cfg)
    assert len(cfg.get_regions()) == region_count + 1
    assert add_result.region.endswith("_rl1")

    remove_result = Action.remove_category(region_index=0).apply(cfg)
    assert len(cfg.get_regions()) == region_count
    assert remove_result.cut_name == "category"

    print("PASS: Action class strings, dict conversion, and apply paths work")


if __name__ == "__main__":
    main()
