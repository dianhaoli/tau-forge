"""Phase 6 data prep: convert the real tau2-bench retail *train* tasks into the
static, trusted scenario format the harness (`tau_forge.harness`) trains and
smoke-tests against.

This is the first point in the project where the real tasks' content (user
scenario text, gold action sequences) is read -- deliberately, and only for
the **train** split. The Phase 0/5 decontamination discipline is about never
letting task content leak into the *synthetic-data generator*'s prompts;
grounding RLVR training directly on the real train split is the whole point
of Phase 6 and is not in tension with that rule. `tasks.json` test-split
content must still never be read anywhere in this repo.

All 114 retail tasks ship with `initial_state=None` (confirmed in Phase 0),
meaning every task starts from the same shipped `db.json` -- there is no
per-task DB snapshot to extract. The actual "static snapshot" conversion
this phase does is: pull each train task's gold action sequence and rendered
user-scenario prompt out of tau2's own `Task` objects (reusing tau2's own
parsing/rendering, not reimplementing it) into a small, stable JSON file the
rest of tau-forge can depend on without importing tau2's runner package.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tau2.domains.retail.environment import get_tasks

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "trusted" / "train_tasks.json"


@dataclass
class GoldAction:
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class TrustedTask:
    """One real tau2-bench retail train task, reduced to what the Phase 6/7
    harness needs: a rendered user-scenario prompt and the gold action
    sequence to teacher-force turn-level rewards against. `reward_basis`
    and `communicate_info` are carried through for visibility even though
    the harness currently only trains against DB-outcome (per-action)
    reward, not the full task-level COMMUNICATE/NL_ASSERTION criteria those
    describe -- that's tau2's own final-task grading, reserved for Phase 8."""

    task_id: str
    prompt: str
    gold_actions: list[GoldAction]
    reward_basis: list[str]
    communicate_info: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "gold_actions": [asdict(a) for a in self.gold_actions],
            "reward_basis": self.reward_basis,
            "communicate_info": self.communicate_info,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "TrustedTask":
        return cls(
            task_id=d["task_id"],
            prompt=d["prompt"],
            gold_actions=[GoldAction(a["tool_name"], a["arguments"]) for a in d["gold_actions"]],
            reward_basis=d["reward_basis"],
            communicate_info=d["communicate_info"],
        )


def load_trusted_train_tasks() -> list[TrustedTask]:
    """Load the 74 real retail train tasks straight from tau2, reduced to
    `TrustedTask`s. Skips any task whose gold trajectory includes a
    non-assistant (user/env) action or has zero actions -- the harness only
    grades assistant tool calls, and a task with no actions has nothing to
    teacher-force against. Both are asserted absent for the current pinned
    tau2 commit; see `tests/test_trusted_tasks.py`."""
    tasks = get_tasks("train")
    out: list[TrustedTask] = []
    for t in tasks:
        actions = t.evaluation_criteria.actions or []
        gold_actions = [GoldAction(a.name, dict(a.arguments)) for a in actions if a.requestor == "assistant"]
        out.append(
            TrustedTask(
                task_id=t.id,
                prompt=str(t.user_scenario),
                gold_actions=gold_actions,
                reward_basis=[r.value for r in t.evaluation_criteria.reward_basis],
                communicate_info=list(t.evaluation_criteria.communicate_info or []),
            )
        )
    return out


def build_and_save(out_path: Path = DEFAULT_OUTPUT_PATH) -> list[TrustedTask]:
    tasks = load_trusted_train_tasks()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([t.to_json() for t in tasks], indent=2) + "\n")
    return tasks


def load_saved(path: Path = DEFAULT_OUTPUT_PATH) -> list[TrustedTask]:
    data = json.loads(path.read_text())
    return [TrustedTask.from_json(d) for d in data]


if __name__ == "__main__":
    tasks = build_and_save()
    print(f"Wrote {len(tasks)} trusted train tasks to {DEFAULT_OUTPUT_PATH}")
