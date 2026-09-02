"""Mock tool executor for the tau2-bench retail domain.

Executes tool calls against an in-memory `RetailDB` snapshot using tau2's own
`RetailTools` implementation directly, so results and DB side effects are exactly
what the real benchmark would produce. Deliberately does not use `tau2.gym` or the
`tau2 run` orchestrator (no live user-simulator, no conversational harness) -- those
are reserved for the final live-benchmark evaluation. This module is the harness a
synthetic scenario's expected action gets validated against, and the harness the
reward function calls into during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, ValidationError
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.environment.toolkit import ToolType


@dataclass
class ToolResult:
    """Outcome of executing one tool call against a DB snapshot."""

    ok: bool
    tool_name: str
    arguments: dict[str, Any]
    value: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    schema_valid: Optional[bool] = None
    mutates_state: bool = False


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


class UnknownToolError(KeyError):
    pass


class RetailEnv:
    """Live, stateful wrapper around one `RetailDB` snapshot.

    Construct with a specific `db` to control the starting state (e.g. for
    interactive gold-answer authoring in Phase 2, or a copy under test in the
    reward function); omit it to load the real shipped `db.json`.
    """

    def __init__(self, db: Optional[RetailDB] = None):
        self.db: RetailDB = db if db is not None else RetailDB.load(RETAIL_DB_PATH)
        self._toolkit = RetailTools(self.db)
        self._tools = self._toolkit.get_tools()

    # ---- snapshotting -----------------------------------------------------

    @classmethod
    def from_db_path(cls, path: str) -> "RetailEnv":
        return cls(db=RetailDB.load(path))

    def snapshot(self) -> RetailDB:
        """A deep copy of the current DB state, safe for the caller to mutate
        or keep without affecting this env."""
        return self.db.model_copy(deep=True)

    def reset(self, db: Optional[RetailDB] = None) -> None:
        """Rebind this env to `db` (or reload the shipped db.json)."""
        self.db = db if db is not None else RetailDB.load(RETAIL_DB_PATH)
        self._toolkit = RetailTools(self.db)
        self._tools = self._toolkit.get_tools()

    def db_hash(self) -> str:
        """tau2's own DB hash -- a cheap 'did anything change at all' check."""
        return self._toolkit.get_db_hash()

    # ---- tool metadata ------------------------------------------------------

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def tool_type(self, name: str) -> ToolType:
        return self._toolkit.tool_type(name)

    def tool_mutates_state(self, name: str) -> bool:
        """Whether this tool has a DB-visible side effect. False for read-only
        or generic tools (e.g. `get_order_details`, `transfer_to_human_agents`)
        -- those have no resulting end state to diff, so callers comparing
        outcomes should fall back to argument matching for them."""
        return self._toolkit.tool_mutates_state(name)

    def openai_schema(self, name: str) -> dict[str, Any]:
        """The exact OpenAI-style function schema tau2 hands the agent --
        the same tool interface real baselines were tested against."""
        return self._tools[name].openai_schema

    def all_openai_schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].openai_schema for name in self.tool_names()]

    def args_model(self, name: str) -> type[BaseModel]:
        """The pydantic model tau2 auto-derives from the tool's own signature
        and docstring -- ground truth for argument schema validity."""
        return self._tools[name].params

    def validate_arguments(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate `arguments` against the tool's real parameter model.

        Note: pydantic's default `extra` policy on an auto-`create_model`
        (as tau2 builds `Tool.params`) is 'ignore', not 'forbid' -- so this
        alone will NOT flag an extra/hallucinated argument the tool doesn't
        accept. Use `extra_arguments()` for that; both checks matter and are
        tested independently, see `tests/test_retail_env.py`.
        """
        if name not in self._tools:
            return False, f"Unknown tool: {name}"
        try:
            self._tools[name].params.model_validate(arguments)
        except ValidationError as e:
            return False, str(e)
        return True, None

    def extra_arguments(self, name: str, arguments: dict[str, Any]) -> list[str]:
        """Argument keys not recognized by the tool's schema at all -- the
        signal a reward function's hallucination penalty should key off."""
        if name not in self._tools:
            return list(arguments)
        known = set(self._tools[name].params.model_fields)
        return [k for k in arguments if k not in known]

    # ---- execution ------------------------------------------------------

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute tool_name(**arguments) against this env's live DB, mutating
        it in place exactly as the real RetailTools implementation would.

        Never raises for expected tool-level failures (unknown tool, bad
        schema, invalid id, wrong state) -- those come back as a failed
        ToolResult so callers (rule checker, reward function) can inspect them
        uniformly instead of catching exceptions themselves.
        """
        if tool_name not in self._tools:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                arguments=arguments,
                error=f"Unknown tool: {tool_name}. Available: {self.tool_names()}",
                error_type="UnknownTool",
            )

        mutates = self.tool_mutates_state(tool_name)
        schema_valid, _schema_err = self.validate_arguments(tool_name, arguments)

        tool = self._tools[tool_name]
        try:
            value = tool(**arguments)
        except (ValueError, TypeError) as e:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                arguments=arguments,
                error=str(e),
                error_type=type(e).__name__,
                schema_valid=schema_valid,
                mutates_state=mutates,
            )
        return ToolResult(
            ok=True,
            tool_name=tool_name,
            arguments=arguments,
            value=_to_jsonable(value),
            schema_valid=schema_valid,
            mutates_state=mutates,
        )


def execute_against(
    db: RetailDB, tool_name: str, arguments: dict[str, Any]
) -> tuple[ToolResult, RetailDB]:
    """Stateless: run one tool call against a deep copy of `db`, never mutating
    `db` itself. Returns `(result, resulting_db)` -- `resulting_db` is the
    (deep-copied) starting state unchanged if the call failed.

    This is the primitive a reward function uses to compare a rollout's and
    the gold action's end states starting from the *same* snapshot, e.g.:

        predicted_result, predicted_db = execute_against(db_state, rollout.tool_name, rollout.tool_input)
        gold_result, gold_db = execute_against(db_state, expected.tool_name, expected.tool_input)
        same_outcome = predicted_result.ok and gold_result.ok and predicted_db == gold_db
    """
    working = db.model_copy(deep=True)
    result = RetailEnv(db=working).execute(tool_name, arguments)
    return result, working
