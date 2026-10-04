"""The episode task record and the constants every episode module shares.

An `EpisodeTask` is everything needed to run and grade one conversation:
the opening user message (the only part that is a fixed prompt), the hidden
profile the scripted user answers from, the canonical gold action chain and
the tau2 end-state hash that chain produces. The gold chain is kept for
verification and for reference agents only -- grading compares end states,
exactly as tau2's `EnvironmentEvaluator` does, so any action sequence that
reaches the gold state scores.
"""

from __future__ import annotations

import functools
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH

AUTH_TOOLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})
READ_TOOLS = frozenset(
    AUTH_TOOLS
    | {
        "get_user_details",
        "get_order_details",
        "get_product_details",
        "get_item_details",
        "list_all_product_types",
        "calculate",
    }
)
WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "return_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
    }
)
TRANSFER_TOOL = "transfer_to_human_agents"

# tau2's user-simulator stop token (`tau2.user.user_simulator_base.STOP`).
STOP = "###STOP###"

# Every database id shape the retail tools take as an argument: order ids,
# 10-digit item/product ids, payment method ids, user ids. Emails, zips (5
# digits), card last-fours (4) and tracking ids (12) deliberately do not match,
# so "the user never says an id" can be checked with this one pattern.
ID_RE = re.compile(
    r"#W\d{7}|\b\d{10}\b|\b(?:credit_card|gift_card|paypal)_\d{7}\b|\b[a-z]+_[a-z]+_\d{4}\b"
)
ORDER_ID_RE = re.compile(r"#W\d{7}")

TEMPLATES = ("cancel", "exchange", "return_fallback", "modify_payment", "foreign_order_refusal")


@functools.lru_cache(maxsize=1)
def base_db() -> RetailDB:
    """The shipped db.json, loaded once per process (0.22 s) and SHARED.

    Never mutate it: generators read it, the runner builds copy-on-write views
    over it, and `tests/test_episodes.py` asserts its hash is unchanged after
    a full batch of write-heavy episodes."""
    return RetailDB.load(RETAIL_DB_PATH)


@functools.lru_cache(maxsize=1)
def base_db_hash() -> str:
    from tau_forge.envs.retail import RetailEnv

    return RetailEnv(db=base_db()).db_hash()


@dataclass
class EpisodeTask:
    id: str
    template: str
    user_id: str
    opening: str
    # What the scripted user knows and says: paraphrase pools (lists of
    # strings) keyed by intent, plus the recap keys -- see `user.py`.
    profile: dict[str, Any]
    # Canonical chain, reads and writes. Verification replays it; grading
    # never compares against it action by action.
    gold_actions: list[dict[str, Any]]
    target_order: str
    expect_no_write: bool = False
    difficulty: dict[str, Any] = field(default_factory=dict)
    gold_db_hash: str = ""
    # Natural-language targets only reference agents read (never rendered to
    # the policy). The ids those agents use still come from tool outputs.
    hidden: dict[str, Any] = field(default_factory=dict)
    # Users whose records the task touches (the requester, plus the owner of a
    # foreign order) -- what decontamination checks against.
    involved_users: list[str] = field(default_factory=list)

    @property
    def dedupe_key(self) -> tuple:
        items = tuple(sorted(self.hidden.get("item_ids", [])))
        return (self.template, self.user_id, self.target_order, items)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodeTask":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
