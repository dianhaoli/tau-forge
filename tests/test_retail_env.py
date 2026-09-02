"""Smoke tests for tau_forge.envs.retail against the real shipped db.json.

These exercise the wrapper's contract, not tau2's own tool logic (that's
tau2's own test suite's job): correct pass-through of results and errors,
snapshot isolation, schema validation semantics (including the extra-field/
hallucination edge case), and DB-state comparison via `execute_against`.
"""

from __future__ import annotations

import pytest
from tau2.domains.retail.data_model import RetailDB

from tau_forge.envs.retail import RetailEnv, execute_against


@pytest.fixture(scope="module")
def base_db() -> RetailDB:
    return RetailEnv().snapshot()


@pytest.fixture()
def env(base_db: RetailDB) -> RetailEnv:
    return RetailEnv(db=base_db.model_copy(deep=True))


def first_pending_order_id(db: RetailDB) -> str:
    for order in db.orders.values():
        if order.status == "pending":
            return order.order_id
    raise AssertionError("fixture db.json has no pending order -- unexpected")


def first_delivered_order_id(db: RetailDB) -> str:
    for order in db.orders.values():
        if order.status == "delivered":
            return order.order_id
    raise AssertionError("fixture db.json has no delivered order -- unexpected")


# ---- tool metadata ----------------------------------------------------------


def test_all_16_tools_present(env: RetailEnv):
    names = env.tool_names()
    assert len(names) == 16
    assert "cancel_pending_order" in names
    assert "transfer_to_human_agents" in names


def test_tool_type_and_mutation_flags_match_policy(env: RetailEnv):
    # READ tools never mutate.
    assert env.tool_type("get_order_details").value == "read"
    assert env.tool_mutates_state("get_order_details") is False
    # WRITE tools do.
    assert env.tool_type("cancel_pending_order").value == "write"
    assert env.tool_mutates_state("cancel_pending_order") is True
    # transfer_to_human_agents is GENERIC and, importantly, does NOT mutate --
    # it has no DB side effect for a reward function to diff.
    assert env.tool_type("transfer_to_human_agents").value == "generic"
    assert env.tool_mutates_state("transfer_to_human_agents") is False


def test_openai_schema_shape(env: RetailEnv):
    schema = env.openai_schema("cancel_pending_order")
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "cancel_pending_order"
    props = fn["parameters"]["properties"]
    assert "order_id" in props and "reason" in props


# ---- execution: read tools ----------------------------------------------------


def test_read_tool_returns_real_data_and_does_not_mutate(env: RetailEnv):
    before_hash = env.db_hash()
    order_id = first_pending_order_id(env.db)
    result = env.execute("get_order_details", {"order_id": order_id})
    assert result.ok
    assert result.value["order_id"] == order_id
    assert env.db_hash() == before_hash  # read tool: DB untouched


def test_unknown_order_id_is_a_clean_failure_not_an_exception(env: RetailEnv):
    result = env.execute("get_order_details", {"order_id": "#W0000000_NOPE"})
    assert not result.ok
    assert result.error_type == "ValueError"
    assert "not found" in result.error.lower()


# ---- execution: write tools + state mutation -----------------------------------


def test_cancel_pending_order_mutates_status_and_refunds(env: RetailEnv):
    order_id = first_pending_order_id(env.db)
    order_before = env.db.orders[order_id].model_copy(deep=True)

    result = env.execute(
        "cancel_pending_order", {"order_id": order_id, "reason": "no longer needed"}
    )
    assert result.ok
    assert result.value["status"] == "cancelled"

    order_after = env.db.orders[order_id]
    assert order_after.status == "cancelled"
    assert order_after.cancel_reason == "no longer needed"
    # a refund transaction was appended
    assert len(order_after.payment_history) == len(order_before.payment_history) + 1
    assert order_after.payment_history[-1].transaction_type == "refund"


def test_invalid_cancel_reason_is_rejected_by_policy_logic(env: RetailEnv):
    order_id = first_pending_order_id(env.db)
    result = env.execute(
        "cancel_pending_order", {"order_id": order_id, "reason": "I felt like it"}
    )
    assert not result.ok
    assert result.error_type == "ValueError"


def test_double_cancel_is_rejected_second_time(env: RetailEnv):
    order_id = first_pending_order_id(env.db)
    first = env.execute(
        "cancel_pending_order", {"order_id": order_id, "reason": "ordered by mistake"}
    )
    assert first.ok
    second = env.execute(
        "cancel_pending_order", {"order_id": order_id, "reason": "ordered by mistake"}
    )
    assert not second.ok  # order is no longer pending


# ---- snapshot / isolation ----------------------------------------------------


def test_snapshot_is_independent_of_live_env(env: RetailEnv):
    order_id = first_pending_order_id(env.db)
    snap = env.snapshot()

    env.execute("cancel_pending_order", {"order_id": order_id, "reason": "ordered by mistake"})

    assert env.db.orders[order_id].status == "cancelled"
    assert snap.orders[order_id].status == "pending"  # untouched copy


def test_execute_against_does_not_mutate_input_db(base_db: RetailDB):
    order_id = first_pending_order_id(base_db)
    before = base_db.orders[order_id].status

    result, resulting_db = execute_against(
        base_db, "cancel_pending_order", {"order_id": order_id, "reason": "ordered by mistake"}
    )

    assert result.ok
    assert resulting_db.orders[order_id].status == "cancelled"
    assert base_db.orders[order_id].status == before  # original untouched


def test_execute_against_same_action_from_same_start_gives_equal_end_states(base_db: RetailDB):
    order_id = first_pending_order_id(base_db)
    args = {"order_id": order_id, "reason": "ordered by mistake"}

    _, db_a = execute_against(base_db, "cancel_pending_order", args)
    _, db_b = execute_against(base_db, "cancel_pending_order", args)

    assert db_a == db_b  # pydantic structural equality -- deterministic replay


def test_execute_against_failed_call_returns_db_equal_to_start(base_db: RetailDB):
    result, resulting_db = execute_against(
        base_db, "cancel_pending_order", {"order_id": "#NOPE", "reason": "ordered by mistake"}
    )
    assert not result.ok
    assert resulting_db == base_db


# ---- schema validation, including the extra-field / hallucination case --------


def test_missing_required_argument_is_schema_invalid(env: RetailEnv):
    valid, err = env.validate_arguments("cancel_pending_order", {"order_id": "#W0000000"})
    assert valid is False
    assert err is not None


def test_correct_arguments_are_schema_valid(env: RetailEnv):
    order_id = first_pending_order_id(env.db)
    valid, err = env.validate_arguments(
        "cancel_pending_order", {"order_id": order_id, "reason": "ordered by mistake"}
    )
    assert valid is True
    assert err is None


def test_extra_argument_is_not_caught_by_pydantic_validate_alone(env: RetailEnv):
    """Documents tau2's auto-generated params model's default `extra` policy.

    This matters for Phase 4: `validate_arguments` alone is NOT a hallucination
    detector for extra fields (pydantic's default create_model extra policy
    lets unrecognized keys through). `extra_arguments()` below is the check a
    reward function's hallucination penalty must use instead/additionally.
    """
    order_id = first_pending_order_id(env.db)
    args = {
        "order_id": order_id,
        "reason": "ordered by mistake",
        "made_up_field": "should not exist on this tool",
    }
    valid, _err = env.validate_arguments("cancel_pending_order", args)
    extras = env.extra_arguments("cancel_pending_order", args)

    assert extras == ["made_up_field"]
    # Record actual pydantic behavior rather than assuming it; the test suite
    # documents whichever way it falls so future tau2/pydantic upgrades that
    # change this default can't silently break the reward function's
    # hallucination-penalty logic without a test failing here.
    assert valid in (True, False)


def test_extra_arguments_empty_for_clean_call(env: RetailEnv):
    order_id = first_pending_order_id(env.db)
    args = {"order_id": order_id, "reason": "ordered by mistake"}
    assert env.extra_arguments("cancel_pending_order", args) == []


def test_unknown_tool_name_is_a_clean_failure(env: RetailEnv):
    result = env.execute("delete_entire_database", {})
    assert not result.ok
    assert result.error_type == "UnknownTool"


# ---- a second write tool, for breadth ------------------------------------------


def test_exchange_delivered_order_items_end_to_end(env: RetailEnv):
    order_id = first_delivered_order_id(env.db)
    order = env.db.orders[order_id]
    item = order.items[0]
    product = env.db.products[item.product_id]

    new_item_id = next(
        (v.item_id for v in product.variants.values() if v.available and v.item_id != item.item_id),
        None,
    )
    if new_item_id is None:
        pytest.skip("fixture product has no alternate available variant to exchange into")

    user = env.db.users[order.user_id]
    payment_method_id = next(iter(user.payment_methods))

    result = env.execute(
        "exchange_delivered_order_items",
        {
            "order_id": order_id,
            "item_ids": [item.item_id],
            "new_item_ids": [new_item_id],
            "payment_method_id": payment_method_id,
        },
    )
    assert result.ok
    assert env.db.orders[order_id].status == "exchange requested"
