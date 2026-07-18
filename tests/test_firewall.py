"""Behavioural tests for the deterministic action firewall.

These are the checks that eyeballing does not catch: silent misattribution in
the ledger, and checks that quietly never fire. Both coworker harnesses lost
versions to string/structure bugs that looked fine on inspection, so assert on
behaviour, not on shape.

Run:  uv run python -m unittest tests.test_firewall -v
"""
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "my_agent"))

from firewall import (  # noqa: E402
    CHECK_KINDS, ProvenanceIndex, StateLedger, check_action, entity_vocabulary,
    suspect_entities, unresolved_required_args,
)


def _call(name, args, call_id=None):
    tc = {"function": {"name": name, "arguments": json.dumps(args)}}
    if call_id:
        tc["id"] = call_id
    return tc


def _ok(result):
    return json.dumps({"status": "SUCCESS", "result": result})


class TestStateLedger(unittest.TestCase):
    def test_results_matched_by_id_not_position(self):
        """Out-of-order results must not shift every later tool name by one."""
        messages = [
            {"role": "assistant", "tool_calls": [
                _call("get_climate_settings", {}, "c1"),
                _call("get_vehicle_window_positions", {}, "c2"),
            ]},
            # Provider returns the SECOND call's result first.
            {"role": "tool", "tool_call_id": "c2", "content": _ok({"windows": "closed"})},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"status": "ERROR"})},
        ]
        ledger = StateLedger.from_messages(messages)
        self.assertTrue(ledger.called("get_vehicle_window_positions"))
        # Positional matching would have credited the climate call instead.
        self.assertFalse(ledger.called("get_climate_settings"))
        self.assertIn("get_climate_settings", ledger.failed_tools)

    def test_positional_fallback_without_ids(self):
        messages = [
            {"role": "assistant", "tool_calls": [_call("get_climate_settings", {})]},
            {"role": "tool", "content": _ok({"temperature": 21})},
        ]
        ledger = StateLedger.from_messages(messages)
        self.assertTrue(ledger.called("get_climate_settings"))

    def test_failed_call_does_not_count_as_verified(self):
        messages = [
            {"role": "assistant", "tool_calls": [_call("get_user_preferences", {}, "c1")]},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"status": "ERROR"})},
        ]
        self.assertFalse(StateLedger.from_messages(messages).called("get_user_preferences"))

    def test_malformed_result_is_not_verified(self):
        messages = [
            {"role": "assistant", "tool_calls": [_call("get_user_preferences", {}, "c1")]},
            {"role": "tool", "tool_call_id": "c1", "content": "not json at all"},
        ]
        ledger = StateLedger.from_messages(messages)
        self.assertFalse(ledger.called("get_user_preferences"))


class TestProvenance(unittest.TestCase):
    def _prov(self, texts, results=None):
        messages = [{"role": "user", "content": t} for t in texts]
        ledger = StateLedger()
        for value in results or []:
            ledger.result_scalars.add(value)
        return ProvenanceIndex.build(messages, ledger)

    def test_number_from_user_text_is_supported(self):
        self.assertTrue(self._prov(["set it to 22 degrees"]).supports_number(22))

    def test_invented_number_is_not_supported(self):
        self.assertFalse(self._prov(["make it warmer"]).supports_number(73))

    def test_implied_fully_licenses_100(self):
        self.assertTrue(self._prov(["open the sunroof fully"]).supports_number(100))

    def test_implied_german_licenses_100(self):
        self.assertTrue(self._prov(["mach das Dach ganz auf"]).supports_number(100))

    def test_trivial_numbers_always_supported(self):
        prov = self._prov(["hello"])
        self.assertTrue(prov.supports_number(0))
        self.assertTrue(prov.supports_number(1))

    def test_tool_result_scalars_are_supported(self):
        self.assertTrue(self._prov(["hello"], results=[42.0]).supports_number(42))


class TestCheckAction(unittest.TestCase):
    def setUp(self):
        self.constraints = {
            "preconditions": [{"tool": "set_air_conditioning",
                               "requires_tools": ["get_climate_settings"]}],
            "defaults": [{"tool": "open_sunroof", "argument": "percentage", "value": 50}],
        }
        self.empty_ledger = StateLedger()
        self.prov = ProvenanceIndex.build(
            [{"role": "user", "content": "open the sunroof to 100"}], StateLedger())

    def test_precondition_violation_fires(self):
        v = check_action([_call("set_air_conditioning", {"on": True})],
                         self.empty_ledger, self.prov, self.constraints)
        self.assertEqual([x["kind"] for x in v], ["precondition"])

    def test_precondition_satisfied_in_same_batch(self):
        v = check_action(
            [_call("get_climate_settings", {}), _call("set_air_conditioning", {"on": True})],
            self.empty_ledger, self.prov, self.constraints)
        self.assertEqual(v, [])

    def test_precondition_satisfied_by_ledger(self):
        ledger = StateLedger()
        ledger.successful_tools.append("get_climate_settings")
        v = check_action([_call("set_air_conditioning", {"on": True})],
                         ledger, self.prov, self.constraints)
        self.assertEqual(v, [])

    def test_default_deviation_fires(self):
        v = check_action([_call("open_sunroof", {"percentage": 100})],
                         self.empty_ledger, self.prov, self.constraints)
        self.assertEqual([x["kind"] for x in v], ["default"])
        self.assertIn("50", v[0]["message"])

    def test_matching_the_default_is_silent(self):
        v = check_action([_call("open_sunroof", {"percentage": 50})],
                         self.empty_ledger, self.prov, self.constraints)
        self.assertEqual(v, [])

    def test_default_and_provenance_do_not_double_report(self):
        """A policy default is the specific signal; provenance must not also
        fire on the same argument and double the nudge."""
        v = check_action([_call("open_sunroof", {"percentage": 100})],
                         self.empty_ledger, self.prov, self.constraints)
        self.assertEqual(len(v), 1)

    def test_provenance_fires_on_unsourced_value(self):
        prov = ProvenanceIndex.build([{"role": "user", "content": "warm it up"}], StateLedger())
        v = check_action([_call("set_temperature", {"celsius": 73})],
                         self.empty_ledger, prov, {"preconditions": [], "defaults": []})
        self.assertEqual([x["kind"] for x in v], ["provenance"])

    def test_no_constraints_fails_open(self):
        self.assertEqual(
            check_action([_call("open_sunroof", {"percentage": 100})],
                         self.empty_ledger, self.prov, None),
            [])

    def test_malformed_arguments_fail_open(self):
        bad = {"function": {"name": "open_sunroof", "arguments": "{not json"}}
        self.assertEqual(
            check_action([bad], self.empty_ledger, self.prov, self.constraints), [])

    def test_booleans_are_not_treated_as_numbers(self):
        v = check_action([_call("set_air_conditioning", {"on": True})],
                         self.empty_ledger, self.prov,
                         {"preconditions": [], "defaults": []})
        self.assertEqual(v, [])


class TestCheckGating(unittest.TestCase):
    """Each check must be independently disableable, or it cannot be ablated."""

    def setUp(self):
        self.constraints = {
            "preconditions": [{"tool": "set_ac", "requires_tools": ["get_climate"]}],
            "defaults": [{"tool": "open_sunroof", "argument": "percentage", "value": 50}],
        }
        self.ledger = StateLedger()
        self.prov = ProvenanceIndex.build([{"role": "user", "content": "go"}], StateLedger())

    def test_disabling_default_check_silences_only_it(self):
        calls = [_call("open_sunroof", {"percentage": 100}), _call("set_ac", {})]
        full = check_action(calls, self.ledger, self.prov, self.constraints)
        self.assertEqual({x["kind"] for x in full}, {"default", "precondition"})

        without = check_action(calls, self.ledger, self.prov, self.constraints,
                               enabled={"precondition", "provenance"})
        self.assertEqual({x["kind"] for x in without}, {"precondition"})

    def test_empty_enabled_set_disables_everything(self):
        v = check_action([_call("open_sunroof", {"percentage": 100})],
                         self.ledger, self.prov, self.constraints, enabled=set())
        self.assertEqual(v, [])

    def test_check_kinds_matches_what_check_action_emits(self):
        """Guards against a check being added without a CHECK_KINDS entry —
        it would be silently unablatable."""
        calls = [_call("open_sunroof", {"percentage": 100}), _call("set_ac", {}),
                 _call("set_temperature", {"celsius": 73})]
        emitted = {x["kind"] for x in
                   check_action(calls, self.ledger, self.prov, self.constraints)}
        self.assertTrue(emitted <= CHECK_KINDS, f"unknown kinds: {emitted - CHECK_KINDS}")
        self.assertEqual(emitted, set(CHECK_KINDS))


class TestSuspectEntities(unittest.TestCase):
    TOOLS = [
        {"function": {"name": "open_close_sunshade"}},
        {"function": {"name": "get_climate_settings"}},
    ]

    def test_entity_vocabulary_skips_verbs(self):
        vocab = entity_vocabulary(self.TOOLS)
        self.assertIn("sunshade", vocab)
        self.assertNotIn("open", vocab)
        self.assertNotIn("get", vocab)

    def test_unsupported_entity_is_flagged(self):
        suspects = suspect_entities("I closed the sunshade for you.",
                                    StateLedger(), self.TOOLS)
        self.assertIn("sunshade", suspects)

    def test_entity_backed_by_successful_call_is_clean(self):
        ledger = StateLedger()
        ledger.successful_tools.append("open_close_sunshade")
        self.assertEqual(
            suspect_entities("I closed the sunshade.", ledger, self.TOOLS), [])

    def test_empty_draft_is_clean(self):
        self.assertEqual(suspect_entities("", StateLedger(), self.TOOLS), [])


class TestUnresolvedRequiredArgs(unittest.TestCase):
    SCHEMA = {"parameters": {"type": "object",
                             "properties": {"percentage": {"type": "number"}},
                             "required": ["percentage"]}}

    def test_flags_numeric_arg_with_no_values_available(self):
        prov = ProvenanceIndex()
        self.assertEqual(unresolved_required_args(self.SCHEMA, prov), ["percentage"])

    def test_clean_when_episode_has_numbers(self):
        prov = ProvenanceIndex()
        prov.numbers.add(50.0)
        self.assertEqual(unresolved_required_args(self.SCHEMA, prov), [])

    def test_empty_schema_is_clean(self):
        self.assertEqual(unresolved_required_args({}, ProvenanceIndex()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
