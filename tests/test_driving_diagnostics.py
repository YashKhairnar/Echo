import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.data_collection.show_visible_neighbors import SLOT_ORDER
from scripts.evaluation.diagnose_driving import classify_crash, select_driving


class DiagnosticTests(unittest.TestCase):
    def test_rule_none_explicitly_disables_private_previews(self):
        env = SimpleNamespace(request_neighbor_action=lambda _: self.fail("Private request in rule_none"))
        with patch("scripts.evaluation.diagnose_driving.choose_action", return_value=(1, {})) as choose:
            action, message, _, values = select_driving("rule_none", env, {}, None, None, None, random.Random(0))
        self.assertEqual(action, 1)
        self.assertEqual(choose.call_args.kwargs["messages"], {})
        self.assertIsNone(message)
        self.assertIsNone(values)

    def test_rule_all_uses_requested_ids_and_preserves_empty_slots(self):
        visible = {SLOT_ORDER[7]: {"vehicle": SimpleNamespace(vehicle_id=17)}}
        env = SimpleNamespace(request_neighbor_action=lambda vid: {"vehicle_id": vid})
        with patch("scripts.evaluation.diagnose_driving.choose_action", return_value=(1, {})) as choose:
            _, bundle, _, _ = select_driving("rule_all", env, visible, None, None, None, random.Random(0))
        self.assertEqual(choose.call_args.kwargs["messages"], {17: {"vehicle_id": 17}})
        self.assertEqual(bundle["messages_by_slot"], [None] * 7 + [{"vehicle_id": 17}])

    def test_crash_geometry_and_unknown_partner(self):
        before = {"ego": {"lane": 1, "target_lane": 1, "position": [0, 4]},
                  "neighbors": [{"vehicle_id": 7, "lane": 1, "dx": 10, "crashed": False}]}
        after = {"ego": {"position": [0, 4]},
                 "neighbors": [{"vehicle_id": 7, "distance": 3, "crashed": True}]}
        self.assertEqual(classify_crash(before, after, 1)["category"], "front_contact")
        before["neighbors"][0]["dx"] = -10
        self.assertEqual(classify_crash(before, after, 1)["category"], "rear_contact")
        before["ego"]["target_lane"] = 2
        self.assertEqual(classify_crash(before, after, 2)["category"], "lane_change_or_lateral_conflict")
        after["neighbors"][0]["distance"] = 100
        self.assertEqual(classify_crash(before, after, 2)["category"], "unknown")


if __name__ == "__main__":
    unittest.main()
