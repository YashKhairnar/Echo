import json
import tempfile
import unittest
from pathlib import Path

from scripts.docs.plot_training_graphs import load_dqn_history


class GraphHistoryTests(unittest.TestCase):
    def log(self, root, name, episodes, encoding="spatial_v1", start=0):
        run = root / name
        run.mkdir()
        path = run / "request_dqn_training.csv"
        path.write_text("episode,return\n" + "".join(f"{episode},1\n" for episode in episodes))
        (run / "config.json").write_text(json.dumps({
            "request_action_encoding": encoding, "start_episode": start,
            "resume_from": str(root / "active" / "best.pt"),
        }))
        return path

    def test_follows_resume_and_excludes_compressed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.log(root, "01_old", [1, 2], "compressed_v0")
            first = self.log(root, "02_fixed", [1, 2])
            latest = self.log(root, "03_resume", [3, 4], start=2)
            rows, paths, boundaries = load_dqn_history(latest)
            self.assertEqual([int(row["episode"]) for row in rows], [1, 2, 3, 4])
            self.assertEqual(paths, [first, latest])
            self.assertEqual(boundaries, [2])

    def test_rejects_ambiguous_history_and_overlapping_explicit_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.log(root, "01", [1, 2])
            other = self.log(root, "02", [1, 2])
            latest = self.log(root, "03", [3, 4], start=2)
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                load_dqn_history(latest)
            with self.assertRaisesRegex(ValueError, "without duplicates"):
                load_dqn_history(latest, [first, other, latest])
            rows, _, _ = load_dqn_history(latest, [first, latest])
            self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
