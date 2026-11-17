import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from run_submission import ROOT, attach_cv, digest


class CVAttachmentTests(unittest.TestCase):
    def fixture(self, root, overwrite=False):
        players = np.ones((6, 2, 2))
        ball = np.ones((6, 2))
        ball[2] = np.nan
        teams = ['H', 'A']
        ids = np.arange(1, 4)
        recovered = np.ones((3, 2))
        recovered[1] = [1.2, .8]
        np.savez(root/'observations.npz', frame_ids=ids, ball=recovered,
                 players=players[ids], teams=teams, ball_observed=[True]*3)
        np.savez(root/'quality.npz', frame_ids=ids, baseline_ball=ball[ids],
                 cv_used=[overwrite, True, False], cv_source=['sam2_extended_verified']*3,
                 image_observed=[True]*3)
        manifest = dict(training_eligible=False, camera='tactical', fps=25, frame_start=1, n_frames=3,
                        quality_policy_sha256=digest(ROOT/'scripts/possession_observation_quality.py'),
                        files=dict(observations='observations.npz'), quality_sidecar='quality.npz',
                        quality_outputs_sha256={n: digest(root/n) for n in ['observations.npz', 'quality.npz']})
        path = root/'manifest.json'
        path.write_text(json.dumps(manifest))
        return players, ball, teams, path

    def test_only_qualified_gap_is_filled(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            merged, masks, _ = attach_cv(*args)
            np.testing.assert_allclose(merged[2], [1.2, .8])
            np.testing.assert_array_equal(merged[[0,1,3,4,5]], args[1][[0,1,3,4,5]])
            self.assertEqual(masks['cv_used'].sum(), 1)
            self.assertFalse(masks['cv_available'][0])
            self.assertTrue(np.isnan(args[1][2]).all())

    def test_existing_backbone_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'only missing'):
                attach_cv(*self.fixture(Path(d), overwrite=True))

    def test_modified_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            with (Path(d)/'quality.npz').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                attach_cv(*args)


if __name__ == '__main__':
    unittest.main()
