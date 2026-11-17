import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
from run_submission import ROOT, attach_cv, digest, run
from possession_observation_quality import QualityConfig, select_observations


class CVAttachmentTests(unittest.TestCase):
    def test_submission_cannot_silently_fall_back_to_tracking_only(self):
        with self.assertRaisesRegex(ValueError, 'requires --cv-manifest'):
            run('unused', 'unused')
        with self.assertRaisesRegex(ValueError, 'Choose CV-supported'):
            run('unused', 'unused', cv_manifest='unused', backbone_only=True)

    def test_unknown_height_is_preserved_through_long_gap_recovery(self):
        b=np.ones((100,2));b[10:90]=np.nan
        cv=np.ones((100,2));confidence=np.full(100,.9);sigma=np.full(100,.2);airborne=np.full(100,np.nan)
        selected,q=select_observations(b,cv,confidence,sigma,airborne,[[1,0,99]],QualityConfig(unknown_airborne_policy='allow_uncertain'))
        self.assertEqual(q['cv_used'].sum(),80)
        self.assertTrue(q['ground_projection_uncertain'].all())
        self.assertFalse(q['airborne_known'].any())
        airborne[20]=1.;confidence[21]=.1;sigma[22]=3.
        selected,q=select_observations(b,cv,confidence,sigma,airborne,[[1,0,99]],QualityConfig(unknown_airborne_policy='allow_uncertain'))
        self.assertTrue(np.isnan(selected[20:23]).all())

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
                 image_observed=[True]*3, ground_projection_uncertain=[True]*3)
        manifest = dict(training_eligible=False, camera='tactical', fps=25, frame_start=1, n_frames=3,
                        quality_policy=asdict(QualityConfig(unknown_airborne_policy='allow_uncertain')),
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
            self.assertTrue(masks['cv_projection_uncertain'][2])
            self.assertTrue(np.isnan(args[1][2]).all())

    def test_existing_backbone_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'only missing'):
                attach_cv(*self.fixture(Path(d), overwrite=True))

    def test_old_anchor_policy_cannot_be_mistaken_for_updated_submission(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d));path=args[-1];m=json.loads(path.read_text())
            m['quality_policy']['unknown_airborne_policy']='require_anchors'
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'allow_uncertain'):
                attach_cv(*args)

    def test_modified_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            with (Path(d)/'quality.npz').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                attach_cv(*args)


if __name__ == '__main__':
    unittest.main()
