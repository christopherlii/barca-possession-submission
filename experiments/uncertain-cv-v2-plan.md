# Unknown-height CV recovery v2: locked development plan

Change only unknown-height eligibility. `allow_uncertain` admits existing accepted
CV observations that pass the unchanged confidence .25 and localization sigma
2 m checks. Unknown height remains explicitly unknown; it is not converted to
grounded. Known airborne risk above .25 still rejects. Existing rejection and
mask provenance checks remain upstream. Available baseline positions remain
unchanged. The old anchor rule remains available as `require_anchors`.

Use the frozen original cycle018 TCN/cycle005 tree and decoder. No model training.
The prior controlled experiment favored confidence-only over the anchor policy.
The new hypothesis is that the anchor veto prevents useful long-gap recovery.

Before applying the new policy to supplied-match inputs, test J03WN1 development
with three fixed conditions: accurate recovery across 10-second gaps; simulated
ground-projection/noisy recovery across 10-second and 30-second gaps. Hide 40%
of baseline observations. Noise: .15 m regular coordinate noise; 12% five-frame
bursts with 8 m noise, confidence .1 and localization sigma 8; other points have
confidence .9 and sigma .2. Camera: pitch-center X, Y=-35 m, Z=25 m. Native
provider height creates projection errors but is withheld from model/policy.
Observation RNG seeds: 4198 gaps, 4295 noise, 4296 noise placement. These are
development observation draws, not independent matches or fresh held-out scores.

Compare backbone, v1 anchors, v2 uncertain, naive filling and the four official
baselines on v1/v2 inputs. Gate: candidate spell F1 cannot regress in any condition;
macro/accuracy cannot drop by more than .01; mean spell gain on the two noisy
conditions must exceed .01; accurate recovery must equal naive recovery. No
threshold changes after inspection. Report failures and preserve the incumbent
if the gate fails. Scores rely on simulated confidence/error relationships; they
do not calibrate actual detector confidence or establish supplied-match accuracy.

If development passes, lock the same policy for fresh synthetic held-out seeds
450201/450202 under all five existing scenarios, 12,000 frames each, with the same
10-second observation-gap stress. Check/consume the ledger once. Compare v1/v2;
gate mean spell gain >= .01, no scenario mean regression > .02, macro/accuracy
mean regression no more than .01. Never tune on these held-out results. Preserve
reserved final-audit data and all previous checkpoints and submission files.
