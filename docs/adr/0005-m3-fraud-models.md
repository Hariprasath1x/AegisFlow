# ADR 0005 - Milestone 3 fraud modelling decisions

Status: accepted (M3)

- **D1 Time splits only, with an embargo.** Train weeks 1-6, test 7-8, and drop the
  24 hours straddling the cut. Fraud is adversarial: a random split lets the model
  see next week's ring. The embargo exists because velocity windows span the
  boundary, so adjacent rows leak through the feature store even when the labels do
  not.
- **D2 Label delay is modelled, not ignored.** Chargebacks land weeks after the
  transaction. Rows whose label had not arrived at the as-of date are EXCLUDED from
  training, never silently treated as legitimate.
- **D3 No feature is invented in M3.** Every column comes from M2's
  `features/definitions.py`, so anything the model learns can be computed in under
  100 ms by M4's scoring service. A missing feature is a change to definitions.py.
- **D4 Frequency encoding, fitted on train only.** Target encoding puts the label
  inside the encoder; it is the most common silent leak in fraud notebooks.
- **D5 scale_pos_weight, not SMOTE.** Synthetic minority samples invent customers
  who never existed, and the invented region is exactly where the model then errs.
- **D6 Calibration on a held-out TIME slice.** The last 15% of training time, never
  random rows, and the calibrator is logged WITH the model - shipping them
  separately is how a service ends up serving raw margins.
- **D7 The rules baseline ships as code.** Objective 5.2 claims a 30% lift. Without
  the baseline in the repo, fitted on the same data and compared at the same FPR,
  it is a slogan. M4's deterministic rule layer reuses these same pure functions.
- **D8 Aliases, not stages; and no self-promotion.** `champion` serves, `challenger`
  shadow-scores. A training run can only set `challenger`. Promotion re-checks the
  rule at promotion time, so a stale terminal cannot promote a model that stopped
  winning.
- **D9 A promotion needs a margin.** A relative lift below 2% on recall@1%FPR is
  inside the noise and is not worth a production swap.
- **D10 Accuracy is never reported.** Recall at 1% FPR, PR-AUC, cost curves and
  cohort slices. At a 3.5% fraud rate, approving everything scores 96.5%.
- **D11 The fairness screen is a finding, not a gate.** An unexplained cohort gap
  goes in the model card and in front of a human. Automating a fix here would be
  worse than naming the problem.
- **D12 Typologies are evaluated separately.** The supervised model is trained on
  card fraud; smurfing and layering are measured on the generator's labelled
  episodes, and an episode counts only when 30% of its transactions were flagged.
