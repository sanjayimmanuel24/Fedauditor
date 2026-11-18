# Worked example: `brfss-lr-fedavg-r20-subgroup_bias-high-s20` audited by Haiku 5.5 (prompt v5)

**Planted fault:** subgroup_bias (high) in MA, MI, WI from round 1.  
**Score:** precision 1.00, recall 1.00, type correct, onset error 0.0.  
**Fact-check:** 25/25 claims supported.  
**Budget:** ε 20 of 20; 10 tool calls, 4 turns.

## Tool calls

| id | tool | args | ε |
|---|---|---|---|
| c1 | federation_overview |  | 0 |
| c2 | client_updates |  | 3 |
| c3 | client_performance |  | 3 |
| c4 | drift_by_round |  | 3 |
| c5 | feature_summary |  | 3 |
| c6 | subgroup_outcomes | {"attribute": "Sex"} | 2 |
| c7 | subgroup_outcomes | {"attribute": "AgeGroup"} | 2 |
| c8 | subgroup_outcomes | {"attribute": "Income"} | 2 |
| c9 | client_updates | {"client": "MI"} | 1 |
| c10 | client_updates | {"client": "WI"} | 1 |

## Agent summary

Screened all 52 clients with the all-client views of client_updates, client_performance, drift_by_round, feature_summary, and subgroup_outcomes (Sex, AgeGroup, Income), then confirmed per-round update behaviour for MI and WI. Three clients (MA, MI, WI) share a clear, consistent fault: their Sex=1 subgroup has near-zero recorded outcomes against normal predicted risk (O/E 0.0 to 0.10, z_noise about -8.5 to -9), which pulls their overall prevalence well below predicted (positive CITL). Their updates are also anomalous (norm ratios 3 to 5.7, cosine drifting to negative), consistent with persistent local bias. I reported them as subgroup_bias with onset round 1, since the anomaly appears from the first round. I did not report several weaker leads: GU and AL update-norm outliers (small or borderline z), AR's Sex=0 FNR and Income=1 outliers (single statistic, small subgroup), MI's DiffWalk shift (z=3.4, no drift onset), and scattered feature and Income/AgeGroup O/E outliers that look like noise from small subgroups. No missing-artefact sentinels or explanation-instability patterns were seen. The full budget of 20.0 epsilon was spent.

## Findings and the logged values they cite

### MA: subgroup_bias (onset 1, confidence 0.8)

Sex=1 subgroup has an observed outcome rate of 0 against a mean predicted risk of 0.111 (O/E 0.0, z_noise -8.98), i.e. female-subgroup positives appear to be recorded as negative. Overall prevalence (0.053) is roughly half the predicted mean (0.102), consistent with the Sex=1 positives being zeroed out, and CITL is +0.049 (z=+3.7). Local AUC (0.837) is normal, so this is not general label noise. Updates are anomalous (median norm ratio 3.07 vs a median of about 1, mean cos 0.03), which fits a client whose biased local optimum pulls against the federation. The Sex=1 pattern is the same as in MI and WI.

| claim | metric | claimed | logged | cites | row |
|---|---|---|---|---|---|
| client=MA, feature=Sex, group=1.0 | oe_ratio | 0.0 | 0.0 | c6 subgroup_outcomes | outliers |
| client=MA, group=1.0 | z_noise | -8.981 | -8.981 | c6 subgroup_outcomes | outliers |
| client=MA, round=20 | prevalence | 0.05323 | 0.05323 | c3 client_performance | clients |
| client=MA, round=20 | mean_pred | 0.1024 | 0.1024 | c3 client_performance | clients |
| client=MA, round=20 | citl | 0.04914 | 0.04914 | c3 client_performance | clients |
| client=MA | median_norm_ratio | 3.069 | 3.069 | c2 client_updates | clients |
| client=MA | mean_cos | 0.02725 | 0.02725 | c2 client_updates | clients |

### MI: subgroup_bias (onset 1, confidence 0.85)

Sex=1 subgroup has an outcome rate of 0.012 against a mean predicted risk of 0.148 (O/E 0.084, z_noise -8.53), so female-subgroup positives are systematically recorded as negative. Overall prevalence is 0.072 vs a predicted 0.141 (CITL +0.069, z=+5.1), with ECE also elevated (z=+3.8). The client's update norm ratio grows steadily from 1.31 in round 1 to 4.36 in round 20, and its cosine falls from 0.63 to -1.0. This divergence from the federation is consistent with a persistent local bias that the converging global model cannot absorb. Local AUC is lower (0.76) but the Sex=1 zeroing is the dominant signature.

| claim | metric | claimed | logged | cites | row |
|---|---|---|---|---|---|
| client=MI, feature=Sex, group=1.0 | oe_ratio | 0.08406 | 0.08406 | c6 subgroup_outcomes | outliers |
| client=MI, feature=Sex, group=1.0 | outcome_rate | 0.01241 | 0.01241 | c6 subgroup_outcomes | outliers |
| client=MI, group=1.0 | z_noise | -8.534 | -8.534 | c6 subgroup_outcomes | outliers |
| client=MI, round=20 | prevalence | 0.07193 | 0.07193 | c3 client_performance | clients |
| client=MI, round=20 | citl | 0.06928 | 0.06928 | c3 client_performance | clients |
| client=MI | median_norm_ratio | 3.596 | 3.596 | c2 client_updates | clients |
| client=MI, round=1 | norm_ratio | 1.312 | 1.312 | c9 client_updates | per_round |
| client=MI, round=20 | norm_ratio | 4.356 | 4.356 | c9 client_updates | per_round |
| client=MI, round=20 | cos | -1 | -1 | c9 client_updates | per_round |

### WI: subgroup_bias (onset 1, confidence 0.85)

Sex=1 subgroup has an outcome rate of 0.015 against a mean predicted risk of 0.150 (O/E 0.098, z_noise -8.53), so female-subgroup positives are recorded as negative. Overall prevalence is 0.087 vs a predicted 0.155 (CITL +0.068, z=+5.0), with ECE elevated (z=+3.1). The update norm ratio grows from 1.5 in round 1 to 5.7 by round 20, and the cosine turns negative from about round 8 onward. The pattern matches MI and MA, and the local AUC (0.80) is not the main signal.

| claim | metric | claimed | logged | cites | row |
|---|---|---|---|---|---|
| client=WI, feature=Sex, group=1.0 | oe_ratio | 0.09839 | 0.09839 | c6 subgroup_outcomes | outliers |
| client=WI, feature=Sex, group=1.0 | outcome_rate | 0.01475 | 0.01475 | c6 subgroup_outcomes | outliers |
| client=WI, group=1.0 | z_noise | -8.528 | -8.528 | c6 subgroup_outcomes | outliers |
| client=WI, round=20 | prevalence | 0.0865 | 0.0865 | c3 client_performance | clients |
| client=WI, round=20 | citl | 0.06823 | 0.06823 | c3 client_performance | clients |
| client=WI | median_norm_ratio | 4.713 | 4.713 | c2 client_updates | clients |
| client=WI, round=1 | norm_ratio | 1.505 | 1.505 | c10 client_updates | per_round |
| client=WI, round=20 | norm_ratio | 5.703 | 5.703 | c10 client_updates | per_round |
| client=WI, round=20 | cos | -0.7513 | -0.7513 | c10 client_updates | per_round |
