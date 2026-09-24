# Scientific audit — 2026-09-24

## Status and scope

**Not scientifically validated.** This is an initial audit of the active application,
not certification of the entire repository. Historical applications under `version/`,
vendored virtual-environment packages, and `original_source.txt` were not fully audited.
Do not deploy historical versions as audited alternatives.

Reviewed: active formula/data definitions, weight and carcass arithmetic, input scoring,
sensitivity ranges, target inversion, batch input, report generation and existing tests.
Runtime UI testing requires dependencies absent from the current Python environment.

## Findings supported by repository evidence

| Finding | Evidence | Action/status |
|---|---|---|
| Invalid measurements propagate into estimates | `hitung_berat_badan` squared negative LD and accepted NaN/infinity | Reject non-positive/non-finite dimensions and booleans. |
| Invalid carcass mass propagates | `hitung_komponen_karkas` lacked a domain check | Reject negative/non-finite mass; preserve zero. |
| Invalid sensitivity margins | `calculate_error_range` accepted negative, >100%, NaN/infinite margins; invalid text silently became 10% | Reject invalid values; document user scenario, not statistical interval. |
| Infinite uploaded price accepted | `clean_price_value` checked only positivity | Infinity now uses existing fallback policy. |
| “Accuracy” is not measured accuracy | `calculate_input_accuracy_score` starts at 100 and subtracts manually specified penalties (18, 14, 12, etc.) | Global warning added. No validation dataset or error metrics supplied. Existing labels remain legacy terminology. |
| Histogram is not population evidence | `create_weight_distribution_chart` evaluates a grid made with `np.linspace` | Global warning identifies simulated sizes, not sampled animals. |
| Formula/data provenance incomplete | `ANIMAL_FORMULAS`, breed/sex factors, size ranges and slaughter percentages lack traceable source-to-coefficient extraction and validation data | All remain unverified, including original breeds. No replacement coefficients invented. |
| NSA Australia returns zero for ordinary example | `max(0, 0.0000627 * 100 * 100 - 3.91)` equals zero | Regression check reproduces defect; coefficient cannot be repaired without source. Do not use this formula for research. |
| Target inversion has hard bounds | `estimate_dimensions_for_target_weight` searches only scales 0.40–2.50 without proving target is bracketed | Unresolved: may return boundary estimate for unreachable target. |
| Exported evidence status incomplete | Existing PDF/CSV and AI prompt paths do not consistently carry scientific caveats | Unresolved: UI warning does not make exported results validated. |
| Default market claims unverified | `LATEST_PRICE_DEFAULTS` names dates/agencies without retrievable price records; breed multipliers are hard-coded | Global warning; verify local records before use. |

## External verification limitations

Requests to these publisher pages returned HTTP 403 during audit:
- https://academic.oup.com/jas/issue/1/1
- https://www.publish.csiro.au/cp/AbouttheJournal

These failures establish neither validity nor fabrication of the stored references.
No citation was promoted to “verified”. Bibliographic strings alone do not establish
that a paper contains the implemented equation, units or correction factors.

## Requirements before academic use

1. Obtain primary texts/DOIs and identify exact equation/table/page, units, species,
   breed, sex, age, sample size and calibration population for every enabled model.
2. Remove or disable unsupported models for research deployment; do not replace
   suspicious coefficients by guessing from familiar formulas.
3. Validate against paired measurements and scale weights on independent animals.
   Report sample selection, MAE/RMSE, signed bias, subgroup performance and domain limits.
4. Estimate prediction intervals from appropriate validation data. User-selected
   percentages and heuristics must not be presented as measured accuracy.
5. Validate carcass definitions and component denominators against actual slaughter
   records; check economics against dated, local price evidence.
6. Propagate model provenance and limitations into every export and AI prompt.

## Reproducible checks

```sh
python3 /Users/macbookpro/Documents/GitHub/bbternak/test_cattle_data.py
python3 /Users/macbookpro/Documents/GitHub/bbternak/test_scientific_audit.py
```

These check software invariants only. Passing tests is not biological validation.
