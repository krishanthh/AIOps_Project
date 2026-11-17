# Test Execution Report

I generated this report from executed checks. Synthetic unit-test fixtures are separate from the real MRI dataset evaluation.

## Automated application tests

Tests: 29 · Passed: 29 · Failures: 0 · Errors: 0 · Skipped: 0

| Test case | Actual outcome |
|---|---|
| tests.test_application.test_health_does_not_require_model | passed |
| tests.test_application.test_missing_file | passed |
| tests.test_application.test_invalid_image[] | passed |
| tests.test_application.test_invalid_image[not an image] | passed |
| tests.test_application.test_invalid_image[GIF89a] | passed |
| tests.test_application.test_oversized_image | passed |
| tests.test_application.test_unsupported_decoded_format | passed |
| tests.test_application.test_small_image_rejected | passed |
| tests.test_application.test_serving_failure_modes[ok-200] | passed |
| tests.test_application.test_serving_failure_modes[down-503] | passed |
| tests.test_application.test_serving_failure_modes[timeout-504] | passed |
| tests.test_application.test_serving_failure_modes[bad-503] | passed |
| tests.test_application.test_readiness_tracks_serving[ok-200] | passed |
| tests.test_application.test_readiness_tracks_serving[down-503] | passed |
| tests.test_application.test_pipeline_disabled | passed |
| tests.test_application.test_pipeline_rejects_parallel_run | passed |
| tests.test_application.test_empty_operational_states | passed |
| tests.test_application.test_pipeline_log_missing | passed |
| tests.test_application.test_metrics_available_during_model_outage | passed |
| tests.test_application.test_brightness_warning_requires_samples | passed |
| tests.test_application.test_atomic_artifact_roundtrip | passed |
| tests.test_application.test_deterministic_split_and_duplicate_removal | passed |
| tests.test_application.test_archive_traversal_rejected | passed |
| tests.test_application.test_pipeline_failure_is_recorded | passed |
| tests.test_application.test_pyfunc_reload_explanation_and_class_contract | passed |
| tests.test_application.test_mlflow_packaging_rebuilds_runtime_lock | passed |
| tests.test_application.test_preparation_failure_preserves_completed_output | passed |
| tests.test_application.test_training_rejects_incomplete_prepared_data | passed |
| tests.test_application.test_acquisition_rejects_wrong_archive_checksum | passed |

## Acceptance checks against the test plan

| Test-plan area | Acceptance criterion | Actual result | Status |
|---|---|---|---|
| Pipeline / data | Complete splits, successful stages, measured Ray throughput | 5/5 checks passed; `reports/pipeline-acceptance.json` | PASS |
| Software packaging | Compose health, independent HTTP serving, restart and rollback | 15/15 checks passed; `reports/deployment.json` | PASS |
| Real-service integration | Real MRI probabilities/Grad-CAM, signature rejection, latency | 11/11 checks passed; `reports/integration.json` | PASS |
| Monitoring | Healthy scrape targets, populated Grafana panels, evaluated alert rules | 11/11 checks passed; `reports/monitoring.json` | PASS |
| MLflow Projects | Declared entrypoint executes and dependencies match the pinned environment | 8/8 checks passed; `reports/reproducibility.json` | PASS |
| Browser UX | Navigation, real upload/heatmap, narrow viewport, no console errors | Final browser acceptance not recorded | NOT EXECUTED |

## Measured performance

Ray prepared 6,878 images with 2 workers in 30.63 seconds (224.59 images/second).
The recorded full seven-stage run took 209.26 seconds. Later unchanged DVC runs reused cached artifacts; their shorter durations are not presented as fresh model training.
Measured Nginx → FastAPI → MLflow prediction without Grad-CAM: 0.240 seconds (single-user target ≤10 seconds).
Measured Nginx → FastAPI → MLflow prediction with Grad-CAM: 0.131 seconds (single-user target ≤10 seconds).

## Model evaluation

Held-out images: 1579. Accuracy: 0.8391. Macro-F1: 0.8311.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | 0.8991 | 0.7441 | 0.8143 |
| meningioma | 0.8601 | 0.6212 | 0.7214 |
| notumor | 0.7960 | 0.9950 | 0.8844 |
| pituitary | 0.8319 | 0.9900 | 0.9041 |

Selected model run: `6d3f391a229a464fb61d805da5fa66d3`. Training Git commit: `cc521f8af8167a7e903baebad7aa1449f26fa570`. Validation macro-F1: 0.9276; target ≥0.85. Experimental flag: `False`.

## Interpretation and limitations

Image-level evaluation does not establish patient-level independence or clinical validity. Perceptual-hash similarities and missing patient identifiers remain documented data limitations. A Grad-CAM overlay is not a tumour segmentation. Brightness drift is a limited distribution check. The viva and submission video are separate deliverables, not automated test results.
