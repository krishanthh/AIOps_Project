"""Generate the submission test report from real unit and acceptance-check artifacts."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path


def load(path, fallback=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else fallback


root = ET.parse("reports/junit.xml").getroot()
suites = list(root.iter("testsuite"))
counts = {
    key: sum(int(s.get(key, 0)) for s in suites)
    for key in ["tests", "failures", "errors", "skipped"]
}
passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
# This stable DVC output links the code/test gate to training without timestamp churn.
Path("reports/code-checks.json").write_text(json.dumps({**counts, "passed": passed}, indent=2))
lines = [
    "# Test Execution Report",
    "",
    "I generated this report from executed checks. "
    "Synthetic unit-test fixtures are separate from the real MRI dataset evaluation.",
    "",
    "## Automated application tests",
    "",
    f"Tests: {counts['tests']} · Passed: {passed} · Failures: {counts['failures']} · "
    f"Errors: {counts['errors']} · Skipped: {counts['skipped']}",
    "",
    "| Test case | Actual outcome |",
    "|---|---|",
]
for case in root.iter("testcase"):
    outcome = "passed"
    for tag in ["failure", "error", "skipped"]:
        if case.find(tag) is not None:
            outcome = tag
    lines.append(f"| {case.get('classname')}.{case.get('name')} | {outcome} |")
lines += [
    "",
    "## Acceptance checks against the test plan",
    "",
    "| Test-plan area | Acceptance criterion | Actual result | Status |",
    "|---|---|---|---|",
]
areas = [
    (
        "Pipeline / data",
        "Complete splits, successful stages, measured Ray throughput",
        "reports/pipeline-acceptance.json",
    ),
    (
        "Software packaging",
        "Compose health, independent HTTP serving, restart and rollback",
        "reports/deployment.json",
    ),
    (
        "Real-service integration",
        "Real MRI probabilities/Grad-CAM, signature rejection, latency",
        "reports/integration.json",
    ),
    (
        "Monitoring",
        "Healthy scrape targets, populated Grafana panels, evaluated alert rules",
        "reports/monitoring.json",
    ),
    (
        "MLflow Projects",
        "Declared entrypoint executes and dependencies match the pinned environment",
        "reports/reproducibility.json",
    ),
]
for area, criterion, filename in areas:
    evidence = load(filename)
    if evidence and "cases" in evidence:
        cases = evidence["cases"]
        failures = [c["case"] for c in cases if not c.get("passed")]
        outcome = f"{len(cases) - len(failures)}/{len(cases)} checks passed; `{filename}`"
        lines.append(f"| {area} | {criterion} | {outcome} | {'PASS' if not failures else 'FAIL'} |")
    else:
        lines.append(f"| {area} | {criterion} | No completed execution artifact | NOT EXECUTED |")
browser = load("../project-support/verification-evidence/browser.json")
if browser and browser.get("passed") and browser.get("real_prediction"):
    lines.append(
        "| Browser UX | Navigation, real upload/heatmap, narrow viewport, no console errors | "
        "Firefox; 390 px viewport; real prediction rendered | PASS |"
    )
else:
    lines.append(
        "| Browser UX | Navigation, real upload/heatmap, narrow viewport, no console errors | "
        "Final browser acceptance not recorded | NOT EXECUTED |"
    )
preparation = load("reports/preparation.json")
full_pipeline = load("reports/pipeline-acceptance.json", {}).get("completed_run")
integration = load("reports/integration.json", {}).get("cases", [])
if preparation:
    lines += [
        "",
        "## Measured performance",
        "",
        f"Ray prepared {preparation['processed']:,} images with "
        f"{preparation['workers']} workers in {preparation['seconds']:.2f} seconds "
        f"({preparation['images_per_second']:.2f} images/second).",
    ]
if full_pipeline:
    duration = full_pipeline["finished_at"] - full_pipeline["started_at"]
    lines += [
        f"The recorded full seven-stage run took {duration:.2f} seconds. "
        "Later unchanged DVC runs reused cached artifacts; their shorter durations "
        "are not presented as fresh model training."
    ]
for case in integration:
    if case["case"].startswith("compose-prediction-explain-"):
        explanation = "with" if case["case"].endswith("True") else "without"
        lines += [
            f"Measured Nginx → FastAPI → MLflow prediction {explanation} "
            f"Grad-CAM: {case['seconds']:.3f} seconds (single-user target ≤10 seconds)."
        ]
report = load("reports/evaluation.json")
if report:
    lines += [
        "",
        "## Model evaluation",
        "",
        f"Held-out images: {report['samples']}. Accuracy: {report['accuracy']:.4f}. "
        f"Macro-F1: {report['macro avg']['f1-score']:.4f}.",
        "",
        "| Class | Precision | Recall | F1 |",
        "|---|---|---|---|",
    ]
    for label in ["glioma", "meningioma", "notumor", "pituitary"]:
        metrics = report[label]
        lines.append(
            f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1-score']:.4f} |"
        )
manifest = load("artifacts/manifest.json")
if manifest:
    lines += [
        "",
        f"Selected model run: `{manifest['run_id']}`. Training Git commit: "
        f"`{manifest['git_commit']}`. Validation macro-F1: "
        f"{manifest['validation_macro_f1']:.4f}; target ≥0.85. "
        f"Experimental flag: `{manifest['experimental']}`.",
    ]
lines += [
    "",
    "## Interpretation and limitations",
    "",
    "Image-level evaluation does not establish patient-level independence or clinical validity. "
    "Perceptual-hash similarities and missing patient identifiers remain documented data limitations. "
    "A Grad-CAM overlay is not a tumour segmentation. Brightness drift is a limited distribution "
    "check. The viva and submission video are separate deliverables, not automated test results.",
]
Path("reports/TEST_REPORT.md").write_text("\n".join(lines) + "\n")
print(f"Wrote reports/TEST_REPORT.md: {passed}/{counts['tests']} unit tests passed")
