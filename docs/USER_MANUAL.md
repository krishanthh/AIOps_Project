# User Manual

I designed BrainScope as a local educational MRI classification application.

1. Open `http://localhost:3100`. A green “Model connected” badge means analysis is available. If the badge says unavailable, wait for startup or ask the operator to check serving logs.
2. On **Analyse MRI**, click the upload box or drag a JPEG/PNG image into it. The maximum file size is 10 MB. Check that the displayed preview is the image you intended to upload.
3. Leave **Include Grad-CAM explanation** enabled to request an overlay, or disable it for a faster prediction. Click **Analyse MRI** and wait for completion.
4. Read the predicted category and the four model scores. “No tumour” is a model category, not a medical conclusion. An experimental badge means the configured validation target was missed.
5. The colour overlay shows regions influencing the predicted class. It does not outline a tumour. The page also identifies the model version and measured request duration.
6. Choose another file by clicking the preview. Errors explain how to retry; no result is shown for a failed request.
7. Open **ML pipeline** to inspect stages, outcomes, duration, measured throughput, and available logs. Run controls may be disabled in the serving deployment; the operator runs the local pipeline command.
8. Open **Model evaluation** for actual experiment comparison and held-out test results. Empty panels mean training/evaluation artifacts are not yet available.
9. Open **Observability** to reach Grafana, Prometheus, and MLflow. A brightness warning describes input distribution changes, not measured model accuracy.

The prediction API processes uploaded images in memory and does not save them. The training dataset and generated example artifacts remain on disk.