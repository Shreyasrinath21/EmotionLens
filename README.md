# EmotionLens

**Deep Learning-Based Voice Emotion Recognition** is a local-first academic application developed for RV University, School of Computer Science & Engineering. EmotionLens combines a FastAPI service, audio feature extraction, local CNN/CNN–LSTM checkpoints, a separate pretrained audio classifier, and a responsive browser dashboard.

The interface follows the core of the all-features design blueprint; it never fabricates model outputs or evaluation metrics. This working copy has locally trained RAVDESS checkpoints under `backend/artifacts/`. Other checkouts can use the separately labelled pretrained model or train a local model; datasets and weights are not bundled.

## Current capabilities

- WAV upload, browser decoding, drag-and-drop, metadata, playback, and explicit analysis consent.
- Optional microphone recording initiated by the user; audio is encoded and analysed locally.
- Backend file, duration, channel, sample-rate, silence, and numerical validation; mono conversion, resampling to 16 kHz, peak normalisation, and an adaptive frame-energy activity estimate with a no-activity prediction gate. This rough estimate cannot distinguish speech from noise.
- Log-Mel, MFCC, pitch, RMS, spectral centroid, zero-crossing, waveform, and coarse amplitude-envelope descriptors.
- CNN, CNN–LSTM, and CNN–BiLSTM with temporal attention PyTorch models; RAVDESS and CREMA-D filename parsing; speaker-independent splits; early stopping; validation-only rejection threshold selection; held-out metrics; and saved model metadata.
- Separate SUPERB wav2vec2 emotion classifier option with four English IEMOCAP-derived classes (neutral, happy, angry, sad); its pinned model files download on first use and inference remains local. It does not replace or map into the eight-class RAVDESS model.
- Actual checkpoint inference, segment scores and timestamps, uncertainty rejection, acoustic visualisations, emotion-track and confidence-trend charts, automatic timeline smoothing, segment replay, response prompts, and JSON/CSV export.
- Sticky Analyze / Results / Evaluation workspace navigation, with labeled graph axes, units, scales, and legends for acoustic and experiment charts.
- Automatic input-gradient sensitivity maps for trained checkpoints, clearly described as model sensitivity rather than causal explanations.
- Evaluation section displays only saved results from a real training run.
- Each analysis displays the selected filename and that clip's own prediction and score distribution (when a compatible model is available). Held-out accuracy and F1 are labelled as fixed model-benchmark metrics and are not recalculated for each upload.
- Optional consent-based speaker reference comparison of acoustic descriptors; reference audio and features remain in memory only.
- Reproducible held-out robustness runs for clean audio, white-noise SNR levels, and a Wiener-filter baseline.
- Optional Wiener denoising for individual analyses and reference features, off by default and explicitly identified as experimental.
- Fair CNN/CNN–LSTM/CNN–BiLSTM-attention comparison command using the same speaker split, plus model card and actual evaluation endpoints.
- Dataset audit CLI and dashboard report for labelled-file integrity, class counts, speakers, audio properties, and deterministic speaker-split membership.
- No-fine-tuning cross-corpus evaluation CLI for RAVDESS/CREMA-D using their six shared emotion labels, a source-only threshold, and target-corpus metrics; requires a separately obtained, permitted target dataset.
- Audio is processed in memory by the API; the backend does not retain uploaded audio.
- CREMA-D filename emotion codes can be displayed as dataset annotations when the installed model is from a different corpus; these labels are explicitly not model predictions.

No claim of model quality, calibration, real-time performance, denoising benefit, speaker adaptation, cross-corpus generalisation, or multimodal performance is made without corresponding evaluated experiments.

### Optional pretrained model and licence notes

The model selector includes `superb/wav2vec2-base-superb-er` at the pinned Hugging Face revision `441a7599c3b22107314dcbd9166621c5c83f2cc5`. Selecting it triggers a first-use download of approximately 757 MB into `backend/artifacts/pretrained_cache`; it requires an internet connection and sufficient disk space. Audio analysis then runs on this machine. Uploaded audio is not sent to Hugging Face.

The [Hugging Face emotion-model card](https://huggingface.co/superb/wav2vec2-base-superb-er) and [wav2vec2-base card](https://huggingface.co/facebook/wav2vec2-base) mark their respective repositories Apache-2.0. The emotion head is trained for the SUPERB/IEMOCAP task and returns only neutral, happy, angry, and sad. The model card reports 62.58% accuracy on SUPERB session 1; this is an external result, not an EmotionLens measurement or a guarantee for new recordings. EmotionLens does not calibrate or apply an uncertainty-rejection threshold to this model, so its displayed raw scores must not be interpreted as probabilities. The model has not been validated here on RAVDESS, CREMA-D, or everyday audio.

USC's [IEMOCAP release page](https://sail.usc.edu/iemocap/iemocap_release.htm) requires an access request and release-form review for source recordings. The Hugging Face model card does not explain how the model-weight licence relates to all source-data terms. The checkpoint is downloaded for the user's local use and is not redistributed by this project; review the current model and dataset terms for the intended use before relying on it.

## Requirements and setup

Python 3.13 is the tested runtime. Exact direct dependency versions are pinned in `backend/requirements.txt`, including Transformers for the optional pretrained classifier. Install the backend dependencies in a virtual environment:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The app is served by FastAPI; no external frontend build or cloud service is required:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Stop the service with `Ctrl+C`. API documentation is at <http://127.0.0.1:8000/docs>. Microphone permissions work on localhost; audio is not sent to a remote service.

Choose the pretrained model in the model selector to use it. Its first analysis downloads the model files; no separate setup or audio upload is needed. If the model download fails, the API reports the error, and retrying after restoring network access is safe.

## Dataset and training

Obtain the RAVDESS speech audio dataset from its official source and review its current research and redistribution terms before use. This project does not download datasets automatically. Place the unmodified WAV files in a local directory with the standard RAVDESS file names, such as:

```text
datasets/
  Actor_01/
    03-01-03-01-01-01-01.wav
```

Run training from `backend`:

```powershell
python -m app.training --dataset ..\datasets\RAVDESS --architecture cnn_lstm --epochs 20 --batch-size 16 --seed 42
```

Audit the local corpus before training. The measured report is saved to `backend/artifacts/dataset_audit.json` and appears in the Evaluation dashboard:

```powershell
python -m app.training --dataset ..\datasets\RAVDESS --dataset-format ravdess --audit-dataset
```

For a fair architecture experiment, append `--compare`. This trains CNN, CNN–LSTM, and CNN–BiLSTM with temporal attention on the same held-out speaker splits, stores their checkpoints locally, enables the model selector for available checkpoints, and writes measured comparison results. This triples the training work and may take substantially longer. To run the corruption and denoising experiment, append `--robustness` (this adds several evaluations on the held-out test set and takes longer). Record the precise dataset release and citation using `--dataset-version` and `--dataset-source`. The combined command is:

```powershell
python -m app.training --dataset ..\datasets\RAVDESS --architecture cnn_lstm --epochs 20 --batch-size 16 --seed 42 --dataset-version "your dataset release" --dataset-source "official dataset citation" --compare --robustness
```

Individual recording analysis uses the original audio by default. The optional Wiener checkbox applies the same experimental filter to an uploaded recording and its separately consented reference; filtering may remove emotion-relevant cues and is not claimed to improve a prediction.

After training a source-corpus model and obtaining a second permitted dataset, run a separate cross-corpus evaluation without target fine-tuning or threshold tuning:

```powershell
python -m app.cross_corpus --dataset ..\datasets\CREMA-D --dataset-format crema-d
```

The report compares only the six labels shared by RAVDESS and CREMA-D. CREMA-D is not bundled, so no transfer metric is claimed until the actual permitted recordings are supplied.

For the baseline alone, pass `--architecture cnn`. The pipeline reads standard RAVDESS **audio-only speech** filenames and speaker IDs, audits the speaker-disjoint splits, trains with the same mono/resample/normalisation and log-Mel feature path used at inference, selects the rejection threshold on validation data, measures calibration/coverage-risk on held-out test data, and writes outputs under `backend/artifacts/`. Model weights and generated results are ignored by Git. Re-run the API after training to load the checkpoint and experiment reports.

The RAVDESS label mapping is 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, and 08 surprised. Only standard speech WAV filenames with speaker IDs are included. Do not mix datasets with incompatible class definitions without a separate documented mapping and evaluation.

### Train for CREMA-D audio

The model supplied in `backend/artifacts/` was trained on RAVDESS and must not be treated as validated on CREMA-D. CREMA-D data is not bundled or downloaded by this application. Request access through the [official CREMA-D repository](https://github.com/CheyneyComputerScience/CREMA-D#access), follow its access and licence terms, and obtain the actual WAV audio files (a ZIP containing only Git LFS pointers is not sufficient). The filename format is `1001_IEO_ANG_HI.wav`; extract the audio files into a local folder, for example `datasets\CREMA-D\`.

From `backend`, train a CREMA-D-specific six-class model and held-out evaluation:

```powershell
python -m app.training --dataset ..\datasets\CREMA-D --dataset-format crema-d --architecture cnn_lstm --epochs 20 --batch-size 32 --seed 42 --dataset-version "CREMA-D" --dataset-source "CREMA-D official source" --compare --robustness
```

CREMA-D labels are mapped as `NEU` neutral, `HAP` happy, `SAD` sad, `ANG` angry, `FEA` fearful, and `DIS` disgust. The trainer keeps actor IDs disjoint across train, validation, and test splits. After successful training, restart the API so it loads the new CREMA-D checkpoint. The API will then report its dataset and six-class mapping; CREMA-D-named clips will no longer be blocked. Evaluation is still required to determine whether the model is accurate enough for your intended research use.

## Tests

From the `backend` directory, after installing requirements:

```powershell
python -m pytest -q
```

Tests cover audio validation and preprocessing, feature and segment dimensions, both model output shapes, RAVDESS and CREMA-D parsing, API health, extension validation, and cross-corpus safeguards. A CREMA-D model can only be trained and evaluated after you provide the permitted CREMA-D audio locally.

## API

- `GET /api/health` — local service, trained-checkpoint, and pretrained-model status.
- `GET /api/model` — model metadata, mapping, threshold, pretrained-model provenance, available model choices, and readiness.
- `GET /api/evaluation` — saved real evaluation output, or an explicit `not_evaluated` status.
- `GET /api/comparison` and `GET /api/robustness` — actual experiment outputs or explicit `not_evaluated` status.
- `GET /api/cross-corpus` and `GET /api/dataset-audit` — saved cross-corpus evaluation and dataset audit, or explicit `not_evaluated` status.
- `GET /api/model-card` and `GET /api/capabilities` — intended use, limitations, and the enabled feature set.
- `POST /api/features` — consented feature extraction without an emotion prediction, used for optional reference comparison.
- `POST /api/analyze` — multipart WAV upload; returns summary scores, timestamped segment scores, acoustic features, and reliability caveats. Choose `superb_wav2vec2_base_er` in the `architecture` field to use the optional pretrained classifier. For a CREMA-D-named file without a CREMA-D-trained model, it returns acoustic features and the encoded filename label as dataset metadata, not as a model prediction.

Files are capped at 10 MB and 30 seconds. When the pretrained option is selected, the first analysis downloads its pinned model files; the audio itself stays local. If no model is selected and there is no local trained checkpoint, the API returns genuine audio measurements with `status: "features_only"` and no emotion prediction. The application never creates a fake result. Raw uploads are not written to disk or returned in exports.

## Features F01–F20 checklist

| Feature | Status | Notes |
|---|---|---|
| F01 Upload and validation | Implemented | WAV, browser and backend validation, consent, loading/error states; limits documented. |
| F02 Preprocessing | Implemented | Mono, 16 kHz resample, peak normalisation; optional experimental Wiener denoising can be applied consistently to the current recording and reference. |
| F03 Feature extraction | Implemented | Log-Mel, MFCC, pitch and spectral/energy summaries with waveform views. |
| F04 CNN–LSTM classifier | Implemented | CNN, CNN–LSTM, and CNN–BiLSTM-attention are trainable; this working copy has a local RAVDESS CNN–LSTM checkpoint and separately offers a four-class pretrained wav2vec2 model. |
| F05 Emotion timeline | Implemented | 3-second windows, 2-second hop, raw emotion track and score trend plots, automatic client smoothing, replay; coarse energy-based silence/activity estimate included. |
| F06 Uncertainty | Implemented | Threshold selected using validation data. Softmax is not called calibrated. |
| F07 Noise robustness | Implemented | Saved held-out results display a macro-F1 chart and table for clean audio, 20/10/0 dB white noise, and the Wiener-filtered condition. Running new experiments requires `--robustness`. |
| F08 Speaker-independent evaluation | Implemented in training | Actor-disjoint train/validation/test split and overlap audit. |
| F09 Explainability | Implemented | Audio-derived acoustic views and automatic trained-model input-gradient sensitivity map, with interpretability limitations disclosed. |
| F10 Speaker adaptation | Partial | Optional reference feature deltas are calculated in memory after separate consent. No model adaptation is applied, and no accuracy improvement is claimed or evaluated. |
| F11 Response suggestions | Implemented | Cautious response templates are editable for accepted predictions and suppressed when the output is uncertain. |
| F12 Dashboard | Implemented | Upload, results, audio views, plotted timeline, reliability, optional speaker baseline, evaluation, model comparison and export. |
| F13 Model comparison | Implemented | The saved comparison includes measured CNN, CNN–LSTM, and CNN–BiLSTM-attention results on the same 240 held-out RAVDESS clips. |
| F14 Evaluation/reporting | Implemented in core pipeline | Actual held-out metrics, per-class results, confusion matrix, loss history, calibration/coverage-risk, robustness, and browser print-to-PDF; peak memory tracking is not available. |
| F15 Versioning | Implemented | Metadata, seed, split speakers, config, versions, checkpoint ID, metrics. |
| F16 Testing | Implemented | Pytest covers audio validation/activity estimates, features, all model outputs, split leakage, dataset audit, cross-corpus plumbing, thresholds, calibration, noise, and API flows; browser E2E verifies dashboard inference and plots. |
| F17 Local deployment | Implemented | Local FastAPI app, checkpoint-based inference, health/model/evaluation APIs. |
| F18 Privacy and responsible use | Implemented | Local-only UI, in-memory uploads, opt-in recording, explicit consent, research limitations. |
| F19 Multilingual/cross-corpus | Partially implemented | Cross-corpus evaluation tooling and UI support RAVDESS↔CREMA-D shared labels; actual evaluation requires permitted target audio. Multilingual datasets and evaluation are not supported. |
| F20 Multimodal fusion | Planned | Voice-only pipeline remains independent; no video/text modality is enabled. |

## Limitations and end-to-end demonstration

1. Install dependencies and obtain RAVDESS under its terms.
2. Train using the command above; inspect `backend/artifacts/model.json` and `evaluation.json`.
3. Start the API and open the local page.
4. Upload a consented WAV recording, confirm permission, and analyse. Explore the plots, timeline, reliability panel, and CSV/JSON exports.

RAVDESS is acted English speech. Results may not generalise across speakers, languages, accents, demographic groups, or recording contexts. Softmax values are raw model scores, not calibrated probabilities. Do not use this software for high-stakes decisions or as a determination of anyone's internal emotional state.
