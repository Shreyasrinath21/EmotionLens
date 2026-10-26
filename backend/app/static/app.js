const fileInput = document.querySelector("#audio-file");
const dropZone = document.querySelector("#drop-zone");
const selectedAudio = document.querySelector("#selected-audio");
const player = document.querySelector("#audio-player");
const analyseButton = document.querySelector("#analyse-button");
const recordButton = document.querySelector("#record-button");
const denoiseCheckbox = document.querySelector("#denoise-audio");
let selectedFile = null;
let currentResult = null;
let audioUrl = null;
let activeRecorder = null;
let recordingTimer = null;
let referenceFile = null;
let referenceFeatures = null;
let activeWorkspacePage = "analyze";

const workspacePages = {
  analyze: document.querySelector("#nav-analyze"),
  results: document.querySelector("#nav-results"),
  evaluation: document.querySelector("#nav-evaluation"),
};

function setWorkspacePage(page) {
  activeWorkspacePage = page;
  for (const [name, button] of Object.entries(workspacePages)) {
    button.classList.toggle("active", name === page);
    if (name === page) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  document.querySelector("#results").hidden = page === "analyze";
  if (page === "analyze") {
    document.querySelector("#input-card").scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
  if (page === "evaluation") {
    document.querySelector('[data-tab="evaluation"]').click();
  } else {
    document.querySelector('[data-tab="overview"]').click();
  }
  document.querySelector("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

workspacePages.analyze.addEventListener("click", () => setWorkspacePage("analyze"));
workspacePages.results.addEventListener("click", () => {
  if (currentResult) setWorkspacePage("results");
});
workspacePages.evaluation.addEventListener("click", () => setWorkspacePage("evaluation"));

function setMessage(message, error = false) {
  const target = document.querySelector("#input-message");
  target.textContent = message;
  target.style.color = error ? "#b13648" : "#3e7c67";
}

function formatDuration(seconds) {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function architectureLabel(architecture) {
  return ({
    cnn: "CNN baseline",
    cnn_lstm: "CNN–LSTM",
    cnn_bilstm_attention: "CNN–BiLSTM + attention",
    superb_wav2vec2_base_er: "SUPERB wav2vec2 · 4 classes",
  })[architecture] || architecture || "unknown";
}

function readWavMetadata(buffer) {
  const view = new DataView(buffer);
  if (view.byteLength < 12) return null;
  const readTag = (offset) => String.fromCharCode(
    view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3),
  );
  if (readTag(0) !== "RIFF" || readTag(8) !== "WAVE") return null;
  for (let offset = 12; offset + 8 <= view.byteLength;) {
    const chunkId = readTag(offset);
    const chunkSize = view.getUint32(offset + 4, true);
    if (chunkId === "fmt " && chunkSize >= 16 && offset + 24 <= view.byteLength) {
      return {
        channels: view.getUint16(offset + 10, true),
        sampleRate: view.getUint32(offset + 12, true),
      };
    }
    offset += 8 + chunkSize + (chunkSize % 2);
  }
  return null;
}

async function inspectFile(file) {
  setMessage("");
  if (!file || !/\.wav$/i.test(file.name)) {
    setMessage("Choose a WAV audio file.", true);
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    setMessage("Audio files must be 10 MB or smaller.", true);
    return;
  }
  try {
    const contents = await file.arrayBuffer();
    const wavMetadata = readWavMetadata(contents);
    const context = new AudioContext();
    let decoded;
    try {
      decoded = await context.decodeAudioData(contents);
    } finally {
      await context.close();
    }
    if (decoded.duration > 30 || decoded.duration < 0.05) {
      setMessage("Audio must be between 0.05 and 30 seconds long.", true);
      return;
    }
    selectedFile = file;
    currentResult = null;
    document.querySelector("#results").hidden = true;
    workspacePages.results.disabled = true;
    setWorkspacePage("analyze");
    compareReferenceButton.disabled = true;
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(file);
    player.src = audioUrl;
    document.querySelector("#file-name").textContent = file.name;
    document.querySelector("#file-meta").textContent =
      `${formatDuration(decoded.duration)} · ${(wavMetadata?.sampleRate || decoded.sampleRate).toLocaleString()} Hz · ${(wavMetadata?.channels || decoded.numberOfChannels)} channel(s) · ${(file.size / 1024).toFixed(0)} KB`;
    selectedAudio.hidden = false;
    analyseButton.disabled = !document.querySelector("#consent").checked;
    setMessage("WAV decoded successfully. Confirm permission before analysis.");
  } catch {
    setMessage("The audio file could not be decoded. Try a standard PCM WAV file.", true);
  }
}

fileInput.addEventListener("change", () => inspectFile(fileInput.files[0]));
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  inspectFile(event.dataTransfer.files[0]);
});
document.querySelector("#consent").addEventListener("change", (event) => {
analyseButton.disabled = !selectedFile || !event.target.checked;
});
document.querySelector("#remove-audio").addEventListener("click", () => {
  selectedFile = null;
  currentResult = null;
  fileInput.value = "";
  player.pause();
  player.removeAttribute("src");
  selectedAudio.hidden = true;
  document.querySelector("#results").hidden = true;
  workspacePages.results.disabled = true;
  setWorkspacePage("analyze");
  compareReferenceButton.disabled = true;
  analyseButton.disabled = true;
  setMessage("");
});

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
}

function encodeWav(buffer) {
  const channels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const frameCount = buffer.length;
  const output = new ArrayBuffer(44 + frameCount * channels * 2);
  const view = new DataView(output);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + frameCount * channels * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, frameCount * channels * 2, true);
  const channelData = Array.from({ length: channels }, (_, index) => buffer.getChannelData(index));
  let offset = 44;
  for (let frame = 0; frame < frameCount; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const sample = Math.max(-1, Math.min(1, channelData[channel][frame]));
      view.setInt16(offset, sample < 0 ? sample * 32768 : sample * 32767, true);
      offset += 2;
    }
  }
  return new Blob([output], { type: "audio/wav" });
}

recordButton.addEventListener("click", async () => {
  if (activeRecorder) {
    activeRecorder.stop();
    recordButton.textContent = "Record audio";
    activeRecorder = null;
    return;
  }
  if (!window.isSecureContext && location.hostname !== "localhost") {
    setMessage("Microphone recording requires a secure context such as localhost.", true);
    return;
  }
  if (!window.confirm("Recording is optional. Confirm that everyone recorded has given permission.")) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    const recorder = new MediaRecorder(stream);
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    recorder.addEventListener("stop", async () => {
      window.clearTimeout(recordingTimer);
      recordingTimer = null;
      stream.getTracks().forEach((track) => track.stop());
      recordButton.textContent = "Record audio";
      try {
        const context = new AudioContext();
        let audio;
        try {
          audio = await context.decodeAudioData(await new Blob(chunks).arrayBuffer());
        } finally {
          await context.close();
        }
        const wav = encodeWav(audio);
        await inspectFile(new File([wav], "microphone-recording.wav", { type: "audio/wav" }));
      } catch (error) {
        setMessage(`Recording could not be prepared: ${error.message}`, true);
      }
    });
    recorder.addEventListener("error", (event) => {
      stream.getTracks().forEach((track) => track.stop());
      window.clearTimeout(recordingTimer);
      recordingTimer = null;
      activeRecorder = null;
      recordButton.textContent = "Record audio";
      setMessage(`Recording failed: ${event.error ? event.error.message : "an unexpected error occurred"}`, true);
    });
    activeRecorder = recorder;
    recorder.start();
    recordButton.textContent = "Stop recording";
    setMessage("Recording locally. Select “Stop recording” when finished.");
    recordingTimer = window.setTimeout(() => {
      if (activeRecorder === recorder && recorder.state === "recording") {
        setMessage("The 30-second recording limit was reached. Preparing the audio.");
        recorder.stop();
        activeRecorder = null;
      }
    }, 30_000);
  } catch (error) {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    setMessage(`Microphone unavailable: ${error.message}`, true);
  }
});

analyseButton.addEventListener("click", async () => {
  if (!selectedFile || !document.querySelector("#consent").checked) return;
  analyseButton.disabled = true;
  analyseButton.textContent = "Analysing…";
  const architecture = document.querySelector("#model-select").value;
  setMessage(architecture === "superb_wav2vec2_base_er"
    ? "Processing audio on the app server. First use downloads the pretrained model files (~757 MB); audio is processed in memory and not retained by the app."
    : "Processing audio locally. This may take a moment.");
  try {
    const form = new FormData();
    form.append("file", selectedFile);
    form.append("consent", "true");
    form.append("include_attribution", "true");
    form.append("denoise", String(denoiseCheckbox.checked));
    if (architecture) form.append("architecture", architecture);
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Analysis failed (${response.status}).`);
    currentResult = payload;
    renderResult(payload);
    document.querySelector("#results").hidden = false;
      workspacePages.results.disabled = false;
      setWorkspacePage("results");
      setMessage(payload.filename_annotation
      ? `CREMA-D dataset label: ${payload.filename_annotation.label}. This is not a model prediction.`
      : payload.analysis_warning || (payload.status === "features_only"
        ? "Audio features were analysed. No emotion prediction is available."
        : "Audio analysis and emotion prediction completed successfully."));
    document.querySelector("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    analyseButton.textContent = "Analyse audio →";
    analyseButton.disabled = !selectedFile || !document.querySelector("#consent").checked;
  }
});

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((item) => {
    const active = item === tab;
    item.classList.toggle("active", active);
    item.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.hidden = panel.id !== `tab-${tab.dataset.tab}`;
  });
  if (activeWorkspacePage !== "analyze") {
    const page = tab.dataset.tab === "evaluation" ? "evaluation" : "results";
    for (const [name, button] of Object.entries(workspacePages)) {
      button.classList.toggle("active", name === page);
      if (name === page) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    }
    activeWorkspacePage = page;
  }
  if (currentResult) drawVisualizations(currentResult);
  if (currentResult && tab.dataset.tab === "timeline") drawTimelineCharts(currentResult);
}));
document.querySelector("#print-report").addEventListener("click", () => window.print());

const referenceInput = document.querySelector("#reference-file");
const referenceConsent = document.querySelector("#reference-consent");
const compareReferenceButton = document.querySelector("#compare-reference");
const resetReferenceButton = document.querySelector("#reset-reference");
const referenceMessage = document.querySelector("#reference-message");

referenceInput.addEventListener("change", () => {
  referenceFile = referenceInput.files[0] || null;
  referenceFeatures = null;
  document.querySelector("#reference-results").hidden = true;
  referenceMessage.textContent = referenceFile
    ? `${referenceFile.name} selected. Give separate consent, then compare.`
    : "";
  referenceMessage.style.color = "";
  compareReferenceButton.disabled = !referenceFile || !referenceConsent.checked || !currentResult;
  resetReferenceButton.disabled = !referenceFile && !referenceFeatures;
});

referenceConsent.addEventListener("change", () => {
  compareReferenceButton.disabled =
    !referenceFile || !referenceConsent.checked || !currentResult || currentResult.status !== "features_only" && !currentResult.features;
});

document.querySelector("#compare-reference").addEventListener("click", async () => {
  if (!referenceFile || !referenceConsent.checked || !currentResult) return;
  compareReferenceButton.disabled = true;
  referenceMessage.textContent = "Measuring reference audio locally…";
  referenceMessage.style.color = "";
  try {
    const form = new FormData();
    form.append("file", referenceFile);
    form.append("consent", "true");
    form.append("denoise", String(
      currentResult.audio.preprocessing.includes("experimental Wiener denoising"),
    ));
    const response = await fetch("/api/features", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `Reference analysis failed (${response.status}).`);
    referenceFeatures = result.features;
    renderReferenceComparison();
    referenceMessage.textContent = "Reference measured in memory only. It will be cleared on reset or when you close this page.";
    referenceMessage.style.color = "#3e7c67";
    resetReferenceButton.disabled = false;
  } catch (error) {
    referenceMessage.textContent = error.message;
    referenceMessage.style.color = "#b13648";
  } finally {
    compareReferenceButton.disabled = !referenceFile || !referenceConsent.checked || !currentResult;
  }
});

document.querySelector("#reset-reference").addEventListener("click", () => {
  referenceFile = null;
  referenceFeatures = null;
  referenceInput.value = "";
  referenceConsent.checked = false;
  document.querySelector("#reference-results").hidden = true;
  document.querySelector("#reference-deltas").replaceChildren();
  referenceMessage.textContent = "Reference audio and derived measurements cleared.";
  referenceMessage.style.color = "#3e7c67";
  compareReferenceButton.disabled = true;
  resetReferenceButton.disabled = true;
});

function renderReferenceComparison() {
  if (!currentResult || !referenceFeatures) return;
  const current = currentResult.features.acoustic;
  const reference = referenceFeatures.acoustic;
  const descriptors = [
    ["Median pitch", "pitch_median_hz", "Hz"],
    ["Mean RMS energy", "rms_mean", ""],
    ["Spectral centroid", "spectral_centroid_mean_hz", "Hz"],
    ["Zero-crossing rate", "zero_crossing_rate_mean", ""],
    ["Envelope peaks / second", "speech_rate_proxy_peaks_per_second", "/sec"],
  ];
  const cards = descriptors.map(([label, key, unit]) => {
    const card = document.createElement("div");
    card.className = "feature-chip";
    const title = document.createElement("span");
    title.textContent = label;
    const value = document.createElement("strong");
    const currentValue = current[key];
    const referenceValue = reference[key];
    if (currentValue == null || referenceValue == null) {
      value.textContent = "Unavailable";
    } else {
      const difference = currentValue - referenceValue;
      const percent = referenceValue === 0 ? null : difference / Math.abs(referenceValue) * 100;
      value.textContent = `${difference >= 0 ? "+" : ""}${difference.toFixed(2)} ${unit}`;
      const detail = document.createElement("span");
      detail.textContent = percent == null
        ? `Current ${currentValue.toFixed(2)} · reference ${referenceValue.toFixed(2)}`
        : `${percent >= 0 ? "+" : ""}${percent.toFixed(1)}% vs reference`;
      card.append(title, value, detail);
      return card;
    }
    card.append(title, value);
    return card;
  });
  document.querySelector("#reference-deltas").replaceChildren(...cards);
  document.querySelector("#reference-results").hidden = false;
}

function renderResult(result) {
  compareReferenceButton.disabled = !referenceFile || !referenceConsent.checked;
  if (referenceFeatures) renderReferenceComparison();
  const featuresOnly = result.status === "features_only";
  const recordingName = selectedFile?.name || "Selected audio";
  document.querySelector("#recording-result-name").textContent = recordingName;
  document.querySelector("#recording-result-note").textContent = result.filename_annotation
    ? featuresOnly
      ? `CREMA-D dataset label: ${result.filename_annotation.label}. This comes from the filename, not a model prediction. ${result.reliability.message}`
      : `CREMA-D filename label: ${result.filename_annotation.label}. This is dataset metadata; scores below are the model's estimate for this recording.`
    : featuresOnly
      ? "This recording has individual acoustic measurements only. No supported emotion-model score was produced."
      : `Individual model estimates and class scores for this recording, using ${architectureLabel(result.reliability.architecture)} (${result.reliability.model_version}). ${result.reliability.dataset === "IEMOCAP via SUPERB" ? "This English four-class model is not validated on RAVDESS or everyday recordings; scores are raw and uncalibrated. " : ""}${result.audio.preprocessing.includes("experimental Wiener denoising") ? "Experimental Wiener denoising was applied; it may remove useful cues. " : ""}These are not the fixed held-out evaluation metrics.`;
  const predictionUnavailable = document.querySelector("#prediction-unavailable");
  predictionUnavailable.hidden = !featuresOnly;
  predictionUnavailable.querySelector("strong").textContent = result.filename_annotation
    ? `CREMA-D dataset label: ${result.filename_annotation.label[0].toUpperCase()}${result.filename_annotation.label.slice(1)}`
    : result.source_corpus_hint
    ? "Emotion prediction is unavailable for this recording."
    : "Audio features are ready; emotion prediction is unavailable.";
  document.querySelector("#show-model-setup").hidden = Boolean(result.source_corpus_hint);
  if (result.filename_annotation) {
    predictionUnavailable.querySelector("p").textContent =
      "Read from the CREMA-D filename; this is a dataset annotation, not an EmotionLens prediction.";
  }
  document.querySelector(".metrics-grid").hidden = featuresOnly;
  document.querySelector("#tab-overview > .results-card").hidden = featuresOnly;
  document.querySelector('[data-tab="timeline"]').hidden = featuresOnly;
  document.querySelector("#export-csv").hidden = featuresOnly;
  if (featuresOnly && document.querySelector('[data-tab="timeline"]').classList.contains("active")) {
    document.querySelector('[data-tab="overview"]').click();
  }
  if (featuresOnly && document.querySelector('[data-tab="timeline"]').classList.contains("active")) {
    document.querySelector('[data-tab="overview"]').click();
  }
  if (featuresOnly) {
    predictionUnavailable.querySelector("p").textContent = result.filename_annotation
      ? `This label comes from the CREMA-D filename; it is not a model prediction. ${result.analysis_warning || result.reliability.message}`
      : result.reliability.message;
    document.querySelector("#dominant-emotion").textContent = "Unavailable";
    document.querySelector("#estimate-note").textContent = "No trained emotion estimate is available.";
  } else {
    document.querySelector("#dominant-emotion").textContent =
      result.summary.label === "uncertain" ? "Uncertain" : result.summary.label;
    document.querySelector("#estimate-note").textContent = result.summary.uncertain
      ? `Candidate: ${result.summary.candidate_label} · below the validation threshold`
      : result.reliability.threshold == null
      ? "Uncalibrated model estimate; not a fact about the speaker"
      : "Model output, not a fact about the speaker";
    document.querySelector("#top-score").textContent = `${(result.summary.top_score * 100).toFixed(1)}%`;
  }
  document.querySelector("#duration").textContent = formatDuration(result.audio.duration_seconds);
  document.querySelector("#sample-details").textContent = `${result.audio.sample_rate_hz.toLocaleString()} Hz · ${result.audio.channels} channel(s)`;
  const bars = document.querySelector("#score-bars");
  bars.replaceChildren(...Object.entries(result.summary ? result.summary.scores : {}).map(([label, score]) => {
    const row = document.createElement("div");
    row.className = "score-row";
    const name = document.createElement("span");
    name.textContent = label;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
    track.append(fill);
    const number = document.createElement("span");
    number.className = "score-number";
    number.textContent = `${(score * 100).toFixed(1)}%`;
    row.append(name, track, number);
    return row;
  }));
  if (!featuresOnly) renderTimeline(result);
  const acoustic = result.features.acoustic;
  const entries = [
    ["Median pitch", acoustic.pitch_median_hz == null ? "Unavailable" : `${acoustic.pitch_median_hz.toFixed(0)} Hz`],
    ["Mean RMS energy", acoustic.rms_mean == null ? "Unavailable" : acoustic.rms_mean.toFixed(3)],
    ["Spectral centroid", acoustic.spectral_centroid_mean_hz == null ? "Unavailable" : `${acoustic.spectral_centroid_mean_hz.toFixed(0)} Hz`],
    ["Zero-crossing rate", acoustic.zero_crossing_rate_mean == null ? "Unavailable" : acoustic.zero_crossing_rate_mean.toFixed(3)],
    ["Envelope peaks (rough rate proxy)", acoustic.speech_rate_proxy_peaks_per_second == null ? "Unavailable" : `${acoustic.speech_rate_proxy_peaks_per_second.toFixed(1)} / sec`],
    ["Energy-active audio (rough estimate)", result.features.voice_activity == null ? "Unavailable" : `${(result.features.voice_activity.estimated_active_fraction * 100).toFixed(0)}%`],
  ];
  document.querySelector("#feature-grid").replaceChildren(...entries.map(([label, value]) => {
    const chip = document.createElement("div");
    chip.className = "feature-chip";
    const title = document.createElement("span");
    title.textContent = label;
    const text = document.createElement("strong");
    text.textContent = value;
    chip.append(title, text);
    return chip;
  }));
  const attribution = result.explainability;
  const attributionCard = document.querySelector("#attribution-card");
  const attributionStatus = document.querySelector("#attribution-status");
  const attributionNote = document.querySelector("#attribution-note");
  attributionCard.hidden = featuresOnly;
  if (attribution && attribution.status === "available") {
    attributionStatus.textContent = "MEASURED";
    attributionNote.textContent = `${attribution.method} for candidate “${attribution.target_class}”. Relative sensitivity only; not causal evidence.`;
    if (document.querySelector("#tab-audio-analysis").getBoundingClientRect().width) {
      drawHeatmap(document.querySelector("#attribution-chart"), attribution.feature_sensitivity);
    }
  } else {
    attributionStatus.textContent = "UNAVAILABLE";
    attributionNote.textContent = featuresOnly
      ? "No trained model is installed, so model input sensitivity cannot be calculated."
      : attribution?.limitation || "The model sensitivity map could not be calculated for this result.";
  }
  const reliability = document.querySelector("#reliability-content");
  reliability.replaceChildren();
  if (featuresOnly) {
    const message = document.createElement("p");
    message.textContent = result.reliability.message;
    reliability.append(message);
  }
  const items = [
    `Model: ${result.reliability.model_version || "unknown"} (${result.reliability.architecture || "unknown"}).`,
    result.reliability.threshold == null
      ? result.summary
        ? "No local validation rejection threshold is available; class estimates and softmax scores are uncalibrated."
        : "No emotion prediction or confidence score was produced."
      : result.reliability.threshold > 1
      ? "The validation accuracy target was not reached; all predictions are rejected as uncertain."
      : `Uncertainty threshold: ${(result.reliability.threshold * 100).toFixed(1)}%. Predictions below it are rejected.`,
    ...(result.reliability.score_description ? [result.reliability.score_description] : []),
    ...(result.reliability.segment_count == null
      ? []
      : [`Uncertain segments: ${result.reliability.uncertain_segment_count} of ${result.reliability.segment_count}.`]),
    `Out-of-distribution detection: ${result.reliability.out_of_distribution_detection || "not evaluated"}.`,
    result.audio.preprocessing + ".",
    ...(result.features.voice_activity
      ? [
          `Energy-based activity estimate: ${(result.features.voice_activity.estimated_active_fraction * 100).toFixed(0)}% active; this may mark noise as active or miss quiet speech.`,
        ]
      : []),
    result.caveat,
  ];
  const list = document.createElement("ul");
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    list.append(li);
  }
  reliability.append(list);
  const suggestion = document.querySelector("#suggestion-text");
  suggestion.value = featuresOnly
    ? "Emotion-based suggestions are unavailable because this analysis has no prediction."
    : result.reliability.threshold == null
      ? "Suggestions are disabled because this model has no locally evaluated rejection threshold."
      : result.summary.uncertain
    ? "No emotion-based suggestion is available because this result is uncertain. If useful, ask a neutral, open question."
    : suggestionFor(result.summary.label);
  suggestion.disabled = featuresOnly || result.summary.uncertain || result.reliability.threshold == null;
  drawVisualizations(result);
}

function suggestionFor(label) {
  const suggestions = {
    angry: "Consider acknowledging a concern and asking what help would be useful.",
    sad: "Consider responding gently and giving the speaker room to explain.",
    happy: "You could match a positive conversational tone while attending to the actual words.",
    surprised: "Consider asking an open question about what surprised them.",
    neutral: "No particular response is indicated; respond to the person's words and context.",
    calm: "A steady, open response may fit; use the conversational context.",
    fearful: "Consider offering a calm, supportive response without making assumptions.",
    disgust: "Consider asking neutrally what is troubling them.",
  };
  return suggestions[label] || "No suggestion is available.";
}

function renderTimeline(result) {
  const segments = result.segments;
  const labels = segments.map((segment) => segment.label);
  for (let index = 1; index < labels.length - 1; index += 1) {
    const neighbors = [segments[index - 1].candidate_label, segments[index].candidate_label, segments[index + 1].candidate_label];
    const counts = new Map();
    for (const label of neighbors) counts.set(label, (counts.get(label) || 0) + 1);
    const winner = [...counts].sort((a, b) => b[1] - a[1])[0];
    if (
      winner &&
      result.reliability.threshold != null &&
      winner[1] >= 2 &&
      segments[index].scores[winner[0]] >= result.reliability.threshold
    ) labels[index] = winner[0];
  }
  const list = document.querySelector("#timeline-list");
  list.replaceChildren(...segments.map((segment, index) => {
    const row = document.createElement("div");
    row.className = "timeline-item";
    const time = document.createElement("span");
    time.className = "timeline-time";
    time.textContent = `${formatDuration(segment.start_seconds)}–${formatDuration(segment.end_seconds)}`;
    const label = document.createElement("span");
    label.className = "timeline-label";
    label.textContent = labels[index] === "uncertain" ? `Uncertain · ${segment.candidate_label}` : labels[index];
    const confidence = document.createElement("span");
    confidence.className = "timeline-confidence";
    confidence.textContent = `${(segment.score * 100).toFixed(1)}% raw score`;
    const play = document.createElement("button");
    play.type = "button";
    play.className = "segment-play";
    play.textContent = "Play segment";
    play.addEventListener("click", () => {
      player.currentTime = segment.start_seconds;
      player.play();
      const stopAt = () => {
        if (player.currentTime >= segment.end_seconds) {
          player.pause();
          player.removeEventListener("timeupdate", stopAt);
        }
      };
      player.addEventListener("timeupdate", stopAt);
    });
    row.append(time, label, confidence, play);
    return row;
  }));
  drawTimelineCharts(result);
}

function setupCanvas(canvas, height) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return null;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(height * ratio));
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, height);
  return { context, width: rect.width, height };
}

function drawTimelineCharts(result) {
  if (!result.segments?.length || !result.summary) return;
  const labels = Object.keys(result.summary.scores);
  const duration = result.audio.duration_seconds;
  const track = setupCanvas(document.querySelector("#emotion-track"), Math.max(150, labels.length * 24 + 32));
  if (track) {
    const { context, width, height } = track;
    const left = 96;
    const right = 12;
    const top = 30;
    const bottom = 30;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const laneHeight = plotHeight / labels.length;
    context.font = "10px Arial";
    context.textBaseline = "middle";
    context.textAlign = "left";
    context.fillStyle = "#777c91";
    context.fillText("Candidate class", 4, 12);
    const legendX = Math.max(left + 100, width - 155);
    context.fillStyle = "#7568dd";
    context.fillRect(legendX, 7, 12, 10);
    context.fillStyle = "#777c91";
    context.fillText("estimate", legendX + 16, 12);
    context.strokeStyle = "#5648bd";
    context.setLineDash([3, 2]);
    context.strokeRect(legendX + 69, 7, 12, 10);
    context.setLineDash([]);
    context.fillStyle = "#777c91";
    context.fillText("rejected", legendX + 85, 12);
    labels.forEach((label, index) => {
      const y = top + index * laneHeight;
      context.fillStyle = "#777c91";
      context.fillText(label, 4, y + laneHeight / 2);
      context.strokeStyle = "#eeeef4";
      context.beginPath();
      context.moveTo(left, y + laneHeight);
      context.lineTo(width - right, y + laneHeight);
      context.stroke();
    });
    for (let tick = 0; tick <= 4; tick += 1) {
      const fraction = tick / 4;
      const x = left + fraction * plotWidth;
      context.strokeStyle = "#dedfea";
      context.beginPath();
      context.moveTo(x, top);
      context.lineTo(x, top + plotHeight);
      context.stroke();
      context.fillStyle = "#777c91";
      context.textAlign = tick === 0 ? "left" : tick === 4 ? "right" : "center";
      context.fillText(`${(duration * fraction).toFixed(1)}s`, x, height - 17);
    }
    result.segments.forEach((segment) => {
      const candidate = segment.candidate_label;
      const rowIndex = labels.indexOf(candidate);
      if (rowIndex < 0) return;
      const scale = plotWidth / Math.max(duration, 0.001);
      const x = left + segment.start_seconds * scale;
      const barWidth = Math.max(2, (segment.end_seconds - segment.start_seconds) * scale - 1);
      const y = top + rowIndex * laneHeight + 4;
      const barHeight = Math.max(4, laneHeight - 8);
      context.fillStyle = "#7568dd";
      context.globalAlpha = segment.label === "uncertain" ? 0.24 : 0.8;
      context.fillRect(x, y, barWidth, barHeight);
      context.globalAlpha = 1;
      if (segment.label === "uncertain") {
        context.strokeStyle = "#5648bd";
        context.setLineDash([3, 2]);
        context.strokeRect(x, y, barWidth, barHeight);
        context.setLineDash([]);
      }
    });
    context.textAlign = "center";
    context.fillStyle = "#777c91";
    context.textBaseline = "bottom";
    context.fillText("Time (seconds)", left + plotWidth / 2, height - 4);
  }
  const trend = setupCanvas(document.querySelector("#confidence-trend"), 190);
  if (!trend) return;
  const { context, width, height } = trend;
  const left = 58;
  const right = 12;
  const top = 12;
  const bottom = 28;
  const plotHeight = height - top - bottom;
  const plotWidth = width - left - right;
  context.font = "10px Arial";
  context.fillStyle = "#777c91";
  context.textAlign = "right";
  [0, 0.5, 1].forEach((value) => {
    const y = top + (1 - value) * plotHeight;
    context.strokeStyle = "#eeeef4";
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(width - right, y);
    context.stroke();
    context.fillText(`${Math.round(value * 100)}%`, 4, y + 3);
  });
  context.textAlign = "center";
  for (let tick = 0; tick <= 4; tick += 1) {
    const fraction = tick / 4;
    const x = left + fraction * plotWidth;
    context.fillText(`${(duration * fraction).toFixed(1)}s`, x, height - 17);
  }
  context.save();
  context.translate(12, top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = "center";
  context.fillText("Top raw class score", 0, 0);
  context.restore();
  const threshold = result.reliability.threshold;
  if (threshold != null && threshold <= 1) {
    const y = top + (1 - threshold) * plotHeight;
    context.strokeStyle = "#df7a88";
    context.setLineDash([5, 4]);
    context.beginPath();
    context.moveTo(left, y);
    context.lineTo(width - right, y);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = "#b85d6a";
    context.fillText("threshold", Math.max(left + 2, width - right - 62), y - 4);
  }
  context.strokeStyle = "#6758d8";
  context.lineWidth = 2;
  context.beginPath();
  result.segments.forEach((segment, index) => {
    const center = (segment.start_seconds + segment.end_seconds) / 2;
    const x = left + (center / Math.max(duration, 0.001)) * plotWidth;
    const y = top + (1 - segment.score) * plotHeight;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
  result.segments.forEach((segment) => {
    const center = (segment.start_seconds + segment.end_seconds) / 2;
    const x = left + (center / Math.max(duration, 0.001)) * plotWidth;
    const y = top + (1 - segment.score) * plotHeight;
    context.fillStyle = segment.label === "uncertain" ? "#df7a88" : "#6758d8";
    context.beginPath();
    context.arc(x, y, 3.5, 0, Math.PI * 2);
    context.fill();
  });
  context.fillStyle = "#777c91";
  context.textAlign = "center";
  context.textBaseline = "bottom";
  context.fillText("Time (seconds)", left + plotWidth / 2, height - 6);
}

function drawLineChart(
  canvas,
  values,
  color = "#6b5be0",
  { xLabel, yLabel, xMax = values.length - 1, xFormatter } = {},
) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  context.clearRect(0, 0, width, height);
  const left = 72;
  const right = 12;
  const top = 12;
  const bottom = 36;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  context.font = "10px Arial";
  context.textAlign = "right";
  context.textBaseline = "middle";
  context.strokeStyle = "#eeedf5";
  context.lineWidth = 1;
  context.beginPath();
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = top + (tick / 4) * plotHeight;
    context.moveTo(left, y);
    context.lineTo(width - right, y);
  }
  context.stroke();
  const flat = Array.isArray(values[0]) ? values.flat() : values;
  if (!flat.length) return;
  const dataLow = Math.min(...flat);
  const dataHigh = Math.max(...flat);
  const padding = (dataHigh - dataLow) * 0.08 || 0.5;
  const low = dataLow - padding;
  const high = dataHigh + padding;
  const range = high - low || 1;
  context.fillStyle = "#777c91";
  for (let tick = 0; tick <= 4; tick += 1) {
    const fraction = 1 - tick / 4;
    const y = top + (tick / 4) * plotHeight;
    const value = low + fraction * range;
    context.fillText(value.toFixed(1), left - 8, y);
  }
  context.textAlign = "center";
  for (let tick = 0; tick <= 4; tick += 1) {
    const fraction = tick / 4;
    const x = left + fraction * plotWidth;
    const xValue = xMax * fraction;
    context.fillText(xFormatter ? xFormatter(xValue) : xValue.toFixed(0), x, height - 23);
  }
  context.save();
  context.translate(10, top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText(yLabel || "Value", 0, 0);
  context.restore();
  context.textBaseline = "bottom";
  context.fillText(xLabel || "Sample index", left + plotWidth / 2, height - 5);
  context.strokeStyle = color;
  context.lineWidth = 1.5;
  context.beginPath();
  flat.forEach((value, index) => {
    const x = left + (flat.length <= 1 ? plotWidth / 2 : (index / (flat.length - 1)) * plotWidth);
    const y = top + plotHeight - ((value - low) / range) * plotHeight;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

function drawHeatmap(canvas, values, { durationSeconds, yLabel = "Mel-frequency bin" } = {}) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const rows = values.length;
  if (!rows) return;
  const columns = Math.max(...values.map((row) => row.length));
  if (!rows || !columns) return;
  const left = 50;
  const right = 12;
  const top = 12;
  const bottom = 34;
  const plotWidth = rect.width - left - right;
  const plotHeight = rect.height - top - bottom;
  const cellWidth = plotWidth / columns;
  const cellHeight = plotHeight / rows;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < values[row].length; column += 1) {
      const intensity = Math.max(0, Math.min(1, (values[row][column] + 80) / 80));
      context.fillStyle = `hsl(${260 - intensity * 230} 65% ${18 + intensity * 60}%)`;
      context.fillRect(left + column * cellWidth, top + plotHeight - (row + 1) * cellHeight, Math.ceil(cellWidth), Math.ceil(cellHeight));
    }
  }
  context.font = "10px Arial";
  context.fillStyle = "#777c91";
  context.textAlign = "right";
  context.textBaseline = "middle";
  [0, 0.5, 1].forEach((fraction) => {
    const y = top + (1 - fraction) * plotHeight;
    context.fillText(`${Math.round(rows * fraction)}`, left - 7, y);
  });
  context.textAlign = "center";
  [0, 0.5, 1].forEach((fraction) => {
    const x = left + fraction * plotWidth;
    context.fillText(`${((durationSeconds || 0) * fraction).toFixed(1)}s`, x, rect.height - 21);
  });
  context.save();
  context.translate(12, top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText(yLabel, 0, 0);
  context.restore();
  context.textAlign = "center";
  context.textBaseline = "bottom";
  context.fillText("Time (seconds)", left + plotWidth / 2, rect.height - 4);
}

function drawVisualizations(result) {
  const waveform = document.querySelector("#waveform");
  if (!waveform.getBoundingClientRect().width) return;
  drawLineChart(waveform, result.features.waveform, "#6b5be0", {
    xLabel: "Time (seconds)",
    yLabel: "Normalized amplitude",
    xMax: result.audio.duration_seconds,
    xFormatter: (value) => `${value.toFixed(1)}s`,
  });
  drawHeatmap(document.querySelector("#mel-chart"), result.features.log_mel, {
    durationSeconds: result.audio.duration_seconds,
  });
  if (result.explainability && result.explainability.status === "available") {
    drawHeatmap(
      document.querySelector("#attribution-chart"),
      result.explainability.feature_sensitivity,
      { durationSeconds: result.audio.duration_seconds, yLabel: "Mel-frequency bin" },
    );
  }
  const mfcc = result.features.mfcc;
  const meanMfcc = mfcc.map((row) => row.reduce((sum, value) => sum + value, 0) / row.length);
  drawLineChart(document.querySelector("#mfcc-chart"), meanMfcc, "#e07d87", {
    xLabel: "MFCC coefficient",
    yLabel: "Mean coefficient value",
    xMax: Math.max(0, meanMfcc.length - 1),
  });
}
window.addEventListener("resize", () => {
  if (!currentResult) return;
  drawVisualizations(currentResult);
  if (document.querySelector('[data-tab="timeline"]').classList.contains("active")) {
    drawTimelineCharts(currentResult);
  }
});

function downloadFile(filename, contents, type) {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
document.querySelector("#export-json").addEventListener("click", () => {
  if (currentResult) downloadFile("emotionlens-analysis.json", JSON.stringify(currentResult, null, 2), "application/json");
});
document.querySelector("#export-csv").addEventListener("click", () => {
  if (!currentResult || currentResult.status === "features_only") return;
  const columns = [
    "analyzed_at_utc",
    "model_version",
    "uncertainty_threshold",
    "preprocessing",
    "start_seconds",
    "end_seconds",
    "label",
    "candidate_label",
    "score",
  ];
  const metadata = [
    currentResult.analyzed_at_utc,
    currentResult.reliability.model_version,
    currentResult.reliability.threshold,
    currentResult.audio.preprocessing,
  ];
  const lines = [
    columns.join(","),
    ...currentResult.segments.map((segment) => [
      ...metadata,
      segment.start_seconds,
      segment.end_seconds,
      segment.label,
      segment.candidate_label,
      segment.score,
    ].map((value) => JSON.stringify(value ?? "")).join(",")),
  ];
  downloadFile("emotionlens-timeline.csv", lines.join("\n"), "text/csv");
});
document.querySelector("#show-model-setup").addEventListener("click", () => {
  const setup = document.querySelector("#model-setup");
  setup.open = true;
  setup.scrollIntoView({ behavior: "smooth", block: "center" });
});
document.querySelector("#open-evaluation").addEventListener("click", () => {
  const results = document.querySelector("#results");
  results.hidden = false;
  document.querySelector('[data-tab="evaluation"]').click();
  results.scrollIntoView({ behavior: "smooth", block: "start" });
});

async function loadApplicationInfo() {
  try {
    const [modelResponse, evaluationResponse, comparisonResponse, robustnessResponse, cardResponse, crossCorpusResponse, datasetAuditResponse] = await Promise.all([
      fetch("/api/model"),
      fetch("/api/evaluation"),
      fetch("/api/comparison"),
      fetch("/api/robustness"),
      fetch("/api/model-card"),
      fetch("/api/cross-corpus"),
      fetch("/api/dataset-audit"),
    ]);
    const model = await modelResponse.json();
    const evaluation = await evaluationResponse.json();
    const comparison = await comparisonResponse.json();
    const robustness = await robustnessResponse.json();
    const modelCard = await cardResponse.json();
    const crossCorpus = await crossCorpusResponse.json();
    const datasetAudit = await datasetAuditResponse.json();
    window.modelAvailable = model.available;
    const modelSelect = document.querySelector("#model-select");
    modelSelect.replaceChildren();
    const architectures = model.available_architectures || [];
    if (architectures.length) {
      for (const architecture of architectures) {
        const option = document.createElement("option");
        option.value = architecture;
        option.textContent = architectureLabel(architecture);
        modelSelect.append(option);
      }
      modelSelect.value = architectures.includes(model.architecture)
        ? model.architecture
        : architectures[0];
    } else {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "CNN–LSTM · awaiting checkpoint";
      modelSelect.append(option);
    }
    const pretrainedAvailable = architectures.includes("superb_wav2vec2_base_er");
    document.querySelector("#readiness-title").textContent = model.available
      ? `Model ready · ${model.model_version}`
      : model.load_error
        ? "Model checkpoint could not be loaded"
        : pretrainedAvailable
          ? "Pretrained model available"
          : "Trained model not installed";
    document.querySelector("#readiness-text").textContent = model.available
      ? `${model.dataset || "Local dataset"} model ready. It is not validated for other corpora; scores are raw and may be uncertain.`
      : model.load_error || "You can analyze audio features now. Emotion classification requires a trained model checkpoint.";
    if (!model.available && pretrainedAvailable) {
      document.querySelector("#readiness-text").textContent =
        "Select SUPERB wav2vec2 to download its model files on first use. It predicts four English IEMOCAP-derived classes locally; its scores are uncalibrated and not validated by EmotionLens.";
    }
    document.querySelector("#model-setup").hidden = model.available;
    analyseButton.disabled = !selectedFile || !document.querySelector("#consent").checked;
    if (!model.available) {
      document.querySelector("#readiness-card").style.borderColor = "#efd8ab";
    }
    if (evaluation.status === "evaluated") {
      const card = document.querySelector("#evaluation-card");
      card.classList.add("evaluated");
      const heading = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = `Fixed held-out test results · ${evaluation.model_version}`;
      const paragraph = document.createElement("p");
      const selectiveAccuracy = evaluation.selective_accuracy == null
        ? "n/a"
        : `${(evaluation.selective_accuracy * 100).toFixed(1)}%`;
      paragraph.textContent = `Speaker-independent · ${evaluation.test_sample_count} test clips · Accuracy ${(evaluation.accuracy * 100).toFixed(1)}% · Macro F1 ${(evaluation.macro_f1 * 100).toFixed(1)}% · Macro precision ${(evaluation.macro_precision * 100).toFixed(1)}% · Macro recall ${(evaluation.macro_recall * 100).toFixed(1)}% · Selective coverage ${(evaluation.selective_coverage * 100).toFixed(1)}% · Selective accuracy ${selectiveAccuracy}.`;
      const latency = document.createElement("p");
      latency.textContent = `Measured preprocessing + inference time: ${evaluation.test_preprocessing_and_inference_latency_ms_per_clip.toFixed(1)} ms/clip. This is not a real-time performance claim.`;
      heading.append(title, paragraph, latency);

      const report = document.createElement("div");
      report.className = "evaluation-report";
      const evaluationCopy = document.querySelector("#evaluation-detail");
      evaluationCopy.replaceChildren();
      const evaluationSummary = document.createElement("p");
      evaluationSummary.textContent = paragraph.textContent;
      evaluationCopy.append(title.cloneNode(true), evaluationSummary.cloneNode(true), latency.cloneNode(true));
      const perClassHeading = document.createElement("h3");
      perClassHeading.textContent = "Per-class metrics";
      report.append(perClassHeading, createMetricsTable(evaluation));
      const matrixHeading = document.createElement("h3");
      matrixHeading.textContent = "Confusion matrix · rows actual, columns predicted";
      report.append(matrixHeading, createConfusionMatrix(evaluation));
      const trainingHeading = document.createElement("h3");
      trainingHeading.textContent = "Training and validation loss";
      const trainingChart = document.createElement("canvas");
      trainingChart.className = "training-chart";
      trainingChart.setAttribute("aria-label", "Training and validation loss curves");
      const history = model.training && model.training.history;
      if (Array.isArray(history) && history.length) {
        report.append(trainingHeading, trainingChart);
        requestAnimationFrame(() => drawHistoryChart(trainingChart, history));
      } else {
        const unavailable = document.createElement("p");
        unavailable.textContent = "Training curves are unavailable in the saved model metadata.";
        report.append(trainingHeading, unavailable);
      }

      if (evaluation.calibration) {
        const calibrationHeading = document.createElement("h3");
        calibrationHeading.textContent = "Calibration and selective risk";
        const calibrationSummary = document.createElement("p");
        calibrationSummary.textContent = `Held-out expected calibration error: ${(evaluation.calibration.expected_calibration_error * 100).toFixed(1)}%. Scores remain raw softmax outputs and are not calibrated.`;
        report.append(calibrationHeading, calibrationSummary);
        const calibrationTable = createSimpleTable(
          ["Confidence threshold", "Coverage", "Selective risk"],
          evaluation.calibration.coverage_risk.map((point) => [
            `${(point.threshold * 100).toFixed(1)}%`,
            `${(point.coverage * 100).toFixed(1)}%`,
            `${(point.selective_risk * 100).toFixed(1)}%`,
          ]),
        );
        report.append(calibrationTable);
      }
      card.replaceChildren(heading, report, document.querySelector("#open-evaluation"));
      evaluationCopy.append(report.cloneNode(true));
    } else {
      const evaluationCopy = document.querySelector("#evaluation-detail");
      const message = document.createElement("p");
      message.textContent = evaluation.message || "Not evaluated. Run training with held-out speakers to calculate results.";
      evaluationCopy.replaceChildren(document.createTextNode("Not evaluated"), message);
    }

    renderComparison(comparison);
    renderRobustness(robustness);
    renderCrossCorpus(crossCorpus);
    renderDatasetAudit(datasetAudit);
    renderModelCard(modelCard, model);
    const redrawEvaluationCharts = () => requestAnimationFrame(() => {
      if (robustness.status === "evaluated") drawRobustnessChart(robustness);
      const history = model.training && model.training.history;
      if (Array.isArray(history) && history.length) {
        document.querySelectorAll(".training-chart").forEach((canvas) => {
          drawHistoryChart(canvas, history);
        });
      }
    });
    document.querySelector('[data-tab="evaluation"]').addEventListener("click", redrawEvaluationCharts);
    window.addEventListener("resize", () => {
      if (document.querySelector('[data-tab="evaluation"]').classList.contains("active")) {
        redrawEvaluationCharts();
      }
    });
  } catch {
    document.querySelector("#readiness-title").textContent = "Backend is unavailable";
    document.querySelector("#readiness-text").textContent = "Start the local API server to load model information.";
  }

  function createMetricsTable(evaluation) {
    const table = document.createElement("table");
    table.className = "metrics-table";
    table.setAttribute("aria-label", "Per-class precision, recall, F1, and support");
    const header = document.createElement("tr");
    for (const label of ["Class", "Precision", "Recall", "F1", "Support"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      header.append(cell);
    }
    const head = document.createElement("thead");
    head.append(header);
    const body = document.createElement("tbody");
    for (const label of evaluation.class_labels) {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = label;
      row.append(name);
      const metric = evaluation.per_class[label];
      for (const value of [metric.precision, metric.recall, metric["f1-score"]]) {
        const cell = document.createElement("td");
        cell.textContent = `${(value * 100).toFixed(1)}%`;
        row.append(cell);
      }
      const support = document.createElement("td");
      support.textContent = String(metric.support);
      row.append(support);
      body.append(row);
    }
    table.append(head, body);
    return table;
  }

  function createConfusionMatrix(evaluation) {
    const table = document.createElement("table");
    table.className = "metrics-table confusion-table";
    table.setAttribute("aria-label", "Confusion matrix");
    const headRow = document.createElement("tr");
    const corner = document.createElement("th");
    corner.scope = "col";
    corner.textContent = "Actual ↓ / Predicted →";
    headRow.append(corner);
    for (const label of evaluation.class_labels) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      headRow.append(cell);
    }
    const head = document.createElement("thead");
    head.append(headRow);
    const body = document.createElement("tbody");
    evaluation.confusion_matrix.forEach((values, rowIndex) => {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = evaluation.class_labels[rowIndex];
      row.append(name);
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = String(value);
        row.append(cell);
      });
      body.append(row);
    });
    table.append(head, body);
    return table;
  }

  function createSimpleTable(headers, rows) {
    const table = document.createElement("table");
    table.className = "metrics-table";
    const head = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const label of headers) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      headerRow.append(cell);
    }
    head.append(headerRow);
    const body = document.createElement("tbody");
    for (const values of rows) {
      const row = document.createElement("tr");
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = String(value);
        row.append(cell);
      }
      body.append(row);
    }
    table.append(head, body);
    return table;
  }

  function renderComparison(report) {
    const target = document.querySelector("#comparison-detail");
    if (report.status !== "evaluated") {
      target.replaceChildren(
        document.createTextNode("Not evaluated"),
        document.createElement("p"),
      );
      target.querySelector("p").textContent =
        report.message || "Train the comparison models on identical speaker splits using --compare.";
      return;
    }
    const summary = document.createElement("p");
    summary.textContent = `${report.split_method}; seed ${report.random_seed}. Selected primary model: ${architectureLabel(report.selected_primary_model)}.`;
    const table = createSimpleTable(
      ["Architecture", "Accuracy", "Macro F1", "Macro precision", "Macro recall", "Test clips", "Latency / clip"],
      Object.entries(report.models).map(([name, metrics]) => [
        architectureLabel(name),
        `${(metrics.accuracy * 100).toFixed(1)}%`,
        `${(metrics.macro_f1 * 100).toFixed(1)}%`,
        `${(metrics.macro_precision * 100).toFixed(1)}%`,
        `${(metrics.macro_recall * 100).toFixed(1)}%`,
        metrics.test_sample_count,
        `${metrics.test_preprocessing_and_inference_latency_ms_per_clip.toFixed(1)} ms`,
      ]),
    );
    target.replaceChildren(summary, table);
    requestAnimationFrame(() => drawRobustnessChart(report));
  }

  function drawRobustnessChart(report) {
    const canvas = document.querySelector("#robustness-chart");
    const chart = setupCanvas(canvas, 230);
    if (!chart) return;
    const { context, width, height } = chart;
    const entries = Object.values(report.conditions);
    if (!entries.length) return;
    const left = 42;
    const right = 12;
    const top = 12;
    const bottom = 74;
    const plotHeight = height - top - bottom;
    const plotWidth = width - left - right;
    context.font = "10px Arial";
    context.textAlign = "right";
    context.textBaseline = "middle";
    [0, 0.5, 1].forEach((value) => {
      const y = top + (1 - value) * plotHeight;
      context.strokeStyle = "#eeeef4";
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(width - right, y);
      context.stroke();
      context.fillStyle = "#777c91";
      context.fillText(`${Math.round(value * 100)}%`, left - 7, y);
    });
    context.save();
    context.translate(12, top + plotHeight / 2);
    context.rotate(-Math.PI / 2);
    context.textAlign = "center";
    context.fillStyle = "#777c91";
    context.fillText("Macro F1", 0, 0);
    context.restore();
    const points = entries.map((entry, index) => ({
      x: left + (entries.length === 1 ? plotWidth / 2 : index * plotWidth / (entries.length - 1)),
      y: top + (1 - entry.macro_f1) * plotHeight,
      entry,
    }));
    context.strokeStyle = "#6758d8";
    context.lineWidth = 2;
    context.beginPath();
    points.forEach((point, index) => {
      if (index === 0) context.moveTo(point.x, point.y);
      else context.lineTo(point.x, point.y);
    });
    context.stroke();
    context.textAlign = "center";
    context.textBaseline = "top";
    points.forEach(({ x, y, entry }) => {
      context.fillStyle = "#6758d8";
      context.beginPath();
      context.arc(x, y, 4, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#5f6377";
      context.fillText(`${(entry.macro_f1 * 100).toFixed(1)}%`, x, y - 14);
      const label = entry.snr_db == null
        ? ["Clean"]
        : entry.denoising !== "none"
          ? [`${entry.snr_db} dB`, "+ Wiener"]
          : [`${entry.snr_db} dB`];
      label.forEach((line, lineIndex) => {
        context.fillText(line, x, height - bottom + 8 + lineIndex * 12);
      });
    });
    context.fillStyle = "#777c91";
    context.textBaseline = "bottom";
    context.fillText("Test condition", left + plotWidth / 2, height - 2);
  }

  function renderRobustness(report) {
    const target = document.querySelector("#robustness-detail");
    if (report.status !== "evaluated") {
      document.querySelector("#robustness-chart").hidden = true;
      document.querySelector("#robustness-chart-note").hidden = true;
      const message = document.createElement("p");
      message.textContent = report.message || "Not evaluated. Train with --robustness to test measured held-out audio.";
      target.replaceChildren(document.createTextNode("Not evaluated"), message);
      return;
    }
    document.querySelector("#robustness-chart").hidden = false;
    document.querySelector("#robustness-chart-note").hidden = false;
    const summary = document.createElement("p");
    summary.textContent = report.note;
    const table = createSimpleTable(
      ["Condition", "SNR", "Accuracy", "Macro precision", "Macro recall", "Macro F1", "Samples"],
      Object.entries(report.conditions).map(([name, result]) => [
        name,
        result.snr_db == null ? "Clean" : `${result.snr_db} dB`,
        `${(result.accuracy * 100).toFixed(1)}%`,
        `${(result.macro_precision * 100).toFixed(1)}%`,
        `${(result.macro_recall * 100).toFixed(1)}%`,
        `${(result.macro_f1 * 100).toFixed(1)}%`,
        result.sample_count,
      ]),
    );
    target.replaceChildren(summary, table);
  }

  function renderCrossCorpus(report) {
    const target = document.querySelector("#cross-corpus-detail");
    if (report.status !== "evaluated") {
      const message = document.createElement("p");
      message.textContent = report.message
        || "Not evaluated. Use a separate, permitted target corpus and the cross-corpus CLI; target data is never used to fine-tune or select thresholds.";
      target.replaceChildren(document.createTextNode("Not evaluated"), message);
      return;
    }
    const summary = document.createElement("p");
    summary.textContent = `${report.source_dataset} model ${report.source_model_version} → ${report.target_dataset}; ${report.target_sample_count} clips from ${report.target_speaker_count} target speakers. ${report.evaluation_protocol}`;
    const metrics = document.createElement("p");
    metrics.textContent = `Accuracy ${(report.accuracy * 100).toFixed(1)}% · Macro F1 ${(report.macro_f1 * 100).toFixed(1)}% · Macro precision ${(report.macro_precision * 100).toFixed(1)}% · Macro recall ${(report.macro_recall * 100).toFixed(1)}% · Coverage ${(report.selective_coverage * 100).toFixed(1)}% · Target ECE on shared labels ${(report.target_expected_calibration_error_on_shared_labels * 100).toFixed(1)}%.`;
    const perClass = createSimpleTable(
      ["Shared class", "Precision", "Recall", "F1", "Support"],
      report.shared_class_labels.map((label) => [
        label,
        `${(report.per_class[label].precision * 100).toFixed(1)}%`,
        `${(report.per_class[label].recall * 100).toFixed(1)}%`,
        `${(report.per_class[label]["f1-score"] * 100).toFixed(1)}%`,
        report.per_class[label].support,
      ]),
    );
    const caveat = document.createElement("p");
    caveat.textContent = `${report.note} Excluded target labels: ${report.excluded_target_class_labels.join(", ") || "none"}.`;
    target.replaceChildren(summary, metrics, perClass, caveat);
  }

  function renderDatasetAudit(report) {
    const target = document.querySelector("#dataset-audit-detail");
    if (report.status !== "audited") {
      const message = document.createElement("p");
      message.textContent = report.message
        || "Not audited. Run the dataset audit command before training; dataset licensing and label correctness still need human review.";
      target.replaceChildren(document.createTextNode("Not audited"), message);
      return;
    }
    const summary = document.createElement("p");
    summary.textContent = `${report.dataset}: ${report.valid_labelled_files} valid labelled files, ${report.invalid_labelled_file_count} invalid labelled files, ${report.unrecognized_or_unlabelled_wav_count} unrecognized WAVs, and ${report.speaker_count} speakers.`;
    const distribution = document.createElement("p");
    distribution.textContent = `Class counts: ${Object.entries(report.class_counts).map(([label, count]) => `${label} ${count}`).join(" · ")}.`;
    const media = document.createElement("p");
    media.textContent = `Sample rates: ${Object.entries(report.sample_rate_counts_hz).map(([rate, count]) => `${rate} Hz (${count})`).join(" · ")}. Channels: ${JSON.stringify(report.channel_counts)}. Duration min/median/max: ${report.duration_seconds.minimum.toFixed(2)} / ${report.duration_seconds.median.toFixed(2)} / ${report.duration_seconds.maximum.toFixed(2)} seconds.`;
    const splits = document.createElement("p");
    splits.textContent = report.speaker_split_audit_seed_42
      ? `Seed-42 speaker split audit: ${JSON.stringify(report.speaker_split_audit_seed_42)}`
      : "Fewer than three valid speakers; a train/validation/test split could not be audited.";
    target.replaceChildren(summary, distribution, media, splits);
  }

  function renderModelCard(card, model) {
    const target = document.querySelector("#model-card-detail");
    const lines = [
      `Status: ${card.status}.`,
      `Architecture: ${architectureLabel(card.architecture || "No local trained checkpoint")}.`,
      `Model version: ${card.model_version || "Not available"}.`,
      `Dataset: ${card.dataset || "No dataset recorded"}.`,
      `Dataset version/source: ${model.dataset_version || "Not recorded"} / ${model.dataset_source || "Not recorded"}.`,
      `Random seed: ${model.random_seed ?? "Not recorded"}.`,
      `Speaker split audit: ${model.split_speakers ? JSON.stringify(model.split_speakers) : "Not evaluated"}.`,
      `Intended use: ${card.intended_use}`,
      `Out-of-scope use: ${card.prohibited_use}`,
      ...(model.limitations || card.limitations || []),
      ...(card.pretrained_model
        ? [
            `Separate pretrained model: ${card.pretrained_model.model_id}@${card.pretrained_model.revision}.`,
            `Pretrained labels: ${card.pretrained_model.class_labels.join(", ")}; dataset: ${card.pretrained_model.dataset}.`,
            `Pretrained license: ${card.pretrained_model.license}.`,
            `External result: ${(card.pretrained_model.external_metric.value * 100).toFixed(2)}% ${card.pretrained_model.external_metric.name}; ${card.pretrained_model.external_metric.note}`,
            "Pretrained inference runs locally after the model files download; uploaded audio is not sent to Hugging Face.",
          ]
        : []),
    ];
    const list = document.createElement("ul");
    for (const line of lines) {
      const item = document.createElement("li");
      item.textContent = line;
      list.append(item);
    }
    target.replaceChildren(list);
  }

  function drawHistoryChart(canvas, history) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * ratio));
    const height = 190;
    canvas.height = Math.max(1, Math.floor(height * ratio));
    canvas.style.height = `${height}px`;
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const width = rect.width;
    const losses = [...history.map((item) => item.train_loss), ...history.map((item) => item.validation_loss)];
    const maximum = Math.max(...losses);
    const minimum = Math.min(...losses);
    const range = maximum - minimum || 1;
    const left = 54;
    const right = 12;
    const top = 24;
    const bottom = 38;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    context.font = "10px Arial";
    context.fillStyle = "#777c91";
    context.textAlign = "right";
    context.textBaseline = "middle";
    for (let tick = 0; tick <= 4; tick += 1) {
      const fraction = 1 - tick / 4;
      const y = top + (tick / 4) * plotHeight;
      context.strokeStyle = "#eeeef5";
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(width - right, y);
      context.stroke();
      context.fillText((minimum + range * fraction).toFixed(2), left - 7, y);
    }
    context.textAlign = "center";
    context.textBaseline = "top";
    for (let tick = 0; tick <= 4; tick += 1) {
      const fraction = tick / 4;
      const epoch = 1 + fraction * Math.max(0, history.length - 1);
      context.fillText(`${epoch.toFixed(0)}`, left + fraction * plotWidth, height - bottom + 7);
    }
    context.save();
    context.translate(12, top + plotHeight / 2);
    context.rotate(-Math.PI / 2);
    context.textBaseline = "middle";
    context.fillText("Loss", 0, 0);
    context.restore();
    context.textBaseline = "bottom";
    context.fillText("Epoch", left + plotWidth / 2, height - 2);
    const series = [["train_loss", "#6758d8"], ["validation_loss", "#df7a88"]];
    for (const [key, color] of series) {
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.beginPath();
      history.forEach((item, index) => {
        const x = left + (history.length <= 1 ? plotWidth / 2 : (index / (history.length - 1)) * plotWidth);
        const y = top + plotHeight - ((item[key] - minimum) / range) * plotHeight;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();
    }
    context.textAlign = "left";
    context.textBaseline = "middle";
    series.forEach(([name, color], index) => {
      const x = left + index * 120;
      context.strokeStyle = color;
      context.beginPath();
      context.moveTo(x, 11);
      context.lineTo(x + 20, 11);
      context.stroke();
      context.fillStyle = "#777c91";
      context.fillText(name === "train_loss" ? "Training loss" : "Validation loss", x + 25, 11);
    });
  }
}
loadApplicationInfo();
