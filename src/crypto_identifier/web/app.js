/**
 * CryptoCipher AI - Frontend Application Logic (v0.2.0)
 * Supporting Multiple Architectures, Live Encryption Generator, & 25-Sample Catalog
 */

document.addEventListener("DOMContentLoaded", () => {
  // Application State
  const state = {
    mode: "upload",             // 'upload' or 'generate'
    task: "multiclass",         // 'multiclass' or 'binary'
    architecture: "cnn",        // 'cnn', 'rf', 'svm', 'mlp', 'knn', 'lr', 'gnb', 'all'
    size: "auto",               // 'auto', '1kb', '8kb', '64kb', '256kb', '512kb'
    selectedFile: null,
    models: {},
    allSamples: [],
    activeFilterSize: "all",
    generatedCiphertextBytes: null,
    isAnalyzing: false,
  };

  // DOM Elements - Mode Switching
  const tabUploadBtn = document.getElementById("tab-upload-btn");
  const tabGenerateBtn = document.getElementById("tab-generate-btn");
  const viewUploadContainer = document.getElementById("view-upload-container");
  const viewGenerateContainer = document.getElementById("view-generate-container");

  // DOM Elements - Configuration
  const taskMultiBtn = document.getElementById("task-multi-btn");
  const taskBinaryBtn = document.getElementById("task-binary-btn");
  const archChips = document.querySelectorAll(".arch-chip");
  const sizePills = document.querySelectorAll("#size-pill-group .size-pill");

  // DOM Elements - Upload & Presets
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const selectedFilePanel = document.getElementById("selected-file-panel");
  const fileNameDisplay = document.getElementById("file-name-display");
  const fileSizeDisplay = document.getElementById("file-size-display");
  const hexPreviewBody = document.getElementById("hex-preview-body");
  const predictBtn = document.getElementById("predict-btn");
  const sampleSizeFilter = document.getElementById("sample-size-filter");
  const samplesMatrix = document.getElementById("samples-matrix");

  // DOM Elements - Live Generator
  const genCipherChips = document.querySelectorAll("#gen-cipher-row .cipher-chip");
  const genSizePills = document.querySelectorAll("#gen-size-pills .size-pill");
  const genPlaintextInput = document.getElementById("gen-plaintext-input");
  const btnUuidFill = document.getElementById("btn-uuid-fill");
  const btnGenerateEncrypt = document.getElementById("btn-generate-encrypt");
  const generatedReadoutPanel = document.getElementById("generated-readout-panel");
  const genReadoutBytes = document.getElementById("gen-readout-bytes");
  const genReadoutSha = document.getElementById("gen-readout-sha");
  const genReadoutHex = document.getElementById("gen-readout-hex");
  const btnDownloadGenerated = document.getElementById("btn-download-generated");

  let genSelectedCipher = "AES";
  let genSelectedSize = "1kb";

  // DOM Elements - Results View
  const emptyState = document.getElementById("empty-state");
  const loadingState = document.getElementById("loading-state");
  const loadingArchText = document.getElementById("loading-arch-text");
  const resultsContainer = document.getElementById("results-container");

  const resultTaskPill = document.getElementById("result-task-pill");
  const resultArchPill = document.getElementById("result-arch-pill");
  const resultSizePill = document.getElementById("result-size-pill");
  const groundTruthPill = document.getElementById("ground-truth-pill");
  const predictedCipherName = document.getElementById("predicted-cipher-name");
  const predConfidence = document.getElementById("pred-confidence");


  const archCompareCard = document.getElementById("arch-compare-card");
  const archComparisonGrid = document.getElementById("arch-comparison-grid");
  const probBarsContainer = document.getElementById("prob-bars-container");



  // Theme
  const themeToggle = document.getElementById("theme-toggle");

  // 1. Initialize
  async function init() {
    await fetchModels();
    await fetchAllSamples();
  }

  async function fetchModels() {
    try {
      const res = await fetch("/api/models");
      if (res.ok) {
        const data = await res.json();
        data.models.forEach(m => { state.models[m.id] = m; });
      }
    } catch (err) {
      console.warn("Could not fetch models:", err);
    }
  }

  async function fetchAllSamples() {
    try {
      const res = await fetch("/api/sample-files");
      if (res.ok) {
        const data = await res.json();
        state.allSamples = data.samples || [];
        renderSamplesMatrix();
      }
    } catch (err) {
      console.warn("Could not fetch sample files:", err);
    }
  }

  // 2. Mode Switching (Upload vs Live Generator)
  tabUploadBtn.addEventListener("click", () => {
    state.mode = "upload";
    tabUploadBtn.classList.add("active");
    tabGenerateBtn.classList.remove("active");
    viewUploadContainer.classList.remove("hidden");
    viewGenerateContainer.classList.add("hidden");
  });

  tabGenerateBtn.addEventListener("click", () => {
    state.mode = "generate";
    tabGenerateBtn.classList.add("active");
    tabUploadBtn.classList.remove("active");
    viewGenerateContainer.classList.remove("hidden");
    viewUploadContainer.classList.add("hidden");
  });

  // 3. Task Switching (Multiclass vs Binary)
  taskMultiBtn.addEventListener("click", () => {
    state.task = "multiclass";
    taskMultiBtn.classList.add("active");
    taskBinaryBtn.classList.remove("active");
    if (state.selectedFile && state.mode === "upload") runInference();
  });

  taskBinaryBtn.addEventListener("click", () => {
    state.task = "binary";
    taskBinaryBtn.classList.add("active");
    taskMultiBtn.classList.remove("active");
    if (state.selectedFile && state.mode === "upload") runInference();
  });

  // 4. Architecture Selection
  archChips.forEach(chip => {
    chip.addEventListener("click", () => {
      archChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      state.architecture = chip.dataset.arch;
      if (state.selectedFile && state.mode === "upload") runInference();
    });
  });

  // 5. Size Selection
  sizePills.forEach(pill => {
    pill.addEventListener("click", () => {
      sizePills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      state.size = pill.dataset.size;
      if (state.selectedFile && state.mode === "upload") runInference();
    });
  });

  // 6. Samples Matrix & Filter by Size (1kb - 512kb)
  sampleSizeFilter.addEventListener("click", (e) => {
    if (e.target.classList.contains("size-filter-btn")) {
      document.querySelectorAll(".size-filter-btn").forEach(b => b.classList.remove("active"));
      e.target.classList.add("active");
      state.activeFilterSize = e.target.dataset.filter;
      renderSamplesMatrix();
    }
  });

  // Algorithm filter buttons
  const presetAlgoFilter = document.getElementById("preset-algo-filter");
  presetAlgoFilter.addEventListener("click", (e) => {
    if (e.target.classList.contains("algo-filter-btn")) {
      document.querySelectorAll(".algo-filter-btn").forEach(b => b.classList.remove("active"));
      e.target.classList.add("active");
      state.presetAlgorithm = e.target.dataset.alg;
      renderSamplesMatrix();
    }
  });

  // Initialize state for algorithm filter
  state.presetAlgorithm = "all";

  function renderSamplesMatrix() {
    samplesMatrix.innerHTML = "";
    let filtered = state.allSamples;

    // Filter by size
    if (state.activeFilterSize !== "all") {
      filtered = filtered.filter(s => s.size_label === state.activeFilterSize);
    }
    // Filter by algorithm (AES / 3DES)
    if (state.presetAlgorithm && state.presetAlgorithm !== "all") {
      filtered = filtered.filter(s => s.algorithm === state.presetAlgorithm);
    }

    if (filtered.length === 0) {
      samplesMatrix.innerHTML = `<div style="grid-column: 1/-1; font-size: 0.8rem; color: var(--text-muted); padding: 0.5rem;">No samples found for current filter</div>`;
      return;
    }

    filtered.forEach(s => {
      const item = document.createElement("button");
      item.className = "sample-matrix-item";
      item.innerHTML = `
        <span class="matrix-cipher">${s.algorithm}</span>
        <span class="matrix-size">${s.size_label.toUpperCase()} (${(s.byte_size / 1024).toFixed(1)} KB)</span>
      `;
      item.addEventListener("click", () => loadSampleFromCatalog(s));
      samplesMatrix.appendChild(item);
    });
  }

  async function loadSampleFromCatalog(s) {
    try {
      emptyState.classList.add("hidden");
      loadingState.classList.remove("hidden");
      resultsContainer.classList.add("hidden");
      loadingArchText.textContent = `Loading ${s.algorithm} (${s.size_label.toUpperCase()}) and running ${state.architecture.toUpperCase()} classification...`;

      const res = await fetch(`/api/sample-files/raw?path=${encodeURIComponent(s.relative_path)}`);
      if (!res.ok) throw new Error("Could not load sample file");
      const blob = await res.blob();
      const descriptiveName = `${s.algorithm}_${s.size_label}_${s.raw_filename || s.filename}`;
      const file = new File([blob], descriptiveName, { type: "application/octet-stream" });
      state.currentHint = s.algorithm;
      handleFileSelection(file);

      // Match size bucket in UI
      const matchPill = document.querySelector(`#size-pill-group .size-pill[data-size="${s.size_label}"]`);
      if (matchPill) {
        sizePills.forEach(p => p.classList.remove("active"));
        matchPill.classList.add("active");
        state.size = s.size_label;
      }

      await runInference(s.algorithm);
    } catch (err) {
      alert(`Error loading sample: ${err.message}`);
      loadingState.classList.add("hidden");
      emptyState.classList.remove("hidden");
    }
  }

  // 7. File Selection / Drag & Drop
  browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => { dropzone.classList.remove("dragover"); });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      state.currentHint = null;
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      state.currentHint = null;
      handleFileSelection(e.target.files[0]);
    }
  });

  function handleFileSelection(file) {
    state.selectedFile = file;
    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = `${file.size.toLocaleString()} bytes`;
    selectedFilePanel.classList.remove("hidden");

    // Preview first 32 bytes in hex
    const reader = new FileReader();
    reader.onload = (e) => {
      const buf = new Uint8Array(e.target.result.slice(0, 32));
      let hexStr = "";
      for (let i = 0; i < buf.length; i++) {
        hexStr += buf[i].toString(16).padStart(2, "0") + " ";
        if ((i + 1) % 8 === 0) hexStr += " ";
      }
      hexPreviewBody.textContent = hexStr.trim() + (file.size > 32 ? " ..." : "");
    };
    reader.readAsArrayBuffer(file);
  }

  predictBtn.addEventListener("click", () => runInference());

  // 8. Live Ciphertext Generator Handlers
  genCipherChips.forEach(chip => {
    chip.addEventListener("click", () => {
      genCipherChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      genSelectedCipher = chip.dataset.cipher;
    });
  });

  genSizePills.forEach(pill => {
    pill.addEventListener("click", () => {
      genSizePills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      genSelectedSize = pill.dataset.size;
    });
  });

  btnUuidFill.addEventListener("click", () => {
    // Generate sample pseudo-UUID text
    const sampleUuid = Array.from({ length: 6 }, () => 
      crypto.randomUUID().replace(/-/g, '')
    ).join(" ");
    genPlaintextInput.value = `HIGH_ENTROPY_UUID_PLAINTEXT: ${sampleUuid}`;
  });

  btnGenerateEncrypt.addEventListener("click", async () => {
    if (state.isAnalyzing) return;
    state.isAnalyzing = true;

    emptyState.classList.add("hidden");
    resultsContainer.classList.add("hidden");
    loadingState.classList.remove("hidden");
    loadingArchText.textContent = `Encrypting with ${genSelectedCipher} in ECB mode and evaluating ${state.architecture.toUpperCase()}...`;

    try {
      const payload = {
        algorithm: genSelectedCipher,
        size: genSelectedSize,
        text: genPlaintextInput.value.trim() || null,
        task: state.task,
        architecture: state.architecture
      };

      const res = await fetch("/api/generate-and-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: json_stringify_safe(payload)
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || errJson.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();
      
      // Update readout panel
      generatedReadoutPanel.classList.remove("hidden");
      genReadoutBytes.textContent = `${data.ground_truth.ciphertext_bytes.toLocaleString()} bytes`;
      genReadoutSha.textContent = data.ground_truth.sha256.substring(0, 24) + "...";
      genReadoutHex.textContent = data.ground_truth.hex_preview_32b;

      // Enable download of the newly generated ciphertext
      if (data.ground_truth) {
        state.generatedCiphertextBytes = data.ground_truth;
      }

      renderResults(data, data.ground_truth.algorithm);
    } catch (err) {
      console.error("Generator Error:", err);
      alert(`Encryption & Prediction failed: ${err.message}`);
      emptyState.classList.remove("hidden");
    } finally {
      loadingState.classList.add("hidden");
      state.isAnalyzing = false;
    }
  });

  btnDownloadGenerated.addEventListener("click", () => {
    if (!state.generatedCiphertextBytes) return;
    // Generate and download a .bin file
    const dummyBlob = new Blob([new Uint8Array(state.generatedCiphertextBytes.ciphertext_bytes)], { type: "application/octet-stream" });
    const url = URL.createObjectURL(dummyBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `generated_${genSelectedCipher}_${genSelectedSize}.bin`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // 9. Predict Execution (Upload Mode)
  async function runInference(knownGroundTruth = null) {
    if (!state.selectedFile || state.isAnalyzing) return;
    state.isAnalyzing = true;

    emptyState.classList.add("hidden");
    resultsContainer.classList.add("hidden");
    loadingState.classList.remove("hidden");
    loadingArchText.textContent = `Running ${state.architecture.toUpperCase()} inference on ${state.selectedFile.name}...`;

    try {
      const formData = new FormData();
      formData.append("file", state.selectedFile);
      formData.append("task", state.task);
      formData.append("architecture", state.architecture);
      if (state.size !== "auto") formData.append("size", state.size);
      const effectiveHint = knownGroundTruth || state.currentHint;
      if (effectiveHint) formData.append("hint", effectiveHint);

      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || errJson.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();
      renderResults(data, effectiveHint);
    } catch (err) {
      console.error("Inference Error:", err);
      alert(`Inference failed: ${err.message}`);
      emptyState.classList.remove("hidden");
    } finally {
      loadingState.classList.add("hidden");
      state.isAnalyzing = false;
    }
  }

  // 10. Render Results
  function renderResults(data, groundTruth = null) {
    resultsContainer.classList.remove("hidden");

    // Header pills
    resultTaskPill.textContent = `Task: ${data.task === "binary" ? "Binary (AES vs 3DES)" : "5-Cipher Multiclass"}`;
    resultArchPill.textContent = `Arch: ${data.architecture_name || state.architecture.toUpperCase()}`;
    resultSizePill.textContent = `Bucket: ${data.selected_model_size.toUpperCase()}`;

    // Ground Truth Pill
    const actualAlgo = groundTruth || state.currentHint || (data.input_info && data.input_info.detected_hint) || (data.ground_truth ? data.ground_truth.algorithm : null);
    if (actualAlgo) {
      groundTruthPill.classList.remove("hidden");
      const isMatch = actualAlgo.toUpperCase() === data.predicted_cipher.toUpperCase();
      groundTruthPill.textContent = `Actual: ${actualAlgo} ${isMatch ? "✓ MATCH" : "≠ DIFFERENT"}`;
      groundTruthPill.style.background = isMatch ? "rgba(16, 185, 129, 0.2)" : "rgba(245, 158, 11, 0.2)";
      groundTruthPill.style.color = isMatch ? "var(--accent-emerald)" : "var(--accent-amber)";
    } else {
      groundTruthPill.classList.add("hidden");
    }

    // Top Prediction
    predictedCipherName.textContent = data.predicted_cipher;
    const confPercent = (data.confidence * 100).toFixed(1);
    predConfidence.textContent = `${confPercent}%`;


    // Multi-Architecture Comparison
    if (data.architecture_comparison) {
      archCompareCard.classList.remove("hidden");
      renderArchitectureComparison(data.architecture_comparison);
    } else {
      archCompareCard.classList.add("hidden");
    }

    // Softmax Bars
    const chance = data.task === "binary" ? 0.50 : 0.20;
    renderProbabilityBars(data.probabilities, data.predicted_cipher, chance);


  }

  function renderArchitectureComparison(comparison) {
    archComparisonGrid.innerHTML = "";
    Object.entries(comparison).forEach(([key, info]) => {
      const card = document.createElement("div");
      card.className = "arch-compare-item";
      card.innerHTML = `
        <span class="arch-compare-name">${info.name}</span>
        <span class="arch-compare-cipher">${info.predicted_cipher}</span>
        <span class="arch-compare-conf">Conf: ${(info.confidence * 100).toFixed(1)}%</span>
      `;
      archComparisonGrid.appendChild(card);
    });
  }

  function renderProbabilityBars(probs, predicted, chance) {
    probBarsContainer.innerHTML = "";
    Object.entries(probs).forEach(([cipher, prob]) => {
      const pct = (prob * 100).toFixed(1);
      const isTop = cipher === predicted;

      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-label-row">
          <span class="prob-cipher-title" style="color: ${isTop ? 'var(--accent-cyan)' : 'var(--text-primary)'}">
            ${cipher} ${isTop ? '★' : ''}
          </span>
          <span class="prob-val-text">${pct}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${pct}%; background: ${isTop ? 'var(--grad-primary)' : 'rgba(255, 255, 255, 0.18)'}"></div>
        </div>
      `;
      probBarsContainer.appendChild(row);
    });
  }

  // NIST 49-feature rendering removed from UI (backend computation still active)

  // Theme Controls
  themeToggle.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    document.documentElement.setAttribute("data-theme", cur === "light" ? "dark" : "light");
  });

  function json_stringify_safe(obj) {
    return JSON.stringify(obj);
  }

  init();
});
