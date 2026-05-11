const input = document.querySelector("#imageInput");
const button = document.querySelector("#predictButton");
const buttonLabel = button.querySelector(".button-label");
const dropZone = document.querySelector("#dropZone");
const fileName = document.querySelector("#fileName");
const preview = document.querySelector("#preview");
const emptyPreview = document.querySelector("#emptyPreview");
const processedPreview = document.querySelector("#processedPreview");
const emptyProcessedPreview = document.querySelector("#emptyProcessedPreview");
const statusText = document.querySelector("#status");
const digitText = document.querySelector("#digit");
const confidenceText = document.querySelector("#confidence");
const probabilities = document.querySelector("#probabilities");

let selectedFile = null;
const defaultButtonText = buttonLabel.textContent;

function renderProbabilities(values = []) {
  probabilities.innerHTML = "";
  const maxValue = values.length ? Math.max(...values) : null;
  values.forEach((value, digit) => {
    const row = document.createElement("div");
    row.className = value === maxValue ? "probability-row winner" : "probability-row";
    row.innerHTML = `
      <span>${digit}</span>
      <div class="bar"><i style="width: ${value}%"></i></div>
      <strong>${value.toFixed(2)}%</strong>
    `;
    probabilities.appendChild(row);
  });
}

function setSelectedFile(file) {
  selectedFile = file;
  button.disabled = !selectedFile;
  statusText.textContent = "";
  digitText.textContent = "-";
  confidenceText.textContent = "Confidence: -";
  processedPreview.removeAttribute("src");
  emptyProcessedPreview.style.display = "block";
  fileName.textContent = selectedFile ? selectedFile.name : "No file selected";
  renderProbabilities([]);

  if (!selectedFile) {
    preview.removeAttribute("src");
    emptyPreview.style.display = "block";
    return;
  }

  preview.src = URL.createObjectURL(selectedFile);
  emptyPreview.style.display = "none";
}

input.addEventListener("change", () => {
  setSelectedFile(input.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-active");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file || !file.type.startsWith("image/")) {
    statusText.textContent = "Please drop an image file.";
    return;
  }

  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  input.files = dataTransfer.files;
  setSelectedFile(file);
});

button.addEventListener("click", async () => {
  if (!selectedFile) {
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);

  button.disabled = true;
  button.classList.add("loading");
  buttonLabel.textContent = "Predicting";
  statusText.textContent = "Predicting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Prediction failed.");
    }

    digitText.textContent = result.digit;
    digitText.classList.remove("reveal");
    void digitText.offsetWidth;
    digitText.classList.add("reveal");
    confidenceText.textContent = `Confidence: ${result.confidence.toFixed(2)}%`;
    processedPreview.src = result.processedImage;
    emptyProcessedPreview.style.display = "none";
    renderProbabilities(result.probabilities);
    statusText.textContent = "";
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    button.disabled = false;
    button.classList.remove("loading");
    buttonLabel.textContent = defaultButtonText;
  }
});
