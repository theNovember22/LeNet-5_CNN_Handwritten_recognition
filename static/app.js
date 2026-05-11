const input = document.querySelector("#imageInput");
const button = document.querySelector("#predictButton");
const preview = document.querySelector("#preview");
const emptyPreview = document.querySelector("#emptyPreview");
const processedPreview = document.querySelector("#processedPreview");
const emptyProcessedPreview = document.querySelector("#emptyProcessedPreview");
const statusText = document.querySelector("#status");
const digitText = document.querySelector("#digit");
const confidenceText = document.querySelector("#confidence");
const probabilities = document.querySelector("#probabilities");

let selectedFile = null;

function renderProbabilities(values = []) {
  probabilities.innerHTML = "";
  values.forEach((value, digit) => {
    const row = document.createElement("div");
    row.className = "probability-row";
    row.innerHTML = `
      <span>${digit}</span>
      <div class="bar"><i style="width: ${value}%"></i></div>
      <strong>${value.toFixed(2)}%</strong>
    `;
    probabilities.appendChild(row);
  });
}

input.addEventListener("change", () => {
  selectedFile = input.files[0];
  button.disabled = !selectedFile;
  statusText.textContent = "";
  digitText.textContent = "-";
  confidenceText.textContent = "Confidence: -";
  processedPreview.removeAttribute("src");
  emptyProcessedPreview.style.display = "block";
  renderProbabilities([]);

  if (!selectedFile) {
    preview.removeAttribute("src");
    emptyPreview.style.display = "block";
    return;
  }

  preview.src = URL.createObjectURL(selectedFile);
  emptyPreview.style.display = "none";
});

button.addEventListener("click", async () => {
  if (!selectedFile) {
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);

  button.disabled = true;
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
    confidenceText.textContent = `Confidence: ${result.confidence.toFixed(2)}%`;
    processedPreview.src = result.processedImage;
    emptyProcessedPreview.style.display = "none";
    renderProbabilities(result.probabilities);
    statusText.textContent = "";
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
