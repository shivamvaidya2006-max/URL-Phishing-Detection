// ===============================
// GLOBAL STATE
// ===============================
let mlRendered = false;   // ✅ FIX: defined ONCE

// ===============================
// Utility helpers
// ===============================
function escapeHtml(text) {
  const div = document.createElement("div");
  div.innerText = text;
  return div.innerHTML;
}

function getQueryParam(name) {
  return new URLSearchParams(window.location.search).get(name);
}

// ===============================
// INDEX PAGE (Scan button)
// ===============================
function scan() {
  const input = document.getElementById("inputBox").value.trim();
  if (!input) {
    alert("Please enter a URL");
    return;
  }
  window.location.href = `risk.html?value=${encodeURIComponent(input)}`;
}

// ===============================
// RISK PAGE LOGIC
// ===============================
document.addEventListener("DOMContentLoaded", () => {
   if (homeBtn) {
    homeBtn.addEventListener("click", () => {
      window.location.href = "index.html";
    });
  }
  const url = getQueryParam("value");
  if (!url) return;

  // Show URL
  document.getElementById("detectedType").innerText = "URL";
  document.getElementById("inputShown").innerText = url;

  // 1️⃣ JS Analysis (instant)
  const jsResult = runJsAnalysis(url);
  renderJsResult(jsResult);

  // 2️⃣ ML loading (once)
  showMlLoading();

  // 3️⃣ ML call
  callMLModel(url)
    .then(result => {
      hideMlLoading();
      renderMlResult(result, url);
    })
    .catch(err => {
      console.error(err);
      hideMlLoading();
      renderMlError();
    });
});

// ===============================
// JS RULE-BASED ANALYSIS
// ===============================
function runJsAnalysis(url) {
  let score = 10;
  const reasons = [];

  if (!url.startsWith("https://")) {
    score += 20;
    reasons.push("Uses HTTP instead of HTTPS");
  }

  if (/\d+\.\d+\.\d+\.\d+/.test(url)) {
    score += 30;
    reasons.push("Uses IP address instead of domain");
  }

  if (url.length > 75) {
    score += 10;
    reasons.push("URL length is unusually long");
  }

  if (/login|verify|update|secure/i.test(url)) {
    score += 20;
    reasons.push("Contains phishing-related keywords");
  }

  return {
    risk: Math.min(score, 100),
    reasons
  };
}

// ===============================
// RENDER JS RESULT
// ===============================
function renderJsResult(data) {
  document.getElementById("jsResult").innerHTML = `
    <h3>⚠️ Rule-Based (JS) Analysis</h3>
    <p><strong>Risk Score:</strong> ${data.risk}%</p>
    <ul>
      ${data.reasons.map(r => `<li>${escapeHtml(r)}</li>`).join("")}
    </ul>
  `;
}

// ===============================
// ML BACKEND CALL
// ===============================
async function callMLModel(url) {
  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: url })
  });

  if (!response.ok) throw new Error("ML API failed");
  return response.json();
}

// ===============================
// ML UI HANDLING
// ===============================
function showMlLoading() {
  document.getElementById("mlLoading").style.display = "block";
}

function hideMlLoading() {
  document.getElementById("mlLoading").style.display = "none";
}

function renderMlResult(result, url) {
  if (mlRendered) return;   // 🛑 NO flicker
  mlRendered = true;

  const phishing = (result.phishing_probability * 100).toFixed(2);
  const legit = (result.legitimate_probability * 100).toFixed(2);

  const box = document.getElementById("mlResult");
  box.innerHTML = `
    <h3>🤖 ML Prediction: ${result.label}</h3>

    <div class="bar-container">
      <div class="bar-fill"></div>
      <span class="bar-text">${phishing}%</span>
    </div>

    <p><strong>URL:</strong> ${escapeHtml(url)}</p>
    <p>Phishing Probability: ${phishing}%</p>
    <p>Legitimate Probability: ${legit}%</p>
  `;

  // 🎯 Animate bar
  setTimeout(() => {
    document.querySelector(".bar-fill").style.width = phishing + "%";
  }, 100);
}

function renderMlError() {
  document.getElementById("mlResult").innerHTML =
    `<p style="color:red;">ML Prediction Failed</p>`;
}

// ===============================
// BUTTON HOOK
// ===============================
document.addEventListener("DOMContentLoaded", () => {
  const scanBtn = document.getElementById("scanBtn");
  if (scanBtn) scanBtn.addEventListener("click", scan);
});