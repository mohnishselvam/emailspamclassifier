
const API_BASE_URL = "http://127.0.0.1:5000";
const HISTORY_PAGE_SIZE = 10;

// Scanner elements
const messageInput = document.getElementById("messageInput");
const scanButton = document.getElementById("scanButton");
const scanButtonText = document.getElementById("scanButtonText");
const clearButton = document.getElementById("clearButton");
const charCount = document.getElementById("charCount");

// Prediction result elements
const result = document.getElementById("result");
const resultTitle = document.getElementById("resultTitle");
const resultIcon = document.getElementById("resultIcon");
const resultDescription = document.getElementById("resultDescription");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceTrack = document.querySelector(".confidence-track");

// Keyword analysis elements
const keywordSection = document.getElementById("keywordSection");
const keywordList = document.getElementById("keywordList");

// Feedback elements
const errorMessage = document.getElementById("errorMessage");

// Analytics elements
const totalScans = document.getElementById("totalScans");
const spamCount = document.getElementById("spamCount");
const hamCount = document.getElementById("hamCount");
const averageConfidence = document.getElementById("averageConfidence");
const analyticsStatus = document.getElementById("analyticsStatus");
const refreshAnalyticsButton = document.getElementById(
    "refreshAnalyticsButton"
);

// History elements
const historyBody = document.getElementById("historyBody");
const historyStatus = document.getElementById("historyStatus");
const refreshHistoryButton = document.getElementById(
    "refreshHistoryButton"
);
const previousPageButton = document.getElementById("previousPageButton");
const nextPageButton = document.getElementById("nextPageButton");
const pageIndicator = document.getElementById("pageIndicator");

let historyRecords = [];
let currentHistoryPage = 1;


// --------------------------------------------------
// General helpers
// --------------------------------------------------

function setText(element, value) {
    if (element) {
        element.textContent = value;
    }
}

function hideFeedback() {
    result.classList.add("hidden");
    errorMessage.classList.add("hidden");
    keywordSection.classList.add("hidden");
    keywordList.replaceChildren();
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}

function updateCharacterCount() {
    setText(
        charCount,
        `${messageInput.value.length} / 10000 characters`
    );
}

async function fetchJSON(endpoint, options = {}) {
    let response;

    try {
        response = await fetch(`${API_BASE_URL}${endpoint}`, options);
    } catch {
        throw new Error(
            "Cannot connect to Flask. Make sure the backend is running at " +
            `${API_BASE_URL}.`
        );
    }

    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error(
            "The backend returned an unreadable response. Check the Flask terminal."
        );
    }

    if (!response.ok) {
        throw new Error(
            data.error ||
            data.message ||
            `Request failed with HTTP status ${response.status}.`
        );
    }

    return data;
}


// --------------------------------------------------
// Message scanner
// --------------------------------------------------

messageInput.addEventListener("input", updateCharacterCount);

clearButton.addEventListener("click", () => {
    messageInput.value = "";
    updateCharacterCount();
    hideFeedback();
    messageInput.focus();
});

messageInput.addEventListener("keydown", (event) => {
    if (
        (event.ctrlKey || event.metaKey) &&
        event.key === "Enter"
    ) {
        scanButton.click();
    }
});

scanButton.addEventListener("click", async () => {
    const message = messageInput.value.trim();

    hideFeedback();

    if (!message) {
        showError("Please enter a message before starting the analysis.");
        messageInput.focus();
        return;
    }

    scanButton.disabled = true;
    scanButtonText.textContent = "Analyzing message...";

    try {
        const data = await fetchJSON("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message })
        });

        const prediction = String(data.prediction || "").toUpperCase();
        const confidence = Number(data.confidence);

        if (!["SPAM", "HAM"].includes(prediction)) {
            throw new Error("The backend returned an invalid prediction.");
        }

        if (
            !Number.isFinite(confidence) ||
            confidence < 0 ||
            confidence > 100
        ) {
            throw new Error("The backend returned an invalid confidence score.");
        }

        renderPrediction(prediction, confidence);
        renderKeywords(data.keywords);

        // Refresh saved data after a successful prediction.
        await Promise.allSettled([
            loadAnalytics(),
            loadHistory()
        ]);

        result.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
    } catch (error) {
        showError(error.message || "Unable to analyze this message.");
    } finally {
        scanButton.disabled = false;
        scanButtonText.textContent = "Analyze message";
    }
});

function renderPrediction(prediction, confidence) {
    const isSpam = prediction === "SPAM";

    result.classList.remove("hidden", "spam", "ham");
    result.classList.add(isSpam ? "spam" : "ham");

    resultTitle.textContent = isSpam
        ? "Potential Spam Detected"
        : "Likely Legitimate Message";

    resultIcon.textContent = isSpam ? "!" : "✓";

    resultDescription.textContent = isSpam
        ? "This message has patterns associated with spam. Avoid clicking links or sharing personal information unless you verify the sender."
        : "The model classified this message as legitimate. Still verify unexpected links, payment requests, and requests for sensitive information.";

    confidenceValue.textContent = `${confidence.toFixed(2)}%`;
    confidenceBar.style.width = `${confidence}%`;

    if (confidenceTrack) {
        confidenceTrack.setAttribute(
            "aria-valuenow",
            String(confidence)
        );
    }
}


// --------------------------------------------------
// Keyword contribution analysis
// --------------------------------------------------

function renderKeywords(keywords) {
    keywordList.replaceChildren();

    if (!Array.isArray(keywords) || keywords.length === 0) {
        const emptyMessage = document.createElement("p");
        emptyMessage.className = "keyword-empty";
        emptyMessage.textContent =
            "No significant in-vocabulary keywords were returned for this message.";

        keywordList.appendChild(emptyMessage);
        keywordSection.classList.remove("hidden");
        return;
    }

    const validKeywords = keywords.filter((item) => {
        return (
            item &&
            typeof item.word === "string" &&
            Number.isFinite(Number(item.score))
        );
    });

    if (validKeywords.length === 0) {
        const emptyMessage = document.createElement("p");
        emptyMessage.className = "keyword-empty";
        emptyMessage.textContent =
            "No usable keyword contribution data was returned.";

        keywordList.appendChild(emptyMessage);
        keywordSection.classList.remove("hidden");
        return;
    }

    const maxMagnitude = Math.max(
        ...validKeywords.map((item) => Math.abs(Number(item.score))),
        0.00001
    );

    validKeywords.forEach((item) => {
        const score = Number(item.score);
        const leansSpam = score > 0;
        const leansHam = score < 0;

        const row = document.createElement("div");
        row.className = `keyword-row ${
            leansSpam ? "keyword-spam" : "keyword-ham"
        }`;

        const wordInfo = document.createElement("div");
        wordInfo.className = "keyword-word-info";

        const word = document.createElement("span");
        word.className = "keyword-word";
        word.textContent = item.word;

        const direction = document.createElement("span");
        direction.className = "keyword-direction";
        direction.textContent = leansSpam
            ? "Leans toward spam"
            : leansHam
                ? "Leans toward legitimate"
                : "Neutral contribution";

        wordInfo.append(word, direction);

        const scoreLabel = document.createElement("span");
        scoreLabel.className = "keyword-score";
        scoreLabel.textContent =
            `${score > 0 ? "+" : ""}${score.toFixed(3)}`;

        const track = document.createElement("div");
        track.className = "keyword-track";

        const bar = document.createElement("div");
        bar.className = `keyword-bar ${
            leansSpam ? "spam-bar" : "ham-bar"
        }`;

        const width = Math.max(
            3,
            (Math.abs(score) / maxMagnitude) * 100
        );

        bar.style.width = `${width}%`;

        track.appendChild(bar);
        row.append(wordInfo, scoreLabel, track);
        keywordList.appendChild(row);
    });

    keywordSection.classList.remove("hidden");
}


// --------------------------------------------------
// Analytics dashboard
// --------------------------------------------------

async function loadAnalytics() {
    analyticsStatus.textContent = "Updating analytics...";

    try {
        const response = await fetchJSON("/analytics");
        const stats = response.analytics || response;

        setText(totalScans, Number(stats.total_scans ?? 0).toLocaleString());
        setText(spamCount, Number(stats.spam_count ?? 0).toLocaleString());
        setText(hamCount, Number(stats.ham_count ?? 0).toLocaleString());

        const confidence = Number(stats.average_confidence ?? 0);

        setText(
            averageConfidence,
            `${confidence.toFixed(2)}%`
        );

        analyticsStatus.textContent = "Analytics updated successfully.";
    } catch (error) {
        analyticsStatus.textContent = error.message;
    }
}

refreshAnalyticsButton.addEventListener("click", loadAnalytics);


// --------------------------------------------------
// Prediction history
// --------------------------------------------------

async function loadHistory() {
    historyStatus.textContent = "Loading prediction history...";

    try {
        const response = await fetchJSON("/history");

        // Support the common response formats used by Flask APIs.
        const records = Array.isArray(response)
            ? response
            : Array.isArray(response.history)
                ? response.history
                : Array.isArray(response.records)
                    ? response.records
                    : Array.isArray(response.predictions)
                        ? response.predictions
                        : null;

        if (!records) {
            throw new Error(
                "The history response did not contain a recognizable record list."
            );
        }

        historyRecords = records;
        currentHistoryPage = 1;

        renderHistory();

        historyStatus.textContent = historyRecords.length
            ? `${historyRecords.length} saved prediction(s) loaded.`
            : "No predictions saved yet. Analyze a message to start your history.";
    } catch (error) {
        historyStatus.textContent = error.message;
        historyBody.replaceChildren();

        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 4;
        cell.className = "empty-cell";
        cell.textContent = "Unable to load prediction history.";

        row.appendChild(cell);
        historyBody.appendChild(row);

        updatePagination(0);
    }
}

function renderHistory() {
    historyBody.replaceChildren();

    const totalPages = Math.max(
        1,
        Math.ceil(historyRecords.length / HISTORY_PAGE_SIZE)
    );

    currentHistoryPage = Math.min(currentHistoryPage, totalPages);

    const startIndex = (currentHistoryPage - 1) * HISTORY_PAGE_SIZE;
    const visibleRecords = historyRecords.slice(
        startIndex,
        startIndex + HISTORY_PAGE_SIZE
    );

    if (visibleRecords.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 4;
        cell.className = "empty-cell";
        cell.textContent = "No prediction history available.";

        row.appendChild(cell);
        historyBody.appendChild(row);

        updatePagination(0);
        return;
    }

    visibleRecords.forEach((record) => {
        const row = document.createElement("tr");

        // Message
        const messageCell = document.createElement("td");
        messageCell.className = "message-cell";
        messageCell.textContent = String(record.message ?? "—");
        messageCell.title = String(record.message ?? "");

        // Prediction
        const predictionCell = document.createElement("td");
        const predictionPill = document.createElement("span");
        const prediction = String(record.prediction ?? "").toUpperCase();
        const isSpam = prediction === "SPAM";

        predictionPill.className = `result-pill ${
            isSpam ? "spam-pill" : "ham-pill"
        }`;
        predictionPill.textContent = prediction || "UNKNOWN";
        predictionCell.appendChild(predictionPill);

        // Confidence
        const confidenceCell = document.createElement("td");
        const confidence = Number(record.confidence);

        confidenceCell.textContent = Number.isFinite(confidence)
            ? `${confidence.toFixed(2)}%`
            : "—";

        // Timestamp
        const dateCell = document.createElement("td");
        dateCell.textContent = formatDate(record.created_at);

        row.append(
            messageCell,
            predictionCell,
            confidenceCell,
            dateCell
        );

        historyBody.appendChild(row);
    });

    updatePagination(historyRecords.length);
}

function formatDate(value) {
    if (!value) {
        return "—";
    }

    const date = new Date(
        typeof value === "string" && !value.includes("T")
            ? value.replace(" ", "T")
            : value
    );

    if (Number.isNaN(date.getTime())) {
        return String(value);
    }

    return date.toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit"
    });
}

function updatePagination(totalRecords) {
    const totalPages = Math.max(
        1,
        Math.ceil(totalRecords / HISTORY_PAGE_SIZE)
    );

    pageIndicator.textContent =
        `Page ${currentHistoryPage} of ${totalPages}`;

    previousPageButton.disabled = currentHistoryPage <= 1;
    nextPageButton.disabled = currentHistoryPage >= totalPages;
}

previousPageButton.addEventListener("click", () => {
    if (currentHistoryPage > 1) {
        currentHistoryPage -= 1;
        renderHistory();
    }
});

nextPageButton.addEventListener("click", () => {
    const totalPages = Math.max(
        1,
        Math.ceil(historyRecords.length / HISTORY_PAGE_SIZE)
    );

    if (currentHistoryPage < totalPages) {
        currentHistoryPage += 1;
        renderHistory();
    }
});

refreshHistoryButton.addEventListener("click", loadHistory);


// --------------------------------------------------
// Dashboard initialization
// --------------------------------------------------

updateCharacterCount();

loadAnalytics();
loadHistory();
