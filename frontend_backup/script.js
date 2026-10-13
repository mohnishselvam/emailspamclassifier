
const API_URL = "http://127.0.0.1:5000/predict";

const messageInput = document.getElementById("messageInput");
const scanButton = document.getElementById("scanButton");
const scanButtonText = document.getElementById("scanButtonText");
const clearButton = document.getElementById("clearButton");
const charCount = document.getElementById("charCount");

const result = document.getElementById("result");
const resultTitle = document.getElementById("resultTitle");
const resultIcon = document.getElementById("resultIcon");
const resultDescription = document.getElementById("resultDescription");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");

const errorMessage = document.getElementById("errorMessage");
const keywordSection = document.getElementById("keywordSection");
const keywordList = document.getElementById("keywordList");

function hideFeedback() {
    result.classList.add("hidden");
    errorMessage.classList.add("hidden");
    keywordSection.classList.add("hidden");
    keywordList.replaceChildren();
}

function updateCharacterCount() {
    charCount.textContent =
        `${messageInput.value.length} / 10000 characters`;
}

function renderKeywords(keywords) {
    keywordList.replaceChildren();

    if (!Array.isArray(keywords) || keywords.length === 0) {
        const emptyMessage = document.createElement("p");
        emptyMessage.className = "keyword-empty";
        emptyMessage.textContent =
            "No significant in-vocabulary keywords were found for this message.";
        keywordList.appendChild(emptyMessage);
        keywordSection.classList.remove("hidden");
        return;
    }

    keywords.forEach((item) => {
        if (
            !item ||
            typeof item.word !== "string" ||
            !Number.isFinite(Number(item.score))
        ) {
            return;
        }

        const score = Number(item.score);
        const leansSpam = score > 0;

        const row = document.createElement("div");
        row.className = `keyword-row ${leansSpam ? "keyword-spam" : "keyword-ham"}`;

        const wordInfo = document.createElement("div");
        wordInfo.className = "keyword-word-info";

        const word = document.createElement("span");
        word.className = "keyword-word";
        word.textContent = item.word;

        const direction = document.createElement("span");
        direction.className = "keyword-direction";
        direction.textContent = leansSpam
            ? "Leans toward spam"
            : score < 0
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
        bar.className = `keyword-bar ${leansSpam ? "spam-bar" : "ham-bar"}`;

        // Use relative magnitudes to compare terms within this message.
        const maxMagnitude = Math.max(
            ...keywords
                .map((keyword) => Math.abs(Number(keyword.score)))
                .filter(Number.isFinite),
            0.00001
        );

        bar.style.width =
            `${Math.max(3, Math.abs(score) / maxMagnitude * 100)}%`;

        track.appendChild(bar);
        row.append(wordInfo, scoreLabel);
        row.appendChild(track);
        keywordList.appendChild(row);
    });

    keywordSection.classList.remove("hidden");
}

messageInput.addEventListener("input", updateCharacterCount);

clearButton.addEventListener("click", () => {
    messageInput.value = "";
    updateCharacterCount();
    hideFeedback();
    messageInput.focus();
});

messageInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        scanButton.click();
    }
});

scanButton.addEventListener("click", async () => {
    const message = messageInput.value.trim();

    hideFeedback();

    if (!message) {
        errorMessage.textContent =
            "Please enter a message before starting the analysis.";
        errorMessage.classList.remove("hidden");
        messageInput.focus();
        return;
    }

    scanButton.disabled = true;
    scanButtonText.textContent = "Analyzing message...";

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Unable to analyze this message."
            );
        }

        if (
            !["SPAM", "HAM"].includes(data.prediction) ||
            !Number.isFinite(Number(data.confidence))
        ) {
            throw new Error("The server returned an invalid prediction.");
        }

        const isSpam = data.prediction === "SPAM";
        const confidence = Number(data.confidence);

        if (confidence < 0 || confidence > 100) {
            throw new Error("The server returned an invalid confidence score.");
        }

        result.classList.remove("hidden", "spam", "ham");
        result.classList.add(isSpam ? "spam" : "ham");

        resultTitle.textContent = isSpam
            ? "Potential Spam Detected"
            : "Likely Legitimate Message";

        resultIcon.textContent = isSpam ? "!" : "✓";

        resultDescription.textContent = isSpam
            ? "This message has patterns associated with spam. Avoid clicking links or sharing personal information unless you verify the sender."
            : "The model classified this message as legitimate. Still verify unexpected links, payment requests, and sensitive information requests.";

        confidenceValue.textContent = `${confidence.toFixed(2)}%`;
        confidenceBar.style.width = `${confidence}%`;

        const confidenceTrack = document.querySelector(".confidence-track");
        if (confidenceTrack) {
            confidenceTrack.setAttribute("aria-valuenow", String(confidence));
        }

        renderKeywords(data.keywords);

        result.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });

    } catch (error) {
        errorMessage.textContent =
            error instanceof TypeError
                ? "Could not connect to the backend. Make sure Flask is running at http://127.0.0.1:5000."
                : error.message;

        errorMessage.classList.remove("hidden");

    } finally {
        scanButton.disabled = false;
        scanButtonText.textContent = "Analyze Message";
    }
});

updateCharacterCount();
