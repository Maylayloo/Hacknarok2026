import { FilesetResolver, HandLandmarker } from "./libs/vision_bundle.js";

// ==========================================
// KONFIGURACJA
// ==========================================
const GESTURE_CONFIG = {
    BUFFER_SIZE: 35,         // Model oczekuje 35 klatek
    INFERENCE_INTERVAL: 1,   // Odpalamy model co klatkę (gdy bufor jest pełny)
    CONFIDENCE_THRESHOLD: 0.85, // Minimalna pewność modelu (85%)
    ACTION_COOLDOWN_MS: 1500 // OPÓŹNIENIE: 1.5 sekundy przerwy po wykryciu gestu
};

// Pamiętaj, aby upewnić się, że indeksy odpowiadają Twojemu modelowi!
const CLASS_NAMES = {0: "PALM_OPEN", 1: "SWIPE_LEFT"}; 

// ==========================================
// ZMIENNE GLOBALNE
// ==========================================
let handLandmarker = null;
let webcam = null;
let lastVideoTime = -1;
let keypointsBuffer = [];
let framesProcessed = 0;
let isModelRunning = false;
let onnxSession = null;
let lastActionTimestamp = 0;


function clearGestureBuffer() {
    keypointsBuffer = [];
    console.log("🧹 Bufor gestów wyczyszczony.");
}

// Pozwala czyścić bufor z innej części wtyczki (np. z popup.js)
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "CLEAR_BUFFER") {
        clearGestureBuffer();
        if (message.resetCooldown) {
            lastActionTimestamp = 0; 
        }
        sendResponse({ status: "Bufor wyczyszczony" });
    }
});

// ==========================================
// INICJALIZACJA WSZYSTKIEGO
// ==========================================
async function initAll() {
    try {
        console.log("🚀 Start inicjalizacji...");

        // 1. Konfiguracja ONNX (Zabezpieczenia dla Chrome Extension)
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = false;
        ort.env.wasm.wasmPaths = chrome.runtime.getURL("libs/");
        ort.env.wasm.simd = false;

        console.log("📦 Pobieranie zoptymalizowanego modelu ORT...");
        const modelUrl = chrome.runtime.getURL("models/gesture_model.ort");
        const modelResponse = await fetch(modelUrl);

        if (!modelResponse.ok) {
            throw new Error(`Nie udało się pobrać modelu ONNX: ${modelResponse.statusText}`);
        }

        const modelBuffer = await modelResponse.arrayBuffer();

        console.log("🧠 Tworzenie sesji ONNX...");
        onnxSession = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        console.log("✅ Model ONNX załadowany pomyślnie!");

        // 2. Konfiguracja MediaPipe
        console.log("📸 Inicjalizacja MediaPipe Hand Landmarker...");
        const wasmPath = chrome.runtime.getURL("libs/");
        const vision = await FilesetResolver.forVisionTasks(wasmPath);
        const taskPath = chrome.runtime.getURL("models/hand_landmarker.task");

        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: taskPath,
                delegate: "CPU" // Tryb CPU jest najbardziej stabilny w tle
            },
            runningMode: "VIDEO",
            numHands: 1
        });
        console.log("✅ MediaPipe gotowy!");

        // 3. Start Kamery
        await startWebcam();

    } catch (err) {
        console.error("❌ Krytyczny błąd w initAll:", err);
    }
}

async function startWebcam() {
    webcam = document.getElementById("webcam");
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        webcam.srcObject = stream;
        webcam.onloadedmetadata = () => {
            webcam.play();
            predictWebcam();
        };
        console.log("🎥 Kamera w Offscreen gotowa i działa.");
    } catch (err) {
        console.error("Offscreen Camera Error:", err.name, err.message);
    }
}

async function predictWebcam() {
    let startTimeMs = performance.now();
    if (webcam.currentTime !== lastVideoTime) {
        lastVideoTime = webcam.currentTime;

        const results = handLandmarker.detectForVideo(webcam, startTimeMs);

        if (results.landmarks && results.landmarks.length > 0) {
            handleNewFrame(results.landmarks);
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

async function handleNewFrame(mediaPipeLandmarks) {
    if (Date.now() - lastActionTimestamp < GESTURE_CONFIG.ACTION_COOLDOWN_MS) {
        return; 
    }

    framesProcessed++;
    const frameKeypoints = mediaPipeLandmarks[0].map(lm => [lm.x, lm.y]);
    keypointsBuffer.push(frameKeypoints);

    if (keypointsBuffer.length > GESTURE_CONFIG.BUFFER_SIZE) {
        keypointsBuffer.shift();
    }
    
    if (
        keypointsBuffer.length === GESTURE_CONFIG.BUFFER_SIZE &&
        framesProcessed % GESTURE_CONFIG.INFERENCE_INTERVAL === 0 &&
        !isModelRunning
    ) {
        isModelRunning = true;
        const bufferCopy = [...keypointsBuffer];

        try {
            await predictAction(bufferCopy);
        } catch (error) {
            console.error("Błąd podczas inferencji ONNX:", error);
        } finally {
            isModelRunning = false;
        }
    }
}

async function predictAction(bufferToProcess) {
    if (!onnxSession) return;

    let flatNormalizedData = [];

    const globalRefX = bufferToProcess[0][0][0];
    const globalRefY = bufferToProcess[0][0][1];

    for (let frame of bufferToProcess) {
        for (let i = 0; i < 21; i++) {
            let px = frame[i] ? frame[i][0] : 0.0;
            let py = frame[i] ? frame[i][1] : 0.0;
            
            // Odejmujemy nadgarstek z pierwszej klatki od KAŻDEGO punktu
            flatNormalizedData.push(px - globalRefX);
            flatNormalizedData.push(py - globalRefY);
        }
    }

    const tensorData = new Float32Array(flatNormalizedData);
    const inputTensor = new ort.Tensor('float32', tensorData, [1, GESTURE_CONFIG.BUFFER_SIZE, 42]);

    const results = await onnxSession.run({ 'input_sequence': inputTensor });
    const outputTensor = results['class_probabilities'].data; // Logity


    let maxLogit = -Infinity;
    for (let i = 0; i < outputTensor.length; i++) {
        if (outputTensor[i] > maxLogit) maxLogit = outputTensor[i];
    }

    let sumExp = 0;
    let probabilities = new Float32Array(outputTensor.length);
    for (let i = 0; i < outputTensor.length; i++) {
        probabilities[i] = Math.exp(outputTensor[i] - maxLogit);
        sumExp += probabilities[i];
    }

    for (let i = 0; i < probabilities.length; i++) {
        probabilities[i] /= sumExp;
    }

    let maxProb = -1;
    let predictedIdx = -1;
    for (let i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
            maxProb = probabilities[i];
            predictedIdx = i;
        }
    }

    if (maxProb >= GESTURE_CONFIG.CONFIDENCE_THRESHOLD) {
        const actionLabel = CLASS_NAMES[predictedIdx] || "UNKNOWN_GESTURE";
        console.log(`🎯 Wykryto: ${actionLabel} (${(maxProb * 100).toFixed(1)}%)`);

        lastActionTimestamp = Date.now();
        
        clearGestureBuffer();

        chrome.runtime.sendMessage({
            type: "GESTURE_COMMAND",
            action: actionLabel
        });
    }
}

initAll();