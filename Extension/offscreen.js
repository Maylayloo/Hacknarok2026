// 1. BEZPOŚREDNIE IMPORTY
// UWAGA: Importujemy TYLKO MediaPipe. ONNX (ort) jest ładowany z offscreen.html!
import { FilesetResolver, HandLandmarker } from "./libs/vision_bundle.js";

// ==========================================
// KONFIGURACJA
// ==========================================
// ort jest dostępne globalnie
// 1. Blokujemy WebWorkerów (często używają eval do komunikacji)
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = false;
ort.env.wasm.proxy = false;

// WYMUSZENIE backendu WASM (to omija eval w testach WebGL)
ort.env.wasm.wasmPaths = chrome.runtime.getURL("libs/");

const GESTURE_CONFIG = {
    BUFFER_SIZE: 30,
    INFERENCE_INTERVAL: 10,
    CONFIDENCE_THRESHOLD: 0.85
};
const CLASS_NAMES = {0: "PALM_OPEN", 1: "SWIPE_LEFT"};

// Zmienne globalne
let handLandmarker = null;
let webcam = null;
let lastVideoTime = -1;
let keypointsBuffer = [];
let framesProcessed = 0;
let isModelRunning = false;
let onnxSession = null;

// ==========================================
// INICJALIZACJA WSZYSTKIEGO
// ==========================================
// ==========================================
// INICJALIZACJA WSZYSTKIEGO (Z DEBUGGEREM)
// ==========================================
async function initAll() {
    console.log("--- START INICJALIZACJI ---");

    try {
        console.log("1. Pobieranie zoptymalizowanego modelu ORT...");
        // Wskazujemy na nowy plik z rozszerzeniem .ort!
        const modelUrl = chrome.runtime.getURL("models/gesture_model.ort");
        const response = await fetch(modelUrl);
        const arrayBuffer = await response.arrayBuffer();

        // Zostawiamy 1 wątek i wyłączamy proxy (to zawsze dobra praktyka we wtyczkach)
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = false;

        console.log("2. Tworzenie sesji ORT...");
        // Inicjalizacja z bufora
        onnxSession = await ort.InferenceSession.create(arrayBuffer, {
            executionProviders: ['wasm']
        });

        console.log("✅ SUKCES: Model załadowany!");
    } catch (e) {
        console.error("❌ BŁĄD:", e);
        return;
    }


    // 3. TEST MODELU DŁONI (.task)
    try {
        console.log("3. Pobieranie pliku hand_landmarker.task...");
        const taskPath = chrome.runtime.getURL("models/hand_landmarker.task");
        const response = await fetch(taskPath);
        const buffer = await response.arrayBuffer();

        console.log("4. Inicjalizacja śledzenia dłoni...");
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetBuffer: new Uint8Array(buffer),
                delegate: "CPU" // Używamy CPU dla stabilności w tle
            },
            runningMode: "VIDEO",
            numHands: 1
        });
        console.log("✅ MediaPipe całkowicie gotowy!");
    } catch (e) {
        console.error("❌ BŁĄD MODELU DŁONI! (Czy hand_landmarker.task nie jest przypadkiem plikiem HTML?)", e);
        return;
    }

    // Jeśli doszliśmy tutaj, wszystko działa!
    startWebcam();
}

async function startWebcam() {
    webcam = document.getElementById("webcam");
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
    webcam.addEventListener("loadeddata", predictWebcam);
}

// ==========================================
// GŁÓWNA PĘTLA KAMERY
// ==========================================
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

// ==========================================
// LOGIKA BUFORA I ONNX
// ==========================================
async function handleNewFrame(mediaPipeLandmarks) {
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

    const referencePoint = bufferToProcess[0][0];
    const refX = referencePoint[0];
    const refY = referencePoint[1];

    let flatNormalizedData = [];
    for (let frame of bufferToProcess) {
        for (let i = 0; i < 21; i++) {
            let px = frame[i] ? frame[i][0] : 0.0;
            let py = frame[i] ? frame[i][1] : 0.0;
            flatNormalizedData.push(px - refX);
            flatNormalizedData.push(py - refY);
        }
    }

    const tensorData = new Float32Array(flatNormalizedData);
    const inputTensor = new ort.Tensor('float32', tensorData, [1, GESTURE_CONFIG.BUFFER_SIZE, 42]);

    const results = await onnxSession.run({ 'input_sequence': inputTensor });
    const outputTensor = results['class_probabilities'].data;

    // --- FUNKCJA SOFTMAX (przywrócona!) ---
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
    // ---------------------------------------

    let maxProb = -1;
    let predictedIdx = -1;
    for (let i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
            maxProb = probabilities[i];
            predictedIdx = i;
        }
    }

    if (maxProb >= GESTURE_CONFIG.CONFIDENCE_THRESHOLD) {
        const actionLabel = CLASS_NAMES[predictedIdx];
        console.log(`🎯 Wykryto: ${actionLabel} (${(maxProb * 100).toFixed(1)}%)`);

        chrome.runtime.sendMessage({
            type: "GESTURE_COMMAND",
            action: actionLabel
        });
    }
}

// Start wtyczki!
initAll();