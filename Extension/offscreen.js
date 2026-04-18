// 1. BEZPOŚREDNIE IMPORTY (Działa od razu w przeglądarce, bez bundlerów!)
import { FilesetResolver, HandLandmarker } from "./libs/vision_bundle.js";
import * as ort from "./libs/ort.bundle.min.js";
// ==========================================
// KONFIGURACJA
// ==========================================
ort.env.wasm.wasmPaths = chrome.runtime.getURL("libs/");

const GESTURE_CONFIG = {
    BUFFER_SIZE: 30,         // Twój model oczekuje 30 klatek
    INFERENCE_INTERVAL: 10,  // Odpalamy model co 10 klatek (dla wydajności)
    CONFIDENCE_THRESHOLD: 0.85 // Minimalna pewność modelu (85%)
};
const CLASS_NAMES = {0: "PALM_OPEN", 1: "SWIPE_LEFT"}; // Zaktualizuj, jeśli masz więcej klas

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
async function initAll() {
    console.log("Ładowanie modelu ONNX...");
    const modelUrl = chrome.runtime.getURL("models/gesture_model.onnx");
    onnxSession = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    console.log("Model ONNX załadowany!");

    console.log("Ładowanie MediaPipe...");
    // Podajemy ścieżkę do folderu z plikami vision_wasm_internal
    const wasmPath = chrome.runtime.getURL("libs/");
    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    // Ładujemy lokalny model z folderu models!
    const taskPath = chrome.runtime.getURL("models/hand_landmarker.task");

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: taskPath,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });
    console.log("MediaPipe gotowy!");

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

        // Zdobądź punkty z obrazu
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

    // Zapisz [x, y] dla 21 punktów z pierwszej dłoni
    const frameKeypoints = mediaPipeLandmarks[0].map(lm => [lm.x, lm.y]);
    keypointsBuffer.push(frameKeypoints);

    // Pilnuj rozmiaru bufora (zawsze 30 klatek)
    if (keypointsBuffer.length > GESTURE_CONFIG.BUFFER_SIZE) {
        keypointsBuffer.shift();
    }

    // Warunki odpalenia modelu
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

    // Normalizacja (odjęcie pierwszego punktu z pierwszej klatki)
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

    // Odpal ONNX
    const results = await onnxSession.run({ 'input_sequence': inputTensor });
    const outputTensor = results['class_probabilities'].data;

    // Szukamy największego prawdopodobieństwa
    let maxProb = -1;
    let predictedIdx = -1;
    for (let i = 0; i < outputTensor.length; i++) {
        if (outputTensor[i] > maxProb) {
            maxProb = outputTensor[i];
            predictedIdx = i;
        }
    }

    // Jeśli jesteśmy pewni wyślij do przeglądarki!
    if (maxProb >= GESTURE_CONFIG.CONFIDENCE_THRESHOLD) {
        const actionLabel = CLASS_NAMES[predictedIdx];
        console.log(`🎯 Wykryto: ${actionLabel} (${(maxProb * 100).toFixed(1)}%)`);

        // Komunikacja z background.js
        chrome.runtime.sendMessage({
            type: "GESTURE_COMMAND",
            action: actionLabel
        });
    }
}

// Start wtyczki!
initAll();