import { FilesetResolver, HandLandmarker } from "./libs/vision_bundle.js";

ort.env.wasm.wasmPaths = chrome.runtime.getURL("libs/");

const GESTURE_CONFIG = {
    BUFFER_SIZE: 35,         // Twój model oczekuje 30 klatek
    INFERENCE_INTERVAL: 1,  // Odpalamy model co 10 klatek (dla wydajności)
    CONFIDENCE_THRESHOLD: 0.85 // Minimalna pewność modelu (85%)
};
const CLASS_NAMES = {0: 'BIG_LEFT', 1: 'BIG_RIGHT', 2: 'CINEMA_MODE', 3: 'CLICK', 4: 'IDLE', 5: 'POINT_UP', 6: 'OPEN_PALM', 7: 'SITE_LEFT', 8: 'SMALL_LEFT', 9: 'SMALL_RIGHT', 10: 'SWIPE_LEFT', 11: 'SWIPE_RIGHT', 12: 'VIDEO', 13: 'VOLUME_DOWN', 14: 'VOLUME_UP'}; // Zaktualizuj, jeśli masz więcej klas

// Zmienne globalne
let handLandmarker = null;
let webcam = null;
let lastVideoTime = -1;
let keypointsBuffer = [];
let framesProcessed = 0;
let isModelRunning = false;
let onnxSession = null;
let canvasElement = null;
let canvasCtx = null;

// ==========================================
// INICJALIZACJA WSZYSTKIEGO
// ==========================================
async function initAll() {
    try {
        console.log("🚀 Start inicjalizacji...");

        // 1. Konfiguracja środowiska ONNX
        // Musimy to ustawić ZANIM stworzymy sesję

        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = false;
        ort.env.wasm.wasmPaths = chrome.runtime.getURL("libs/");
        ort.env.wasm.simd = false;

        console.log("📦 Pobieranie modelu ONNX z plików rozszerzenia...");
        const modelUrl = chrome.runtime.getURL("models/gesture_model.ort");
        const modelResponse = await fetch(modelUrl);

        if (!modelResponse.ok) {
            throw new Error(`Nie udało się pobrać modelu ONNX: ${modelResponse.statusText}`);
        }

        const modelBuffer = await modelResponse.arrayBuffer();
        console.log('skibidi', modelBuffer);

        console.log("🧠 Tworzenie sesji ONNX (InferenceSession)...");
        onnxSession = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        console.log("Model ONNX załadowany pomyślnie!");

        // 2. Konfiguracja MediaPipe
        console.log("📸 Inicjalizacja MediaPipe Hand Landmarker...");
        const wasmPath = chrome.runtime.getURL("libs/");
        const vision = await FilesetResolver.forVisionTasks(wasmPath);
        const taskPath = chrome.runtime.getURL("models/hand_landmarker.task");

        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: taskPath,
                delegate: "CPU" // Jeśli tu wywali błąd, zmień na "CPU"
            },
            runningMode: "VIDEO",
            numHands: 1
        });
        console.log("MediaPipe gotowy!");

        // 3. Start kamery
        await startWebcam();

    } catch (err) {
        console.error("Krytyczny błąd w initAll:", err);
    }
}

async function startWebcam() {
    webcam = document.getElementById("webcam");
    canvasElement = document.getElementById("output_canvas");
    canvasCtx = canvasElement.getContext("2d");
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480
            }
        });
        webcam.srcObject = stream;
        webcam.onloadedmetadata = () => {
            webcam.play();
            predictWebcam();
        };
        console.log("Kamera w offscreen g");
    } catch (err) {
        console.error("Offscreen Camera Error:", err.name, err.message);
    }
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
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(webcam, 0, 0, canvasElement.width, canvasElement.height);
        if (results.landmarks && results.landmarks.length > 0) {
            for (const landmarks of results.landmarks) {
                drawLandmarks(landmarks);
            }
            handleNewFrame(results.landmarks);
        }
        canvasCtx.restore();
    }
    window.requestAnimationFrame(predictWebcam);
}
function drawLandmarks(landmarks) {
    // Draw dots for each keypoint
    canvasCtx.fillStyle = "#00FF00"; // Green points
    for (const point of landmarks) {
        const x = point.x * canvasElement.width;
        const y = point.y * canvasElement.height;

        canvasCtx.beginPath();
        canvasCtx.arc(x, y, 3, 0, 2 * Math.PI);
        canvasCtx.fill();
    }

    // Logic to draw lines (connections) can be added here
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
    console.log('bufle', keypointsBuffer.length, framesProcessed);
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
    console.log("maxprob", maxProb);
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