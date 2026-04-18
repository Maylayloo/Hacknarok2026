// ==========================================
// 1. ZARZĄDZANIE OFFSCREEN DOCUMENT (KAMERA W TLE)
// ==========================================
let creating; // Zabezpieczenie przed podwójnym otwarciem

async function setupOffscreenDocument(path) {
    // Sprawdzamy, czy Offscreen już działa
    const existingContexts = await chrome.runtime.getContexts({
        contextTypes: ['OFFSCREEN_DOCUMENT'],
        documentUrls: [chrome.runtime.getURL(path)]
    });

    if (existingContexts.length > 0) {
        return; // Kamera już działa, nic nie robimy
    }

    // Jeśli nie działa, tworzymy nowy Offscreen
    if (creating) {
        await creating;
    } else {
        creating = chrome.offscreen.createDocument({
            url: path,
            reasons: ['USER_MEDIA'],
            justification: 'Wykrywanie gestów z kamery dla YouTube'
        });
        await creating;
        creating = null;
        console.log("Kamera w tle została uruchomiona!");
    }
}

// Uruchamiamy kamerę przy instalacji wtyczki...
chrome.runtime.onInstalled.addListener(() => {
    setupOffscreenDocument('offscreen.html');
});

// ...ORAZ przy każdym nowym włączeniu przeglądarki Chrome!
chrome.runtime.onStartup.addListener(() => {
    setupOffscreenDocument('offscreen.html');
});

// ==========================================
// 2. PRZEKAZYWANIE WIADOMOŚCI (Offscreen -> YouTube)
// ==========================================
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {

    // Zgodnie z tym, co wysyła nasz offscreen.js:
    if (request.type === "GESTURE_COMMAND") {
        console.log(`Background otrzymał gest: ${request.action}. Przekazuję do YouTube'a...`);

        // Znajdź aktywną zakładkę, którą widzi użytkownik
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs[0]) {
                // Przekaż ten sam gest bezpośrednio do pliku video.js!
                chrome.tabs.sendMessage(tabs[0].id, {
                    type: "GESTURE_COMMAND",
                    action: request.action
                });
            }
        });
    }
});