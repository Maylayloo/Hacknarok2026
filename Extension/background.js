async function setupOffscreenDocument() {
    const offscreenUrl = chrome.runtime.getURL('offscreen.html');

    const existingContexts = await chrome.runtime.getContexts({
        contextTypes: ['OFFSCREEN_DOCUMENT'],
        documentUrls: [offscreenUrl]
    });

    if (existingContexts.length > 0) return;

    await chrome.offscreen.createDocument({
        url: 'offscreen.html',
        reasons: ['USER_MEDIA'],
        justification: 'owolowo'
    });
}

chrome.runtime.onInstalled.addListener(setupOffscreenDocument);
chrome.runtime.onStartup.addListener(setupOffscreenDocument);
console.log('eeee')
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log("heay", message)
    if (message.type === "GESTURE_COMMAND") {

        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs.length > 0) {
                chrome.tabs.sendMessage(tabs[0].id, message).catch(() => {

                });
            }
        });
    }
});