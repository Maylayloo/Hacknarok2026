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
        const action = message.action;

        if (action === "SWIPE_LEFT" || action === "SWIPE_RIGHT") {
            switchTab(action === "SWIPE_LEFT" ? -1 : 1);
            return;
        }

        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs.length > 0) {
                chrome.tabs.sendMessage(tabs[0].id, message).catch(() => {

                });
            }
        });
    }
});

function switchTab(direction) {
    chrome.tabs.query({ currentWindow: true }, (tabs) => {
        if (tabs.length <= 1) return; // Nic do przełączania

        const activeTabIndex = tabs.findIndex(tab => tab.active);

        let nextIndex = activeTabIndex + direction;

        if (nextIndex < 0) {
            nextIndex = tabs.length - 1;
        } else if (nextIndex >= tabs.length) {
            nextIndex = 0;
        }

        chrome.tabs.update(tabs[nextIndex].id, { active: true });
    });
}