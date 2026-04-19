console.log("chat does it work")

const getMicro = () => {
    return document.querySelector('[aria-label="Rozpocznij dyktowanie"]')
}

const getAccept = () => {
    return document.querySelector('[aria-label="Wyślij dyktowanie"]')
}

const getSend = () => {
    return document.querySelector('[aria-label="Wyślij polecenie"]')

}
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === "GESTURE_COMMAND") {
        const action = request.action;
        console.log("gesture", action);

        if (action === "CLICK") {
            getMicro().click();
        }
        else if (action === "VOLUME_UP") {
            getAccept().click();
        }
        else if (action === "OPEN_PALM") {
            getSend().click();
        }

    }
});