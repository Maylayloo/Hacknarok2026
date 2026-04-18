const sections = [
    getVideoContainer,
    getRelatedSection,
    getCommentsSection,
];

let active_item = -1;
let highlighted_item = null;
let current_section = 0

const getVideo = () => {
    return document.querySelector("video");
}

function getVideoContainer() {
    return document.querySelector(".html5-video-container");
}

function getCommentsSection() {
    return document.querySelector(".ytd-comments");
}

const highlightVideo = () => {
    const videoContainer = getVideoContainer();
    videoContainer.style.border = "2px solid yellow";
    videoContainer.style.padding = "1rem";
}
function getRelatedSection() {
    return document.querySelector("ytd-watch-next-secondary-results-renderer #contents");
}
function startStopVideo() {
    const event = new KeyboardEvent('keydown', {
        key: 'k',
        code: 'KeyK',
        keyCode: 75,
        which: 75,
        bubbles: true,
        cancelable: true,
        view: window
    });

    document.dispatchEvent(event);
}
function goToNextSection() {
    highlight(sections[current_section](), 'reset');

    current_section++;

    if (current_section >= sections.length) {
        current_section = 0;
    }

    active_item = -1;
    const targetSection = sections[current_section]();

    if (!targetSection) console.warn("smth's wrong bro");

    targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    highlight(targetSection, 'section');
}
function goToPreviousSection() {
    highlight(sections[current_section](), 'reset');

    current_section--;

    if (current_section < 0) {
        current_section = sections.length - 1;
    }

    active_item = -1;

    const targetSection = sections[current_section]();

    if (!targetSection) {
        console.warn("smth's wrong bro");
        return;
    }

    targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    highlight(targetSection, 'section');
}

function theaterMode() {
    document.dispatchEvent(new KeyboardEvent('keydown', {
        key: 't',
        code: 'KeyT',
        keyCode: 84,
        which: 84,
        bubbles: true
    }));
}

function muteVideo() {
    document.dispatchEvent(new KeyboardEvent('keydown', {
        key: 'm',
        code: 'KeyM',
        keyCode: 77,
        which: 77,
        bubbles: true
    }));
}
function volumeUp() {
    const video = getVideo();
    const event = new KeyboardEvent('keydown', {
        key: 'ArrowUp',
        code: 'ArrowUp',
        keyCode: 38,
        which: 38,
        bubbles: true,
        cancelable: true
    });

    video.dispatchEvent(event);
}
function volumeDown() {
    const video = getVideo();
    const event = new KeyboardEvent('keydown', {
        key: 'ArrowDown',
        code: 'ArrowDown',
        keyCode: 40,
        which: 40,
        bubbles: true,
        cancelable: true
    });

    video.dispatchEvent(event);
}
function highlight(element, type) {
    if (type === 'reset') {
        element.style.backgroundColor = "";
        element.style.padding = "";
        element.style.border = "";
    }
    else if (type === 'related_video') {
        element.style.backgroundColor = "rgba(255, 255, 255, 0.1)";
        element.style.padding = "1rem";
        element.style.border = "1px solid rgba(255, 255, 255, 0.3)";
    }
    else if (type === 'section') {
        element.style.border = "3px solid white";
    }
    // else {
    //     element.style.border = "3px solid white";
    // }

}

// won't work until I find workaround :/
// function fullScreen() {
//     document.dispatchEvent(new KeyboardEvent('keydown', {
//         key: 'f',
//         code: 'KeyF',
//         keyCode: 70,
//         which: 70,
//         bubbles: true
//     }));
//     const video = getVideo();
//     video.requestFullscreen();
// }
function scrollToNextChild(section) {
    const children = Array.from(section.children).filter(el => el.offsetHeight > 0);

    active_item++;

    if (active_item >= children.length) {
        active_item = 0;
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
    }

    const next = children[active_item];
    highlighted_item = next;
    highlight(next, 'related_video');
    highlight(children[active_item - 1], 'reset');

    const rect = next.getBoundingClientRect();
    const viewportHeight = window.innerHeight;

    if (rect.bottom > viewportHeight || rect.top > viewportHeight * 0.9) {
        next.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }
}

function scrollToPreviousChild(section) {
    const children = Array.from(section.children).filter(el => el.offsetHeight > 0);

    active_item--;

    if (active_item < 0) {
        active_item = children.length - 1;

        const lastChild = children[active_item];
        lastChild.scrollIntoView({ behavior: 'smooth', block: 'center' });

        highlight(lastChild, 'related_video');
        highlight(children[0], 'reset');
        return;
    }

    const prev = children[active_item];
    highlighted_item = prev;

    highlight(prev, 'related_video');
    highlight(children[active_item + 1], 'reset');

    const rect = prev.getBoundingClientRect();

    if (rect.top < 0 || rect.top < window.innerHeight * 0.1) {
        prev.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }
}

const testActions = [
    () => { theaterMode(); },
    () => startStopVideo(),
    () => muteVideo(),
    () => startStopVideo(),
    () => muteVideo(),
    ...Array(2).fill(() => volumeUp()),
    ...Array(2).fill(() => volumeDown()),
    () => theaterMode(),
    () => window.scrollTo({ top: getCommentsSection().scrollHeight, behavior: 'smooth' }),
    () => window.scrollBy({ top: 200, behavior: 'smooth' }),
    () => window.scrollBy({ top: 200, behavior: 'smooth' }),
    () => window.scrollBy({ top: 300, behavior: 'smooth' }),
    () => window.scrollBy({ top: 500, behavior: 'smooth' }),
    ...Array(4).fill(() => window.scrollBy({ top: -200, behavior: 'smooth' })),
    () => {getRelatedSection().scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
    },
    () => goToNextSection(),
    () => goToNextSection(),
    () => goToNextSection(),
    () => goToNextSection(),
    () => goToPreviousSection(),
    () => goToPreviousSection(),
    () => goToPreviousSection(),
    () => goToNextSection(),
    () => goToNextSection(),
    () => goToNextSection(),
    () => window.scrollBy({ top: -300, behavior: 'smooth' }),
    () => {scrollToNextChild(getRelatedSection());},
    () => {scrollToNextChild(getRelatedSection());},
    () => {scrollToNextChild(getRelatedSection());},
    () => {scrollToPreviousChild(getRelatedSection());},
    () => {scrollToPreviousChild(getRelatedSection());},
    () => {scrollToPreviousChild(getRelatedSection());},
    () => {scrollToNextChild(getRelatedSection());},
    () => {scrollToNextChild(getRelatedSection());},
    () => {scrollToNextChild(getRelatedSection());},
    // () => {{
    //     const clickableVideo = highlighted_item.querySelector('yt-touch-feedback-shape');
    //     clickableVideo.click();
    // }}

];
// yt-touch-feedback-shape
testActions.forEach((action, index) => {
    setTimeout(action, index * 600);
});