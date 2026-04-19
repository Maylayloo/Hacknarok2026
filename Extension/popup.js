document.addEventListener('DOMContentLoaded', async () => {
    const btn = document.getElementById('cameraBtn');
    const status = document.getElementById('status');

    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const hasVideo = devices.some(device => device.kind === 'videoinput' && device.label !== '');
        if (hasVideo) {
            status.innerText = "Bracie. Testuj to.";
            btn.style.display = 'none';
        }
    } catch (e) {
        console.log("Cos jest zle, ale w sumie nie wiem co.");
    }

    btn.addEventListener('click', async () => {
        try {
            status.innerText = "Możemy się ładowac :)";

            const stream = await navigator.mediaDevices.getUserMedia({ video: true });

            stream.getTracks().forEach(track => track.stop());

            status.innerText = "What the dog doin'";
            btn.style.display = 'none';

        } catch (err) {
            status.innerText = "No i dupa";
            console.error("Błąd", err);
        }
    });
});