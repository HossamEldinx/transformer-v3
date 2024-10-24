import { pipeline } from "@huggingface/transformers";

const outputDiv = document.getElementById("output");

async function runSpeechRecognition() {
    try {
        outputDiv.textContent = "Loading speech recognition pipeline...";

        // Create automatic speech recognition pipeline
        const transcriber = await pipeline(
            "automatic-speech-recognition",
            "onnx-community/whisper-tiny.en",
            { device: "webgpu" }
        );

        outputDiv.textContent = "Transcribing audio...";

        // Transcribe audio from a URL
        const url =
            "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";
        const output = await transcriber(url);

        outputDiv.innerHTML = `
            <h3>Speech Recognition Results:</h3>
            <p>Audio URL: ${url}</p>
            <p>Transcription: ${output.text}</p>
        `;
    } catch (error) {
        outputDiv.innerHTML = `<p style="color: red">Error: ${error.message}</p>`;
        console.error(error);
    }
}

window.runSpeechRecognition = runSpeechRecognition;
