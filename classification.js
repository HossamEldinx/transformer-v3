import { pipeline } from "@huggingface/transformers";

const outputDiv = document.getElementById("output");

async function runImageClassification() {
    try {
        outputDiv.textContent = "Loading image classification pipeline...";

        // Create image classification pipeline
        const classifier = await pipeline(
            "image-classification",
            "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
            { device: "webgpu" }
        );

        outputDiv.textContent = "Classifying image...";

        // Classify an image from a URL
        const url =
            "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg";
        const output = await classifier(url);

        outputDiv.innerHTML = `
            <h3>Image Classification Results:</h3>
            <p>Image URL: ${url}</p>
            <img src="${url}" style="max-width: 300px; margin: 10px 0;">
            <h4>Predictions:</h4>
            ${output
                .map(
                    (pred) => `
                <p>${pred.label}: ${(pred.score * 100).toFixed(2)}%</p>
            `
                )
                .join("")}
        `;
    } catch (error) {
        outputDiv.innerHTML = `<p style="color: red">Error: ${error.message}</p>`;
        console.error(error);
    }
}

window.runImageClassification = runImageClassification;
