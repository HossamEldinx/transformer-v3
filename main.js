import "./style.css";
import javascriptLogo from "./javascript.svg";
import viteLogo from "/vite.svg";
import { pipeline } from "@huggingface/transformers";

const outputDiv = document.getElementById("output");

async function runEmbeddings() {
    try {
        outputDiv.textContent = "Loading embeddings pipeline...";

        // Create a feature-extraction pipeline
        const extractor = await pipeline(
            "feature-extraction",
            "mixedbread-ai/mxbai-embed-xsmall-v1",
            {
                dtype: "fp32",
                device: "webgpu",
            }
        );

        outputDiv.textContent = "Computing embeddings...";

        // Compute embeddings
        const texts = ["Hello world!", "This is an example sentence."];
        const embeddings = await extractor(texts, {
            pooling: "mean",
            normalize: true,
        });
        const results = embeddings.tolist();

        // Display results in a formatted way
        outputDiv.innerHTML = `
            <h3>Input Texts:</h3>
            ${texts.map((text, i) => `<p>${i + 1}. "${text}"</p>`).join("")}
            
            <h3>Embeddings:</h3>
            ${results
                .map(
                    (embedding, i) => `
                <p>Text ${i + 1} (showing first 5 dimensions):</p>
                <p>[${embedding
                    .slice(0, 5)
                    .map((n) => n.toFixed(6))
                    .join(", ")}...]</p>
            `
                )
                .join("")}
        `;
    } catch (error) {
        outputDiv.innerHTML = `<p style="color: red">Error: ${error.message}</p>`;
        console.error(error);
    }
}

window.runEmbeddings = runEmbeddings;
