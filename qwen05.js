import { pipeline } from "@huggingface/transformers";

const outputDiv = document.getElementById("output");

async function runTextGeneration() {
    try {
        outputDiv.textContent = "Loading text generation pipeline...";

        // Create a text generation pipeline
        const generator = await pipeline(
            "text-generation",
            "onnx-community/Qwen2.5-0.5B-Instruct",
            { dtype: "q4", device: "webgpu" }
        );

        outputDiv.textContent = "Generating text...";

        // Define the list of messages
        const messages = [
            { role: "system", content: "You are a helpful assistant." },
            { role: "user", content: "Tell me a funny joke." },
        ];

        // Generate a response
        const output = await generator(messages, { max_new_tokens: 128 });
        const response = output[0].generated_text.at(-1).content;

        outputDiv.innerHTML = `
            <h3>Text Generation Results:</h3>
            <h4>Prompt:</h4>
            ${messages
                .map(
                    (msg) => `
                <p><strong>${msg.role}:</strong> ${msg.content}</p>
            `
                )
                .join("")}
            <h4>Generated Response:</h4>
            <p>${response}</p>
        `;
    } catch (error) {
        outputDiv.innerHTML = `<p style="color: red">Error: ${error.message}</p>`;
        console.error(error);
    }
}

window.runTextGeneration = runTextGeneration;
