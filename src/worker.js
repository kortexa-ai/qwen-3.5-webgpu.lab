import {
  AutoProcessor,
  AutoModelForImageTextToText,
  TextStreamer,
  RawImage,
} from "@huggingface/transformers";

const MODEL_ID = "onnx-community/Qwen3.5-0.8B-ONNX";

let processor = null;
let model = null;

// Check if model files are in the transformers cache
async function isModelCached() {
  try {
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();
    return keys.some(
      (req) =>
        req.url.includes("Qwen3.5-0.8B-ONNX") ||
        req.url.includes("Qwen3.5-0.8B-ONNX".replace("/", "%2F"))
    );
  } catch {
    return false;
  }
}

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === "check") {
    const cached = await isModelCached();
    self.postMessage({ type: "cache_status", data: { cached } });
    return;
  }

  if (type === "load") {
    try {
      self.postMessage({ type: "status", data: "Loading processor..." });
      processor = await AutoProcessor.from_pretrained(MODEL_ID);

      self.postMessage({
        type: "status",
        data: "Loading model (q4f16, ~650MB)...",
      });
      model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
        device: "webgpu",
        dtype: "q4f16",
        progress_callback: (progress) => {
          if (progress.status === "progress") {
            self.postMessage({ type: "progress", data: progress });
          }
        },
      });

      self.postMessage({ type: "loaded" });
    } catch (err) {
      console.error("[worker] load error:", err);
      self.postMessage({ type: "error", data: err.message });
    }
    return;
  }

  if (type === "generate") {
    if (!processor || !model) {
      self.postMessage({ type: "error", data: "Model not loaded" });
      return;
    }

    try {
      const messages = data.messages || [];
      const maxTokens = data.maxTokens || 2048;

      // Find the message with an image (if any)
      const imageMsg = messages.find((m) => m.image);

      // Build messages in the format expected by apply_chat_template
      const chatMessages = messages.map((m) => {
        if (m.image) {
          return {
            role: m.role,
            content: [
              { type: "image" },
              { type: "text", text: m.content },
            ],
          };
        }
        return { role: m.role, content: m.content };
      });

      // Apply chat template to get the formatted prompt text
      const text = processor.apply_chat_template(chatMessages, {
        tokenize: false,
        add_generation_prompt: true,
      });

      // Load image if present
      let inputs;
      if (imageMsg) {
        self.postMessage({ type: "status", data: "Processing image..." });
        const image = await RawImage.read(imageMsg.image);
        inputs = await processor(text, [image]);
      } else {
        inputs = await processor(text);
      }

      // Setup token streaming
      let tokenCount = 0;
      const startTime = performance.now();
      self.postMessage({ type: "generate_start" });

      const streamer = new TextStreamer(processor.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: false,
        callback_function: (text) => {
          tokenCount++;
          const elapsed = (performance.now() - startTime) / 1000;
          self.postMessage({
            type: "token",
            data: {
              text,
              tokenCount,
              tokensPerSec: tokenCount / elapsed,
            },
          });
        },
      });

      // Generate with sampling (matches model's generation_config)
      await model.generate({
        ...inputs,
        max_new_tokens: maxTokens,
        do_sample: true,
        temperature: 0.6,
        top_k: 20,
        top_p: 0.95,
        streamer,
      });

      const elapsed = (performance.now() - startTime) / 1000;
      self.postMessage({
        type: "generate_done",
        data: {
          tokenCount,
          elapsed,
          tokensPerSec: tokenCount / elapsed,
        },
      });
    } catch (err) {
      console.error("[worker] generate error:", err);
      self.postMessage({ type: "error", data: err.message });
    }
  }
};
