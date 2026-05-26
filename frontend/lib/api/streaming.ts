import type { QueryRequest, QueryResponse } from "@/lib/api/schemas";

export type StreamHandlers = {
  onText: (chunk: string) => void;
  onFinal?: (response: QueryResponse) => void;
};

export async function readStreamingResponse(
  response: Response,
  handlers: StreamHandlers,
) {
  if (!response.body) {
    throw new Error("Streaming response did not include a readable body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\n\n|\r\n\r\n/);
    buffer = events.pop() ?? "";

    for (const event of events) {
      consumeStreamEvent(event, handlers);
    }
  }

  if (buffer.trim()) consumeStreamEvent(buffer, handlers);
}

function consumeStreamEvent(event: string, handlers: StreamHandlers) {
  const lines = event
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) return;

  const dataLines = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim());
  const payload = dataLines.length ? dataLines.join("\n") : lines.join("\n");

  if (payload === "[DONE]") return;

  try {
    const decoded = JSON.parse(payload) as {
      type?: string;
      delta?: string;
      text?: string;
      answer?: string;
      response?: QueryResponse;
    };
    if (decoded.type === "final" && decoded.response) {
      handlers.onFinal?.(decoded.response);
      return;
    }
    handlers.onText(decoded.delta ?? decoded.text ?? decoded.answer ?? "");
  } catch {
    handlers.onText(payload);
  }
}

export function streamingBody(request: QueryRequest) {
  return JSON.stringify(request);
}
