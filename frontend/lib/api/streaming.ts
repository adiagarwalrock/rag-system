import type { QueryRequest } from "@/lib/api/schemas";

export type StreamHandlers<TFinal = unknown> = {
  onText?: (chunk: string) => void;
  onStatus?: (phase: string) => void;
  onReasoning?: (delta: string) => void;
  onSession?: (session: { session_id?: string; user_message_id?: string }) => void;
  onFinal?: (response: TFinal) => void;
  onError?: (message: string, detail?: unknown) => void;
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

  // Extract the named SSE event type (e.g. "event: status") if present.
  // The FastAPI route emits:
  //   event: status     data: {"delta": "..."}
  //   event: reasoning  data: {"delta": "..."}
  //   event: final      data: {<QueryResponse fields>}
  //   event: error      data: {"detail": "..."}
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const eventType = eventLine ? eventLine.slice(6).trim() : undefined;

  const dataLines = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim());
  const payload = dataLines.length ? dataLines.join("\n") : lines.join("\n");

  if (payload === "[DONE]") return;

  // Route by SSE event name first. Fall back to inspecting a "type" field
  // inside JSON for compatibility with older/plain streaming responses.
  try {
    const decoded = JSON.parse(payload) as {
      type?: string;
      delta?: string;
      text?: string;
      answer?: string;
      response?: unknown;
      result?: unknown;
      detail?: string;
      message?: string;
      session_id?: string;
      user_message_id?: string;
    };

    const resolvedType = eventType ?? decoded.type;

    if (resolvedType === "session") {
      handlers.onSession?.({
        session_id: decoded.session_id,
        user_message_id: decoded.user_message_id,
      });
      return;
    }
    if (resolvedType === "status" || resolvedType === "phase") {
      handlers.onStatus?.(decoded.delta ?? "");
      return;
    }
    if (resolvedType === "reasoning" || resolvedType === "reasoning_delta") {
      handlers.onReasoning?.(decoded.delta ?? "");
      return;
    }
    if (
      resolvedType === "answer" ||
      resolvedType === "answer_delta" ||
      resolvedType === "delta" ||
      resolvedType === "token"
    ) {
      handlers.onText?.(decoded.delta ?? decoded.text ?? decoded.answer ?? "");
      return;
    }
    if (resolvedType === "final") {
      // With named SSE events the final payload is usually the response itself.
      // Some clients wrap it under response/result, so handle both shapes.
      const responsePayload = decoded.response ?? decoded.result ?? decoded;
      handlers.onFinal?.(responsePayload);
      return;
    }
    if (resolvedType === "error") {
      const message = decoded.detail ?? decoded.message ?? "The query stream failed.";
      handlers.onError?.(message, decoded);
      return;
    }
    handlers.onText?.(decoded.delta ?? decoded.text ?? decoded.answer ?? "");
  } catch {
    handlers.onText?.(payload);
  }
}

export function streamingBody(request: QueryRequest) {
  return JSON.stringify(request);
}
