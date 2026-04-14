import { auth } from "@/lib/firebase";
import type { MovieDetails } from "@/types/movie";

export interface SSECallbacks {
  onToken: (token: string) => void;
  onMovies: (movies: MovieDetails[]) => void;
  onTasteProfile: (profile: Record<string, unknown>) => void;
  onDone: () => void;
  onError: (error: string) => void;
}

export async function streamChat(
  message: string,
  history: Array<{ role: string; content: string }>,
  callbacks: SSECallbacks,
): Promise<void> {
  const user = auth.currentUser;
  const token = user ? await user.getIdToken() : "";

  const response = await fetch(
    `${import.meta.env.VITE_BASE_URL}/chatbot/stream`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ message, history }),
    },
  );

  if (!response.ok) {
    callbacks.onError(`Chat request failed: ${response.status}`);
    return;
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE format: "event: <type>\ndata: <json>\n\n"
    const blocks = buffer.split("\n\n");
    // Last element is incomplete — keep it in buffer
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      if (!block.trim()) continue;

      let eventType = "";
      let dataStr = "";

      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataStr = line.slice(5).trim();
        }
      }

      if (!eventType || !dataStr) continue;

      try {
        const data = JSON.parse(dataStr);

        switch (eventType) {
          case "token":
            callbacks.onToken(data.token);
            break;
          case "movies":
            callbacks.onMovies(data.movies);
            break;
          case "taste_profile":
            callbacks.onTasteProfile(data.profile);
            break;
          case "done":
            callbacks.onDone();
            break;
          case "error":
            callbacks.onError(data.error || "Unknown error");
            break;
        }
      } catch {
        // Skip malformed SSE data
      }
    }
  }

  // If stream ended without a done event
  callbacks.onDone();
}
