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
  signal?: AbortSignal,
): Promise<void> {
  const user = auth.currentUser;
  const firebaseToken = user ? await user.getIdToken() : "";

  const response = await fetch(
    `${import.meta.env.VITE_BASE_URL}/chatbot/stream`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${firebaseToken}`,
      },
      body: JSON.stringify({ message, history }),
      signal,
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

    // SSE blocks are separated by double newlines (handles both \r\n\r\n and \n\n)
    // Normalize \r\n to \n first
    buffer = buffer.replace(/\r\n/g, "\n");
    const blocks = buffer.split("\n\n");
    // Last element may be incomplete — keep it in buffer
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
            return;
          case "error":
            callbacks.onError(data.error || "Unknown error");
            return;
        }
      } catch {
        // Skip malformed SSE data
      }
    }
  }

  // If stream ended without a done event
  callbacks.onDone();
}
