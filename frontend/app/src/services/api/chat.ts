import type { ChatMessage } from "@/types/chat";

// Echo mode — no backend calls
export async function sendMessage(message: string): Promise<ChatMessage> {
  await new Promise((resolve) => setTimeout(resolve, 500));

  return {
    id: `msg-${Date.now()}`,
    content: `Echo: ${message}`,
    role: "assistant",
    timestamp: new Date().toISOString(),
  };
}

export async function getChatHistory(): Promise<ChatMessage[]> {
  return [];
}
