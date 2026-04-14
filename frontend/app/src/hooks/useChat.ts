import { useChatStore } from "@/stores/chatStore";
import { streamChat } from "@/services/api/chat";
import type { ChatMessage } from "@/types/chat";

const MAX_HISTORY = 10;

export function useChat() {
  const { messages, isTyping, addMessage, updateLastAssistantContent, setLastAssistantMovies, setTyping } =
    useChatStore();

  const sendMessage = async (content: string) => {
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      content,
      role: "user",
      timestamp: new Date().toISOString(),
    };

    addMessage(userMessage);

    // Create empty assistant message placeholder for streaming
    const assistantMessage: ChatMessage = {
      id: `msg-${Date.now() + 1}`,
      content: "",
      role: "assistant",
      timestamp: new Date().toISOString(),
    };
    addMessage(assistantMessage);
    setTyping(true);

    // Build history from previous messages (cap at MAX_HISTORY)
    const currentMessages = useChatStore.getState().messages;
    const history = currentMessages
      .slice(-(MAX_HISTORY + 2), -2) // exclude the two messages we just added
      .map((msg) => ({ role: msg.role, content: msg.content }));

    try {
      await streamChat(content, history, {
        onToken: (token) => updateLastAssistantContent(token),
        onMovies: (movies) => setLastAssistantMovies(movies),
        onTasteProfile: () => {
          // Taste profile data is consumed by the LLM to generate text;
          // no separate UI rendering needed
        },
        onDone: () => setTyping(false),
        onError: (error) => {
          updateLastAssistantContent(
            error || "Sorry, something went wrong. Please try again.",
          );
          setTyping(false);
        },
      });
    } catch {
      updateLastAssistantContent(
        "Sorry, I couldn't connect to the server. Please try again.",
      );
      setTyping(false);
    }
  };

  return { messages, isTyping, sendMessage };
}
