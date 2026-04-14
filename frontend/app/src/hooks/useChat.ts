import { useChatStore } from "@/stores/chatStore";
import { sendMessage as sendChatMessage } from "@/services/api/chat";
import type { ChatMessage } from "@/types/chat";

export function useChat() {
  const { messages, isTyping, addMessage, setTyping } = useChatStore();

  const sendMessage = async (content: string) => {
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      content,
      role: "user",
      timestamp: new Date().toISOString(),
    };

    addMessage(userMessage);
    setTyping(true);

    try {
      const response = await sendChatMessage(content);
      addMessage(response);
    } catch {
      addMessage({
        id: `msg-err-${Date.now()}`,
        content: "Sorry, I couldn't process your message. Please try again.",
        role: "assistant",
        timestamp: new Date().toISOString(),
      });
    } finally {
      setTyping(false);
    }
  };

  return { messages, isTyping, sendMessage };
}
