import { useCallback, useRef } from "react";
import { useChatStore } from "@/stores/chatStore";
import { streamChat } from "@/services/api/chat";
import type { ChatMessage } from "@/types/chat";

const MAX_HISTORY = 10;

const NETWORK_ERROR_TEXT =
  "Sorry, I couldn't connect to the server. Please try again.";
const GENERIC_ERROR_TEXT =
  "Sorry, something went wrong. Please try again.";

export function useChat() {
  const {
    messages,
    isTyping,
    addMessage,
    updateLastAssistantContent,
    setLastAssistantMovies,
    setLastAssistantError,
    removeMessageById,
    setTyping,
  } = useChatStore();

  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (content: string) => {
      // Cancel any in-flight stream before starting a new one.
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const userMessage: ChatMessage = {
        id: `msg-${Date.now()}`,
        content,
        role: "user",
        timestamp: new Date().toISOString(),
      };

      addMessage(userMessage);

      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        content: "",
        role: "assistant",
        timestamp: new Date().toISOString(),
      };
      addMessage(assistantMessage);
      setTyping(true);

      const currentMessages = useChatStore.getState().messages;
      const history = currentMessages
        .slice(-(MAX_HISTORY + 2), -2) // exclude the two we just added
        .map((msg) => ({ role: msg.role, content: msg.content }));

      try {
        await streamChat(
          content,
          history,
          {
            onToken: (token) => updateLastAssistantContent(token),
            onMovies: (movies) => setLastAssistantMovies(movies),
            onTasteProfile: () => {
              // Consumed by the LLM only.
            },
            onDone: () => setTyping(false),
            onError: (error) => {
              setLastAssistantError(error || GENERIC_ERROR_TEXT);
              setTyping(false);
            },
          },
          controller.signal,
        );
      } catch (err) {
        // AbortError = user pressed Stop; drop the empty assistant placeholder.
        if (err instanceof DOMException && err.name === "AbortError") {
          removeMessageById(assistantMessage.id);
          setTyping(false);
          return;
        }
        setLastAssistantError(NETWORK_ERROR_TEXT);
        setTyping(false);
      }
    },
    [
      addMessage,
      removeMessageById,
      setLastAssistantError,
      setLastAssistantMovies,
      setTyping,
      updateLastAssistantContent,
    ],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const retryLast = useCallback(() => {
    // Find the most recent user message; drop the failed assistant turn that
    // followed it, then re-send.
    const all = useChatStore.getState().messages;
    let lastUserIdx = -1;
    for (let i = all.length - 1; i >= 0; i--) {
      if (all[i].role === "user") {
        lastUserIdx = i;
        break;
      }
    }
    if (lastUserIdx === -1) return;
    const lastUser = all[lastUserIdx];

    // Remove everything from the failed user turn onwards so the retry
    // doesn't get duplicated in the rendered history.
    for (let i = all.length - 1; i >= lastUserIdx; i--) {
      removeMessageById(all[i].id);
    }
    sendMessage(lastUser.content);
  }, [removeMessageById, sendMessage]);

  return { messages, isTyping, sendMessage, cancel, retryLast };
}
