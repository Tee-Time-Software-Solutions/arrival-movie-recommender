import { create } from "zustand";
import type { ChatMessage } from "@/types/chat";
import type { MovieDetails } from "@/types/movie";

interface ChatState {
  messages: ChatMessage[];
  isTyping: boolean;
  addMessage: (message: ChatMessage) => void;
  updateLastAssistantContent: (token: string) => void;
  setLastAssistantMovies: (movies: MovieDetails[]) => void;
  setLastAssistantError: (errorText: string) => void;
  removeMessageById: (id: string) => void;
  setTyping: (isTyping: boolean) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isTyping: false,
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  updateLastAssistantContent: (token) =>
    set((state) => {
      const msgs = [...state.messages];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") {
          msgs[i] = { ...msgs[i], content: msgs[i].content + token };
          break;
        }
      }
      return { messages: msgs };
    }),
  setLastAssistantMovies: (movies) =>
    set((state) => {
      const msgs = [...state.messages];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") {
          msgs[i] = { ...msgs[i], movieRecommendations: movies };
          break;
        }
      }
      return { messages: msgs };
    }),
  setLastAssistantError: (errorText) =>
    set((state) => {
      const msgs = [...state.messages];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") {
          msgs[i] = { ...msgs[i], content: errorText, status: "error" };
          break;
        }
      }
      return { messages: msgs };
    }),
  removeMessageById: (id) =>
    set((state) => ({ messages: state.messages.filter((m) => m.id !== id) })),
  setTyping: (isTyping) => set({ isTyping }),
}));
