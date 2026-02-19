import { useState, useRef, useEffect } from "react";
import { Send } from "lucide-react";
import { useChat } from "@/hooks/useChat";
import { ChatMessage } from "@/components/features/ChatMessage/ChatMessage";

export function ChatPage() {
  const [input, setInput] = useState("");
  const { messages, isTyping, sendMessage } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed) return;
    setInput("");
    await sendMessage(trimmed);
  };

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col md:h-screen">
      {/* Header */}
      <div className="border-b border-border px-4 py-3">
        <h1 className="text-lg font-bold">Movie Assistant</h1>
        <p className="text-xs text-muted-foreground">
          Ask me for movie recommendations
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <p className="mb-2 text-lg font-medium">
                What kind of movie are you in the mood for?
              </p>
              <p className="text-sm text-muted-foreground">
                Try: "Recommend a sci-fi movie" or "Something like Inception"
              </p>
            </div>
          </div>
        )}

        <div className="space-y-3">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}

          {isTyping && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-sm bg-card px-4 py-3">
                <div className="flex gap-1">
                  <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:0ms]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:150ms]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:300ms]" />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="border-t border-border p-4 pb-20 md:pb-4"
      >
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask for a recommendation..."
            className="flex-1 rounded-xl border border-input bg-background px-4 py-2.5 text-sm outline-none focus:ring-2 focus:ring-ring"
          />
          <button
            type="submit"
            disabled={!input.trim() || isTyping}
            className="rounded-xl bg-primary px-4 py-2.5 text-primary-foreground transition hover:bg-primary/90 disabled:opacity-50"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </form>
    </div>
  );
}
