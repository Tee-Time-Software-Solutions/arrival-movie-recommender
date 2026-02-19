import { cn } from "@/lib/utils";
import type { ChatMessage as ChatMessageType } from "@/types/chat";

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex w-full",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-2.5",
          isUser
            ? "rounded-br-sm bg-primary text-primary-foreground"
            : "rounded-bl-sm bg-card text-card-foreground",
        )}
      >
        <p className="whitespace-pre-wrap text-sm">{message.content}</p>

        {message.movieRecommendations &&
          message.movieRecommendations.length > 0 && (
            <div className="mt-3 space-y-2">
              {message.movieRecommendations.map((rec) => (
                <div
                  key={rec.title}
                  className="flex gap-3 rounded-lg bg-black/10 p-2"
                >
                  <img
                    src={rec.posterUrl}
                    alt={rec.title}
                    className="h-16 w-11 shrink-0 rounded object-cover"
                  />
                  <div className="min-w-0">
                    <p className="text-sm font-medium">
                      {rec.title}{" "}
                      <span className="text-xs opacity-70">({rec.year})</span>
                    </p>
                    <p className="mt-0.5 text-xs opacity-70">{rec.reason}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

        <p
          className={cn(
            "mt-1 text-[10px]",
            isUser ? "text-primary-foreground/60" : "text-muted-foreground",
          )}
        >
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </div>
    </div>
  );
}
