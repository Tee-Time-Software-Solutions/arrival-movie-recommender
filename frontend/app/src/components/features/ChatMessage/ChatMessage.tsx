import { cn } from "@/lib/utils";
import { Popcorn, User } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/types/chat";

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex gap-3 py-6",
        isUser && "flex-row-reverse",
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-foreground/10" : "bg-primary/10",
        )}
      >
        {isUser ? (
          <User className="h-4 w-4 text-foreground/70" />
        ) : (
          <Popcorn className="h-4 w-4 text-primary" />
        )}
      </div>

      {/* Content */}
      <div className={cn("min-w-0 max-w-full flex-1", isUser && "flex flex-col items-end")}>
        <p className="mb-1 text-xs font-medium text-muted-foreground">
          {isUser ? "You" : "Arrival"}
        </p>
        <div
          className={cn(
            "prose prose-sm max-w-none text-foreground",
            "prose-p:leading-relaxed prose-p:my-1",
            isUser && "text-right",
          )}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>

        {/* Movie Recommendations */}
        {message.movieRecommendations &&
          message.movieRecommendations.length > 0 && (
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {message.movieRecommendations.map((rec) => (
                <div
                  key={rec.title}
                  className="group flex gap-3 rounded-xl border border-border bg-card p-3 transition-colors hover:bg-accent"
                >
                  <img
                    src={rec.posterUrl}
                    alt={rec.title}
                    className="h-20 w-14 shrink-0 rounded-lg object-cover shadow-sm"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold leading-tight">
                      {rec.title}
                    </p>
                    <p className="mt-0.5 text-xs text-muted-foreground">
                      {rec.year}
                    </p>
                    <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
                      {rec.reason}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
      </div>
    </div>
  );
}
