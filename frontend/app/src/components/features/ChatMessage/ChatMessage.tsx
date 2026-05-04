import { useState } from "react";
import { cn } from "@/lib/utils";
import { Popcorn, User, Star, AlertCircle, RotateCw } from "lucide-react";
import { AnimatePresence } from "framer-motion";
import type { ChatMessage as ChatMessageType } from "@/types/chat";
import type { MovieDetails } from "@/types/movie";
import { MovieDetail } from "@/components/features/MovieDetail/MovieDetail";

interface ChatMessageProps {
  message: ChatMessageType;
  onRetry?: () => void;
}

export function ChatMessage({ message, onRetry }: ChatMessageProps) {
  const isUser = message.role === "user";
  const isError = message.status === "error";
  const [selectedMovie, setSelectedMovie] = useState<MovieDetails | null>(null);

  return (
    <>
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
            isUser
              ? "bg-foreground/10"
              : isError
                ? "bg-destructive/10"
                : "bg-primary/10",
          )}
        >
          {isUser ? (
            <User className="h-4 w-4 text-foreground/70" />
          ) : isError ? (
            <AlertCircle className="h-4 w-4 text-destructive" />
          ) : (
            <Popcorn className="h-4 w-4 text-primary" />
          )}
        </div>

        {/* Content */}
        <div className={cn("min-w-0 max-w-full flex-1", isUser && "flex flex-col items-end")}>
          <p className="mb-1 text-xs font-medium text-muted-foreground">
            {isUser ? "You" : isError ? "Connection error" : "Arrival"}
          </p>
          {isError ? (
            <div className="rounded-xl border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
              <p className="whitespace-pre-wrap">{message.content}</p>
              {onRetry && (
                <button
                  type="button"
                  onClick={onRetry}
                  className="mt-2 inline-flex items-center gap-1.5 rounded-lg border border-destructive/40 bg-background px-2.5 py-1 text-xs font-medium text-destructive transition-colors hover:bg-destructive/10"
                >
                  <RotateCw className="h-3 w-3" />
                  Retry
                </button>
              )}
            </div>
          ) : (
            <div
              className={cn(
                "prose prose-sm max-w-none text-foreground",
                "prose-p:leading-relaxed prose-p:my-1",
                isUser && "text-right",
              )}
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
            </div>
          )}

          {/* Movie Recommendations */}
          {message.movieRecommendations &&
            message.movieRecommendations.length > 0 && (
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {message.movieRecommendations.map((movie) => (
                  <button
                    key={movie.movie_db_id}
                    type="button"
                    onClick={() => setSelectedMovie(movie)}
                    className="group flex gap-3 rounded-xl border border-border bg-card p-3 text-left transition-colors hover:bg-accent cursor-pointer"
                  >
                    {movie.poster_url && (
                      <img
                        src={movie.poster_url}
                        alt={movie.title}
                        className="h-24 w-16 shrink-0 rounded-lg object-cover shadow-sm"
                      />
                    )}
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-semibold leading-tight">
                        {movie.title}
                      </p>
                      <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
                        <span>{movie.release_year}</span>
                        {movie.rating > 0 && (
                          <span className="flex items-center gap-0.5">
                            <Star className="h-3 w-3 fill-yellow-500 text-yellow-500" />
                            {movie.rating.toFixed(1)}
                          </span>
                        )}
                      </div>
                      {movie.genres && movie.genres.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-1">
                          {movie.genres.slice(0, 2).map((genre) => (
                            <span
                              key={genre}
                              className="rounded-full bg-secondary px-2 py-0.5 text-[10px]"
                            >
                              {genre}
                            </span>
                          ))}
                        </div>
                      )}
                      {movie.synopsis && (
                        <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
                          {movie.synopsis}
                        </p>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
        </div>
      </div>

      {/* Movie Detail Modal */}
      <AnimatePresence>
        {selectedMovie && (
          <MovieDetail
            movie={selectedMovie}
            onClose={() => setSelectedMovie(null)}
          />
        )}
      </AnimatePresence>
    </>
  );
}
