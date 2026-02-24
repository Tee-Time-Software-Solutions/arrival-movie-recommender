import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { Star, ChevronDown } from "lucide-react";
import { MovieCard } from "@/components/features/MovieCard/MovieCard";
import { MovieDetailContent } from "@/components/features/MovieDetail/MovieDetail";
import type { MovieDetails } from "@/types/movie";

type DeckMode = "card" | "details";

const SPRING = { type: "spring" as const, stiffness: 300, damping: 30 };

interface SwipeDeckProps {
  movies: MovieDetails[];
  currentIndex: number;
  onLike: (movie: MovieDetails) => void;
  onDislike: (movie: MovieDetails) => void;
  onWatched: (movie: MovieDetails) => void;
  onSuperLike: (movie: MovieDetails) => void;
  onSuperDislike: (movie: MovieDetails) => void;
}

export function SwipeDeck({
  movies,
  currentIndex,
  onLike,
  onDislike,
  onWatched,
  onSuperLike,
  onSuperDislike,
}: SwipeDeckProps) {
  const [mode, setMode] = useState<DeckMode>("card");
  const [forceSwipe, setForceSwipe] = useState<"left" | "right" | "down" | null>(null);

  const currentMovie = movies[currentIndex];

  // Reset when movie changes
  useEffect(() => {
    setMode("card");
    setForceSwipe(null);
  }, [currentIndex]);

  // Trigger like/dislike/watched after swipe animation plays
  const triggerSwipe = useCallback(
    (direction: "left" | "right" | "down") => {
      if (forceSwipe || !currentMovie) return;
      setForceSwipe(direction);
      const movie = currentMovie;
      setTimeout(() => {
        setForceSwipe(null);
        if (direction === "right") onLike(movie);
        else if (direction === "left") onDislike(movie);
        else onWatched(movie);
      }, 300);
    },
    [forceSwipe, currentMovie, onLike, onDislike, onWatched],
  );

  // Keyboard support
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (mode === "details") {
        if (e.key === "ArrowDown" || e.key === "Escape") {
          e.preventDefault();
          setMode("card");
        }
        return;
      }

      // mode === 'card'
      if (!currentMovie) return;

      switch (e.key) {
        case "ArrowRight":
          e.preventDefault();
          triggerSwipe("right");
          break;
        case "ArrowLeft":
          e.preventDefault();
          triggerSwipe("left");
          break;
        case "ArrowUp":
          e.preventDefault();
          setMode("details");
          break;
        case "ArrowDown":
          e.preventDefault();
          triggerSwipe("down");
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [mode, currentMovie, triggerSwipe]);

  if (!currentMovie) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="text-center">
          <p className="text-lg text-muted-foreground">No more movies!</p>
          <p className="mt-2 text-sm text-muted-foreground">
            Check back later for more recommendations.
          </p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="relative mx-auto h-[70vh] w-full max-w-sm">
        {/* Full card — default mode */}
        <motion.div
          className="absolute inset-0"
          animate={{
            opacity: mode === "card" ? 1 : 0,
            scale: mode === "card" ? 1 : 0.95,
          }}
          transition={SPRING}
          style={{ pointerEvents: mode === "card" ? "auto" : "none" }}
        >
          <MovieCard
            key={currentMovie.movie_id}
            movie={currentMovie}
            isTop={true}
            onLike={() => onLike(currentMovie)}
            onDislike={() => onDislike(currentMovie)}
            onExpand={() => setMode("details")}
            onWatched={() => onWatched(currentMovie)}
            onSuperLike={() => onSuperLike(currentMovie)}
            onSuperDislike={() => onSuperDislike(currentMovie)}
            forceSwipe={forceSwipe}
          />
        </motion.div>
      </div>

      {/* Hint text — below the card */}
      {mode === "card" && (
        <div className="mt-4 text-center text-xs text-muted-foreground">
          <p>
            <span className="mr-3">← Dislike</span>
            <span className="mr-3">↑ Details</span>
            <span className="mr-3">↓ Watched</span>
            <span className="mr-3">Like →</span>
            <span>Hold + Swipe = Super</span>
          </p>
        </div>
      )}

      {/* Details overlay — full screen, slides up */}
      <motion.div
        className="absolute inset-0 z-50 flex flex-col"
        initial={false}
        animate={{ y: mode === "details" ? 0 : "100%", opacity: mode === "details" ? 1 : 0 }}
        transition={SPRING}
        style={{ pointerEvents: mode === "details" ? "auto" : "none" }}
      >
        <div className="flex flex-1 flex-col bg-background">
          {/* Sticky header */}
          <div
            className="mx-auto flex w-full max-w-3xl shrink-0 cursor-pointer items-center gap-3 border-b border-border px-6 py-4"
            onClick={() => setMode("card")}
          >
            <img
              src={currentMovie.poster_url}
              alt={currentMovie.title}
              className="h-14 w-10 shrink-0 rounded-lg object-cover"
            />
            <div className="min-w-0 flex-1">
              <h3 className="truncate text-sm font-bold">{currentMovie.title}</h3>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>{currentMovie.release_year}</span>
                <span className="flex items-center gap-0.5">
                  <Star className="h-3 w-3 fill-yellow-500 text-yellow-500" />
                  {(currentMovie.rating ?? 0).toFixed(1)}
                </span>
                <span>{currentMovie.runtime} min</span>
              </div>
            </div>
            <ChevronDown className="h-5 w-5 shrink-0 text-muted-foreground" />
          </div>

          {/* Scrollable content */}
          <div className="flex-1 overflow-y-auto">
            <div className="mx-auto max-w-3xl">
              <MovieDetailContent movie={currentMovie} />
            </div>
          </div>
        </div>
      </motion.div>
    </>
  );
}
