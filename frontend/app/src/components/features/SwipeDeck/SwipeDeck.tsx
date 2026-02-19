import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { Star, ChevronDown, ChevronUp } from "lucide-react";
import { MovieCard } from "@/components/features/MovieCard/MovieCard";
import { MovieDetailContent } from "@/components/features/MovieDetail/MovieDetail";
import type { MovieDetails } from "@/types/movie";
import { cn } from "@/lib/utils";

type DeckMode = "card" | "details" | "rating";

const SPRING = { type: "spring" as const, stiffness: 300, damping: 30 };

interface SwipeDeckProps {
  movies: MovieDetails[];
  currentIndex: number;
  onLike: (movie: MovieDetails) => void;
  onDislike: (movie: MovieDetails) => void;
  onRate: (movie: MovieDetails, rating: number) => void;
}

export function SwipeDeck({
  movies,
  currentIndex,
  onLike,
  onDislike,
  onRate,
}: SwipeDeckProps) {
  const [mode, setMode] = useState<DeckMode>("card");
  const [selectedRating, setSelectedRating] = useState(0);
  const [hoveredRating, setHoveredRating] = useState(0);
  const [forceSwipe, setForceSwipe] = useState<"left" | "right" | null>(null);

  const currentMovie = movies[currentIndex];

  // Reset when movie changes
  useEffect(() => {
    setMode("card");
    setSelectedRating(0);
    setHoveredRating(0);
    setForceSwipe(null);
  }, [currentIndex]);

  // Trigger like/dislike after swipe animation plays
  const triggerSwipe = useCallback(
    (direction: "left" | "right") => {
      if (forceSwipe || !currentMovie) return;
      setForceSwipe(direction);
      const movie = currentMovie;
      setTimeout(() => {
        setForceSwipe(null);
        if (direction === "right") onLike(movie);
        else onDislike(movie);
      }, 300);
    },
    [forceSwipe, currentMovie, onLike, onDislike],
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

      if (mode === "rating") {
        if (e.key === "ArrowUp" || e.key === "Escape") {
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
          setMode("rating");
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

  const displayRating = hoveredRating || selectedRating;

  const handleSubmitRating = () => {
    if (selectedRating > 0) {
      onRate(currentMovie, selectedRating);
    }
  };

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
            onRate={() => setMode("rating")}
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
            <span className="mr-3">↓ Rate</span>
            <span>Like →</span>
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
                  {currentMovie.rating.toFixed(1)}
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

      {/* Rating overlay — full screen, slides down */}
      <motion.div
        className="absolute inset-0 z-50 flex flex-col"
        initial={false}
        animate={{ y: mode === "rating" ? 0 : "-100%", opacity: mode === "rating" ? 1 : 0 }}
        transition={SPRING}
        style={{ pointerEvents: mode === "rating" ? "auto" : "none" }}
      >
        <div className="flex flex-1 flex-col bg-background">
          <div className="flex flex-1 flex-col items-center justify-center px-6">
            <h3 className="mb-1 text-lg font-bold">Rate this movie</h3>
            <p className="mb-6 text-sm text-muted-foreground">
              {currentMovie.title}
            </p>

            <div className="mb-6 flex justify-center gap-2">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onMouseEnter={() => setHoveredRating(star)}
                  onMouseLeave={() => setHoveredRating(0)}
                  onClick={() => setSelectedRating(star)}
                  className="transition-transform hover:scale-110"
                >
                  <Star
                    className={cn(
                      "h-8 w-8 transition-colors",
                      star <= displayRating
                        ? "fill-yellow-500 text-yellow-500"
                        : "text-muted-foreground",
                    )}
                  />
                </button>
              ))}
            </div>

            <div className="flex w-full max-w-xs gap-3">
              <button
                onClick={() => setMode("card")}
                className="flex-1 rounded-lg bg-secondary px-4 py-2.5 text-sm font-medium transition hover:bg-secondary/80"
              >
                Skip
              </button>
              <button
                onClick={handleSubmitRating}
                disabled={selectedRating === 0}
                className="flex-1 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:opacity-50"
              >
                Submit
              </button>
            </div>
          </div>

          {/* Bottom bar to go back */}
          <div
            className="mx-auto flex w-full max-w-3xl shrink-0 cursor-pointer items-center gap-3 border-t border-border px-6 py-4"
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
                  {currentMovie.rating.toFixed(1)}
                </span>
                <span>{currentMovie.runtime} min</span>
              </div>
            </div>
            <ChevronUp className="h-5 w-5 shrink-0 text-muted-foreground" />
          </div>
        </div>
      </motion.div>
    </>
  );
}
