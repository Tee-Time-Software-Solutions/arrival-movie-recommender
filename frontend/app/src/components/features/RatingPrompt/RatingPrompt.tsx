import { useState } from "react";
import { motion } from "framer-motion";
import { Star, X } from "lucide-react";
import type { MovieDetails } from "@/types/movie";
import { cn } from "@/lib/utils";

interface RatingPromptProps {
  movie: MovieDetails;
  onSubmit: (rating: number) => void;
  onClose: () => void;
}

export function RatingPrompt({ movie, onSubmit, onClose }: RatingPromptProps) {
  const [selectedRating, setSelectedRating] = useState(0);
  const [hoveredRating, setHoveredRating] = useState(0);

  const displayRating = hoveredRating || selectedRating;

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <motion.div
        className="relative z-10 w-full max-w-sm rounded-2xl bg-card p-6 shadow-xl"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
      >
        <button
          onClick={onClose}
          className="absolute right-4 top-4 rounded-full p-1 hover:bg-secondary"
        >
          <X className="h-4 w-4" />
        </button>

        <h3 className="mb-1 text-lg font-bold">Rate this movie</h3>
        <p className="mb-6 text-sm text-muted-foreground">{movie.title}</p>

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

        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 rounded-lg bg-secondary px-4 py-2.5 text-sm font-medium transition hover:bg-secondary/80"
          >
            Skip
          </button>
          <button
            onClick={() => selectedRating > 0 && onSubmit(selectedRating)}
            disabled={selectedRating === 0}
            className="flex-1 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:opacity-50"
          >
            Submit
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}
