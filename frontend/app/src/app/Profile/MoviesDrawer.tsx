import { motion } from "framer-motion";
import { X, Star } from "lucide-react";
import type { MovieDetails } from "@/types/movie";

interface MoviesDrawerProps {
  title: string;
  movies: MovieDetails[];
  loading: boolean;
  onClose: () => void;
  onMovieClick: (movie: MovieDetails) => void;
}

export function MoviesDrawer({
  title,
  movies,
  loading,
  onClose,
  onMovieClick,
}: MoviesDrawerProps) {
  return (
    <motion.div
      className="fixed inset-0 left-[72px] z-40 flex items-center justify-center md:left-[72px] max-md:left-0"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <motion.div
        className="relative z-10 h-full w-full overflow-y-auto bg-card"
        initial={{ y: "100%" }}
        animate={{ y: 0 }}
        exit={{ y: "100%" }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
      >
        <div className="mx-auto max-w-3xl">
          <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-card p-4">
            <h2 className="text-lg font-bold">{title}</h2>
            <button
              onClick={onClose}
              className="rounded-full p-1 hover:bg-secondary"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <div className="p-4">
            {loading ? (
              <div className="flex justify-center py-12">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              </div>
            ) : movies.length > 0 ? (
              <div className="grid grid-cols-4 gap-3 sm:grid-cols-5 md:grid-cols-6 lg:grid-cols-8">
                {movies.map((movie) => (
                  <button
                    key={movie.movie_db_id}
                    onClick={() => onMovieClick(movie)}
                    className="group flex flex-col items-center"
                  >
                    <div className="relative overflow-hidden rounded-lg">
                      <img
                        src={movie.poster_url}
                        alt={movie.title}
                        className="h-36 w-24 object-cover transition group-hover:scale-105"
                      />
                      <div className="absolute inset-x-0 bottom-0 flex items-center justify-center gap-0.5 bg-black/70 px-1 py-0.5">
                        <Star className="h-3 w-3 fill-yellow-500 text-yellow-500" />
                        <span className="text-[10px] text-white">
                          {movie.rating.toFixed(1)}
                        </span>
                      </div>
                    </div>
                    <p className="mt-1 w-24 truncate text-center text-[11px] text-muted-foreground">
                      {movie.title}
                    </p>
                  </button>
                ))}
              </div>
            ) : (
              <p className="py-8 text-center text-sm text-muted-foreground">
                No movies yet.
              </p>
            )}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
