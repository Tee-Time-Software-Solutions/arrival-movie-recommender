import { useState, useEffect, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Star, Bookmark } from "lucide-react";
import type { MovieDetails } from "@/types/movie";
import { MovieCard } from "@/components/features/MovieCard/MovieCard";
import { MovieDetail } from "@/components/features/MovieDetail/MovieDetail";
import { getWatchlist, removeFromWatchlist } from "@/services/api/watchlist";
import { registerSwipe } from "@/services/api/movies";

export function WatchlistPage() {
  const [movies, setMovies] = useState<MovieDetails[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [ratingMovie, setRatingMovie] = useState<MovieDetails | null>(null);
  const [selectedMovie, setSelectedMovie] = useState<MovieDetails | null>(null);

  const fetchWatchlist = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getWatchlist(50, 0);
      setMovies(data.items);
      setTotal(data.total);
    } catch {
      setError("Failed to load watchlist.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWatchlist();
  }, [fetchWatchlist]);

  const handleRemoveMovie = (movieId: number) => {
    setMovies((prev) => prev.filter((m) => m.movie_db_id !== movieId));
    setTotal((prev) => prev - 1);
  };

  const handleSwipeFeedback = (
    movie: MovieDetails,
    actionType: "like" | "dislike",
    isSupercharged: boolean = false,
  ) => {
    handleRemoveMovie(movie.movie_db_id);
    setRatingMovie(null);
    registerSwipe(movie.movie_db_id, actionType, isSupercharged).catch(console.error);
    removeFromWatchlist(movie.movie_db_id).catch(console.error);
  };

  if (loading) {
    return (
      <div className="flex h-[80vh] items-center justify-center">
        <div className="text-center">
          <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          <p className="text-sm text-muted-foreground">Loading your watchlist...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-[80vh] items-center justify-center px-6">
        <div className="text-center">
          <p className="mb-4 text-sm text-destructive">{error}</p>
          <button
            onClick={fetchWatchlist}
            className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-4 pb-20 pt-10 md:pb-4">
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Watchlist</h1>
        <p className="text-sm text-muted-foreground">
          {total} {total === 1 ? "movie" : "movies"} saved
        </p>
      </div>

      {movies.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20">
          <Bookmark className="mb-4 h-12 w-12 text-muted-foreground/40" />
          <p className="text-muted-foreground">Your watchlist is empty.</p>
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-7">
          {movies.map((movie) => (
            <div key={movie.movie_db_id} className="flex flex-col items-center">
              <button
                onClick={() => setSelectedMovie(movie)}
                className="group relative overflow-hidden rounded-lg"
              >
                <img
                  src={movie.poster_url}
                  alt={movie.title}
                  className="h-44 w-28 object-cover transition group-hover:scale-105 sm:h-48 sm:w-32"
                />
                <div className="absolute inset-x-0 bottom-0 flex items-center justify-center gap-0.5 bg-black/70 px-1 py-0.5">
                  <Star className="h-3 w-3 fill-yellow-500 text-yellow-500" />
                  <span className="text-[10px] text-white">{movie.rating.toFixed(1)}</span>
                </div>
              </button>
              <p className="mt-1 w-28 truncate text-center text-[11px] text-muted-foreground sm:w-32">
                {movie.title}
              </p>
              <button
                onClick={() => setRatingMovie(movie)}
                className="mt-1 rounded-md bg-violet-500/15 px-3 py-1 text-[11px] font-medium text-violet-500 transition hover:bg-violet-500/25"
              >
                Watched
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Movie detail overlay */}
      <AnimatePresence>
        {selectedMovie && (
          <MovieDetail
            movie={selectedMovie}
            onClose={() => setSelectedMovie(null)}
          />
        )}
      </AnimatePresence>

      {/* Swipe rating overlay */}
      <AnimatePresence>
        {ratingMovie && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setRatingMovie(null)}
          >
            <motion.div
              className="relative h-[70vh] w-[min(90vw,380px)]"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              <p className="mb-3 text-center text-sm font-medium text-white">
                How did you like it? Swipe to rate!
              </p>
              <div className="relative h-full">
                <MovieCard
                  movie={ratingMovie}
                  isTop={true}
                  onLike={() => handleSwipeFeedback(ratingMovie, "like")}
                  onDislike={() => handleSwipeFeedback(ratingMovie, "dislike")}
                  onSuperLike={() => handleSwipeFeedback(ratingMovie, "like", true)}
                  onSuperDislike={() => handleSwipeFeedback(ratingMovie, "dislike", true)}
                  onExpand={() => setSelectedMovie(ratingMovie)}
                  onWatched={() => setRatingMovie(null)}
                />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
