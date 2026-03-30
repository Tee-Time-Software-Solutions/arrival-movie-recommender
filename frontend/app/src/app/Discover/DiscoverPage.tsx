import { useEffect, useCallback } from "react";
import { SwipeDeck } from "@/components/features/SwipeDeck/SwipeDeck";
import { useMovieStore } from "@/stores/movieStore";
import { useAuthStore } from "@/stores/authStore";
import {
  getMovieQueue,
  registerSwipe,
} from "@/services/api/movies";
import type { MovieDetails } from "@/types/movie";
import { QUEUE_PREFETCH_THRESHOLD, QUEUE_BATCH_SIZE } from "@/lib/constants";

export function DiscoverPage() {
  const {
    queue,
    currentIndex,
    loading,
    error,
    addToQueue,
    nextMovie,
    setLoading,
    setError,
  } = useMovieStore();

  const { user, loading: authLoading } = useAuthStore();

  const fetchMovies = useCallback(async () => {
    if (loading) return;
    setLoading(true);
    setError(null);

    try {
      const movies = await getMovieQueue(QUEUE_BATCH_SIZE);
      addToQueue(movies);
    } catch {
      setError("Failed to load movies. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [loading, addToQueue, setLoading, setError]);

  // Initial load — wait for auth to resolve first
  useEffect(() => {
    if (!authLoading && user && queue.length === 0) {
      fetchMovies();
    }
  }, [authLoading, user]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Prefetch when running low
  useEffect(() => {
    const remaining = queue.length - currentIndex;
    if (remaining <= QUEUE_PREFETCH_THRESHOLD && !loading) {
      fetchMovies();
    }
  }, [currentIndex, queue.length, loading, fetchMovies]);

  const handleLike = async (movie: MovieDetails) => {
    nextMovie();
    registerSwipe(movie.movie_db_id, "like").catch(console.error);
  };

  const handleDislike = async (movie: MovieDetails) => {
    nextMovie();
    registerSwipe(movie.movie_db_id, "dislike").catch(console.error);
  };

  const handleWatched = async (movie: MovieDetails) => {
    nextMovie();
    registerSwipe(movie.movie_db_id, "skip").catch(console.error);
  };

  const handleSuperLike = async (movie: MovieDetails) => {
    nextMovie();
    registerSwipe(movie.movie_db_id, "like", true).catch(console.error);
  };

  const handleSuperDislike = async (movie: MovieDetails) => {
    nextMovie();
    registerSwipe(movie.movie_db_id, "dislike", true).catch(console.error);
  };

  const currentMovie = queue[currentIndex];

  if (loading && !currentMovie) {
    return (
      <div className="flex h-[80vh] items-center justify-center">
        <div className="text-center">
          <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          <p className="text-sm text-muted-foreground">
            Loading your movie feed...
          </p>
        </div>
      </div>
    );
  }

  if (error && queue.length === 0) {
    return (
      <div className="flex h-[80vh] items-center justify-center px-6">
        <div className="text-center">
          <p className="mb-4 text-sm text-destructive">{error}</p>
          <button
            onClick={fetchMovies}
            className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex flex-1 flex-col items-center justify-center px-4 pb-20 md:pb-4">
      <SwipeDeck
        movies={queue}
        currentIndex={currentIndex}
        onLike={handleLike}
        onDislike={handleDislike}
        onWatched={handleWatched}
        onSuperLike={handleSuperLike}
        onSuperDislike={handleSuperDislike}
      />
    </div>
  );
}
