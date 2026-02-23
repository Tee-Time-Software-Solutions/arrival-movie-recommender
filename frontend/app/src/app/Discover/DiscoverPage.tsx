import { useEffect, useCallback, useRef } from "react";
import { SwipeDeck } from "@/components/features/SwipeDeck/SwipeDeck";
import { useMovieStore } from "@/stores/movieStore";
import {
  getMovieQueue,
  registerSwipe,
  rateMovie,
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
    likeMovie,
    dislikeMovie,
    setLoading,
    setError,
  } = useMovieStore();

  const fetchingRef = useRef(false);

  const fetchMovies = useCallback(async () => {
    if (fetchingRef.current) return;
    fetchingRef.current = true;
    setLoading(true);
    setError(null);

    try {
      const movies = await getMovieQueue(QUEUE_BATCH_SIZE);
      addToQueue(movies);
    } catch {
      setError("Failed to load movies. Please try again.");
    } finally {
      setLoading(false);
      fetchingRef.current = false;
    }
  }, [addToQueue, setLoading, setError]);

  // Initial load
  useEffect(() => {
    if (queue.length === 0) {
      fetchMovies();
    }
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  // Prefetch when running low
  useEffect(() => {
    const remaining = queue.length - currentIndex;
    if (remaining < QUEUE_PREFETCH_THRESHOLD && remaining > 0) {
      fetchMovies();
    }
  }, [currentIndex, queue.length, fetchMovies]);

  const handleLike = async (movie: MovieDetails) => {
    likeMovie(movie);
    nextMovie();
    // Optimistic: fire and forget
    registerSwipe(movie.movie_id, "like").catch(console.error);
    rateMovie(movie.movie_id, 4).catch(console.error);
  };

  const handleDislike = async (movie: MovieDetails) => {
    dislikeMovie(movie);
    nextMovie();
    registerSwipe(movie.movie_id, "dislike").catch(console.error);
  };

  const handleWatched = async (movie: MovieDetails) => {
    nextMovie();
    registerSwipe(movie.movie_id, "skip").catch(console.error);
  };

  const handleSuperLike = async (movie: MovieDetails) => {
    likeMovie(movie);
    nextMovie();
    registerSwipe(movie.movie_id, "like", true).catch(console.error);
    rateMovie(movie.movie_id, 5).catch(console.error);
  };

  const handleSuperDislike = async (movie: MovieDetails) => {
    dislikeMovie(movie);
    nextMovie();
    registerSwipe(movie.movie_id, "dislike", true).catch(console.error);
  };

  if (loading && queue.length === 0) {
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
