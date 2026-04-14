import { useState, useEffect, useCallback } from "react";
import { SlidersHorizontal } from "lucide-react";
import { SwipeDeck } from "@/components/features/SwipeDeck/SwipeDeck";
import { FilterPanel } from "@/components/features/FilterPanel/FilterPanel";
import { useMovieStore } from "@/stores/movieStore";
import { useAuthStore } from "@/stores/authStore";
import {
  getMovieQueue,
  registerSwipe,
  flushMovieFeed,
} from "@/services/api/movies";
import { getProfileSummary, updatePreferences } from "@/services/api/user";
import type { MovieDetails } from "@/types/movie";
import type { UserPreferences } from "@/types/user";
import { QUEUE_PREFETCH_THRESHOLD, QUEUE_BATCH_SIZE } from "@/lib/constants";

export function DiscoverPage() {
  const {
    queue,
    currentIndex,
    loading,
    error,
    exhausted,
    addToQueue,
    nextMovie,
    resetQueue,
    setLoading,
    setError,
    setExhausted,
  } = useMovieStore();

  const [showFilters, setShowFilters] = useState(false);
  const [preferences, setPreferences] = useState<UserPreferences | null>(null);

  const { user, loading: authLoading } = useAuthStore();

  const fetchMovies = useCallback(async () => {
    if (loading || exhausted) return;
    setLoading(true);
    setError(null);

    try {
      const movies = await getMovieQueue(QUEUE_BATCH_SIZE);
      addToQueue(movies);
    } catch (err: unknown) {
      const status = (err as { response?: { status?: number } })?.response?.status;
      if (status === 404) {
        setExhausted(true);
      } else {
        setError("Failed to load movies. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }, [loading, exhausted, addToQueue, setLoading, setError, setExhausted]);

  // Load current preferences on mount
  useEffect(() => {
    if (!authLoading && user) {
      getProfileSummary(user.uid)
        .then((summary) => setPreferences(summary.preferences))
        .catch(() => {});
    }
  }, [authLoading, user]);

  const handleApplyFilters = useCallback(
    async (prefs: UserPreferences) => {
      if (!user) return;
      setPreferences(prefs);
      setShowFilters(false);
      await updatePreferences(user.uid, prefs).catch(console.error);
      await flushMovieFeed().catch(console.error);
      resetQueue();
      // fetchMovies will be triggered by the prefetch effect when queue.length drops to 0
    },
    [user, resetQueue],
  );

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

  const handleSkipped = async (movie: MovieDetails) => {
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

  const hasActiveFilters =
    preferences &&
    (preferences.included_genres.length > 0 ||
      preferences.excluded_genres.length > 0 ||
      preferences.min_release_year !== null ||
      preferences.max_release_year !== null ||
      preferences.min_rating !== null ||
      preferences.include_adult ||
      preferences.movie_providers.length > 0);

  if (exhausted && !currentMovie) {
    return (
      <div className="relative flex h-[80vh] flex-col items-center justify-center px-6">
        <button
          onClick={() => setShowFilters((prev) => !prev)}
          className={`absolute right-4 top-4 z-40 rounded-full p-2 shadow-md transition-colors ${
            hasActiveFilters
              ? "bg-primary text-primary-foreground"
              : "bg-card text-muted-foreground hover:bg-accent"
          }`}
        >
          <SlidersHorizontal size={18} />
        </button>
        {showFilters && (
          <FilterPanel
            currentPreferences={preferences}
            onApply={handleApplyFilters}
            onClose={() => setShowFilters(false)}
          />
        )}
        <div className="text-center">
          <p className="mb-2 text-sm font-medium text-foreground">
            No more movies to show
          </p>
          <p className="mb-4 text-xs text-muted-foreground">
            {hasActiveFilters
              ? "Try broadening your filters to discover more movies."
              : "You've seen everything we have! Check back later."}
          </p>
          {hasActiveFilters && (
            <button
              onClick={() => setShowFilters(true)}
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
            >
              Adjust Filters
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex flex-1 flex-col items-center justify-center px-4 pb-20 md:pb-4">
      <button
        onClick={() => setShowFilters((prev) => !prev)}
        className={`absolute right-4 top-4 z-40 rounded-full p-2 shadow-md transition-colors ${
          hasActiveFilters
            ? "bg-primary text-primary-foreground"
            : "bg-card text-muted-foreground hover:bg-accent"
        }`}
      >
        <SlidersHorizontal size={18} />
      </button>

      {showFilters && (
        <FilterPanel
          currentPreferences={preferences}
          onApply={handleApplyFilters}
          onClose={() => setShowFilters(false)}
        />
      )}

      <SwipeDeck
        movies={queue}
        currentIndex={currentIndex}
        onLike={handleLike}
        onDislike={handleDislike}
        onSkipped={handleSkipped}
        onSuperLike={handleSuperLike}
        onSuperDislike={handleSuperDislike}
      />
    </div>
  );
}
