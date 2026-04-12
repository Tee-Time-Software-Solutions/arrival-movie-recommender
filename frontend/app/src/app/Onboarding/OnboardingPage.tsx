import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Search, X, Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/stores/authStore";
import {
  getOnboardingMovies,
  searchOnboardingMovies,
  completeOnboarding,
} from "@/services/api/onboarding";
import type {
  OnboardingMovieCard,
  OnboardingSearchResult,
} from "@/types/onboarding";

const MIN_GRID_PICKS = 5;
const MIN_SEARCH_PICKS = 1;
const MAX_SEARCH_PICKS = 3;

export function OnboardingPage() {
  const navigate = useNavigate();
  const { setNeedsOnboarding } = useAuthStore();

  const [step, setStep] = useState<1 | 2>(1);

  // Step 1 state
  const [gridMovies, setGridMovies] = useState<OnboardingMovieCard[]>([]);
  const [selectedGridIds, setSelectedGridIds] = useState<Set<number>>(
    new Set(),
  );
  const [gridLoading, setGridLoading] = useState(true);

  // Step 2 state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<OnboardingSearchResult[]>(
    [],
  );
  const [selectedSearchMovies, setSelectedSearchMovies] = useState<
    OnboardingSearchResult[]
  >([]);
  const [searching, setSearching] = useState(false);

  const [submitting, setSubmitting] = useState(false);

  // Load grid movies on mount
  useEffect(() => {
    getOnboardingMovies()
      .then(setGridMovies)
      .catch(console.error)
      .finally(() => setGridLoading(false));
  }, []);

  // Debounced TMDB search
  useEffect(() => {
    if (step !== 2 || searchQuery.trim().length < 2) {
      setSearchResults([]);
      return;
    }

    const timeout = setTimeout(async () => {
      setSearching(true);
      try {
        const results = await searchOnboardingMovies(searchQuery.trim());
        setSearchResults(results);
      } catch {
        setSearchResults([]);
      } finally {
        setSearching(false);
      }
    }, 400);

    return () => clearTimeout(timeout);
  }, [searchQuery, step]);

  const toggleGridMovie = useCallback((movieDbId: number) => {
    setSelectedGridIds((prev) => {
      const next = new Set(prev);
      if (next.has(movieDbId)) {
        next.delete(movieDbId);
      } else {
        next.add(movieDbId);
      }
      return next;
    });
  }, []);

  const addSearchMovie = useCallback(
    (movie: OnboardingSearchResult) => {
      if (selectedSearchMovies.length >= MAX_SEARCH_PICKS) return;
      if (selectedSearchMovies.some((m) => m.tmdb_id === movie.tmdb_id)) return;
      setSelectedSearchMovies((prev) => [...prev, movie]);
      setSearchQuery("");
      setSearchResults([]);
    },
    [selectedSearchMovies],
  );

  const removeSearchMovie = useCallback((tmdbId: number) => {
    setSelectedSearchMovies((prev) =>
      prev.filter((m) => m.tmdb_id !== tmdbId),
    );
  }, []);

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      const searchDbIds = selectedSearchMovies
        .map((m) => m.movie_db_id)
        .filter((id): id is number => id !== null);
      const allMovieDbIds = [...Array.from(selectedGridIds), ...searchDbIds];
      await completeOnboarding(allMovieDbIds);
      setNeedsOnboarding(false);
      navigate("/", { replace: true });
    } catch (err) {
      console.error("Onboarding failed:", err);
    } finally {
      setSubmitting(false);
    }
  };

  const canProceedStep1 = selectedGridIds.size >= MIN_GRID_PICKS;
  const canSubmit = selectedSearchMovies.length >= MIN_SEARCH_PICKS;

  return (
    <div className="flex min-h-screen flex-col">
      {/* Header */}
      <div className="sticky top-0 z-10 border-b border-border bg-background/95 px-6 py-4 backdrop-blur">
        <div className="mx-auto max-w-3xl">
          <h1 className="text-2xl font-bold tracking-tight">
            {step === 1
              ? "Pick movies you like"
              : "Search your favorites"}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {step === 1
              ? `Select at least ${MIN_GRID_PICKS} movies to personalize your recommendations`
              : `Add ${MIN_SEARCH_PICKS}-${MAX_SEARCH_PICKS} movies you love`}
          </p>
          {/* Step indicator */}
          <div className="mt-3 flex gap-2">
            <div
              className={cn(
                "h-1 flex-1 rounded-full",
                step >= 1 ? "bg-primary" : "bg-muted",
              )}
            />
            <div
              className={cn(
                "h-1 flex-1 rounded-full",
                step >= 2 ? "bg-primary" : "bg-muted",
              )}
            />
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="mx-auto w-full max-w-3xl flex-1 px-4 py-6">
        <AnimatePresence mode="wait">
          {step === 1 ? (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
            >
              {gridLoading ? (
                <div className="flex items-center justify-center py-20">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-5">
                  {gridMovies.map((movie) => {
                    const selected = selectedGridIds.has(movie.movie_db_id);
                    return (
                      <motion.button
                        key={movie.movie_db_id}
                        onClick={() => toggleGridMovie(movie.movie_db_id)}
                        whileTap={{ scale: 0.95 }}
                        className={cn(
                          "group relative overflow-hidden rounded-xl transition-all",
                          selected
                            ? "ring-3 ring-primary ring-offset-2 ring-offset-background"
                            : "hover:ring-2 hover:ring-muted-foreground/30",
                        )}
                      >
                        <img
                          src={movie.poster_url}
                          alt={movie.title}
                          className="aspect-[2/3] w-full object-cover"
                          loading="lazy"
                        />
                        {/* Selected overlay */}
                        <AnimatePresence>
                          {selected && (
                            <motion.div
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              exit={{ opacity: 0 }}
                              className="absolute inset-0 flex items-center justify-center bg-primary/30"
                            >
                              <div className="rounded-full bg-primary p-2">
                                <Check className="h-5 w-5 text-primary-foreground" />
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                        {/* Title on hover */}
                        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-2 opacity-0 transition-opacity group-hover:opacity-100">
                          <p className="text-xs font-medium text-white">
                            {movie.title}
                          </p>
                        </div>
                      </motion.button>
                    );
                  })}
                </div>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="step2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.2 }}
            >
              {/* Search input */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search for a movie..."
                  className="w-full rounded-xl border border-border bg-card py-3 pl-10 pr-4 text-sm outline-none transition focus:ring-2 focus:ring-primary"
                  autoFocus
                />
              </div>

              {/* Search results dropdown */}
              {(searchResults.length > 0 || searching) && (
                <div className="mt-2 rounded-xl border border-border bg-card shadow-lg">
                  {searching ? (
                    <div className="flex items-center justify-center py-4">
                      <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                    </div>
                  ) : (
                    searchResults.map((result) => {
                      const alreadyAdded = selectedSearchMovies.some(
                        (m) => m.tmdb_id === result.tmdb_id,
                      );
                      return (
                        <button
                          key={result.tmdb_id}
                          onClick={() => addSearchMovie(result)}
                          disabled={
                            alreadyAdded ||
                            selectedSearchMovies.length >= MAX_SEARCH_PICKS
                          }
                          className="flex w-full items-center gap-3 border-b border-border px-4 py-3 text-left transition last:border-0 hover:bg-muted/50 disabled:opacity-50"
                        >
                          {result.poster_url ? (
                            <img
                              src={result.poster_url}
                              alt={result.title}
                              className="h-12 w-8 rounded object-cover"
                            />
                          ) : (
                            <div className="flex h-12 w-8 items-center justify-center rounded bg-muted text-xs text-muted-foreground">
                              ?
                            </div>
                          )}
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-sm font-medium">
                              {result.title}
                            </p>
                            {result.release_year && (
                              <p className="text-xs text-muted-foreground">
                                {result.release_year}
                              </p>
                            )}
                          </div>
                          {alreadyAdded && (
                            <Check className="h-4 w-4 text-primary" />
                          )}
                        </button>
                      );
                    })
                  )}
                </div>
              )}

              {/* Selected search movies */}
              {selectedSearchMovies.length > 0 && (
                <div className="mt-6">
                  <p className="mb-3 text-sm font-medium text-muted-foreground">
                    Your favorites ({selectedSearchMovies.length}/
                    {MAX_SEARCH_PICKS})
                  </p>
                  <div className="flex flex-wrap gap-3">
                    {selectedSearchMovies.map((movie) => (
                      <motion.div
                        key={movie.tmdb_id}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="flex items-center gap-2 rounded-xl bg-card px-3 py-2 shadow-sm"
                      >
                        {movie.poster_url && (
                          <img
                            src={movie.poster_url}
                            alt={movie.title}
                            className="h-10 w-7 rounded object-cover"
                          />
                        )}
                        <div>
                          <p className="text-sm font-medium">{movie.title}</p>
                          {movie.release_year && (
                            <p className="text-xs text-muted-foreground">
                              {movie.release_year}
                            </p>
                          )}
                        </div>
                        <button
                          onClick={() => removeSearchMovie(movie.tmdb_id)}
                          className="ml-1 rounded-full p-1 hover:bg-muted"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Bottom action bar */}
      <div className="sticky bottom-0 border-t border-border bg-background/95 px-6 py-4 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between">
          {step === 1 ? (
            <>
              <p className="text-sm text-muted-foreground">
                {selectedGridIds.size} selected
              </p>
              <button
                onClick={() => setStep(2)}
                disabled={!canProceedStep1}
                className="rounded-xl bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground transition hover:bg-primary/90 disabled:opacity-50"
              >
                Next
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => setStep(1)}
                className="text-sm text-muted-foreground hover:text-foreground"
              >
                Back
              </button>
              <button
                onClick={handleSubmit}
                disabled={!canSubmit || submitting}
                className="rounded-xl bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground transition hover:bg-primary/90 disabled:opacity-50"
              >
                {submitting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  "Get Started"
                )}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
