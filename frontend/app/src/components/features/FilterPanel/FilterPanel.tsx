import { useState, useEffect } from "react";
import { X } from "lucide-react";
import type { UserPreferences } from "@/types/user";
import type { MovieProvider } from "@/types/movie";

const TMDB_GENRES = [
  "Action",
  "Adventure",
  "Animation",
  "Comedy",
  "Crime",
  "Documentary",
  "Drama",
  "Family",
  "Fantasy",
  "History",
  "Horror",
  "Music",
  "Mystery",
  "Romance",
  "Science Fiction",
  "Thriller",
  "War",
  "Western",
] as const;

const STREAMING_PROVIDERS = [
  "Netflix",
  "Amazon Prime Video",
  "Disney Plus",
  "Max",
  "Apple TV Plus",
  "Hulu",
  "Paramount Plus",
  "Peacock",
] as const;

const MIN_YEAR = 1900;
const MAX_YEAR = new Date().getFullYear();
const MIN_RATING = 0;
const MAX_RATING = 10;
const RATING_STEP = 0.5;

interface FilterPanelProps {
  currentPreferences: UserPreferences | null;
  onApply: (prefs: UserPreferences) => void;
  onClose: () => void;
}

export function FilterPanel({
  currentPreferences,
  onApply,
  onClose,
}: FilterPanelProps) {
  const [includedGenres, setIncludedGenres] = useState<string[]>([]);
  const [excludedGenres, setExcludedGenres] = useState<string[]>([]);
  const [minYear, setMinYear] = useState("");
  const [maxYear, setMaxYear] = useState("");
  const [minRating, setMinRating] = useState<number | null>(null);
  const [includeAdult, setIncludeAdult] = useState(false);
  const [selectedProviders, setSelectedProviders] = useState<string[]>([]);

  useEffect(() => {
    if (currentPreferences) {
      setIncludedGenres(currentPreferences.included_genres);
      setExcludedGenres(currentPreferences.excluded_genres);
      setMinYear(currentPreferences.min_release_year?.toString() ?? "");
      setMaxYear(currentPreferences.max_release_year?.toString() ?? "");
      setMinRating(currentPreferences.min_rating);
      setIncludeAdult(currentPreferences.include_adult);
      setSelectedProviders(
        currentPreferences.movie_providers.map((p) => p.name),
      );
    }
  }, [currentPreferences]);

  const toggleIncludedGenre = (genre: string) => {
    setIncludedGenres((prev) =>
      prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre],
    );
    setExcludedGenres((prev) => prev.filter((g) => g !== genre));
  };

  const toggleExcludedGenre = (genre: string) => {
    setExcludedGenres((prev) =>
      prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre],
    );
    setIncludedGenres((prev) => prev.filter((g) => g !== genre));
  };

  const toggleProvider = (provider: string) => {
    setSelectedProviders((prev) =>
      prev.includes(provider)
        ? prev.filter((p) => p !== provider)
        : [...prev, provider],
    );
  };

  const clampYear = (value: string): string => {
    if (!value) return "";
    const num = parseInt(value, 10);
    if (isNaN(num)) return "";
    return String(Math.max(MIN_YEAR, Math.min(MAX_YEAR, num)));
  };

  const handleApply = () => {
    const clampedMin = clampYear(minYear);
    const clampedMax = clampYear(maxYear);
    setMinYear(clampedMin);
    setMaxYear(clampedMax);

    const providers: MovieProvider[] = selectedProviders.map((name) => ({
      name,
      provider_type: "flatrate" as const,
    }));

    onApply({
      included_genres: includedGenres,
      excluded_genres: excludedGenres,
      min_release_year: clampedMin ? parseInt(clampedMin, 10) : null,
      max_release_year: clampedMax ? parseInt(clampedMax, 10) : null,
      min_rating: minRating,
      include_adult: includeAdult,
      movie_providers: providers,
    });
  };

  const handleClear = () => {
    setIncludedGenres([]);
    setExcludedGenres([]);
    setMinYear("");
    setMaxYear("");
    setMinRating(null);
    setIncludeAdult(false);
    setSelectedProviders([]);
    onApply({
      included_genres: [],
      excluded_genres: [],
      min_release_year: null,
      max_release_year: null,
      min_rating: null,
      include_adult: false,
      movie_providers: [],
    });
  };

  const hasFilters =
    includedGenres.length > 0 ||
    excludedGenres.length > 0 ||
    minYear !== "" ||
    maxYear !== "" ||
    minRating !== null ||
    includeAdult ||
    selectedProviders.length > 0;

  return (
    <div className="absolute right-4 top-4 z-50 max-h-[80vh] w-80 overflow-y-auto rounded-xl border border-border bg-card p-4 shadow-lg">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Filters</h3>
        <button
          onClick={onClose}
          className="rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <X size={16} />
        </button>
      </div>

      {/* Include genres */}
      <div className="mb-4">
        <label className="mb-2 block text-xs font-medium text-muted-foreground">
          Include Genres
        </label>
        <div className="flex flex-wrap gap-1.5">
          {TMDB_GENRES.map((genre) => (
            <button
              key={genre}
              onClick={() => toggleIncludedGenre(genre)}
              className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                includedGenres.includes(genre)
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground hover:bg-accent"
              }`}
            >
              {genre}
            </button>
          ))}
        </div>
      </div>

      {/* Exclude genres */}
      <div className="mb-4">
        <label className="mb-2 block text-xs font-medium text-muted-foreground">
          Exclude Genres
        </label>
        <div className="flex flex-wrap gap-1.5">
          {TMDB_GENRES.map((genre) => (
            <button
              key={genre}
              onClick={() => toggleExcludedGenre(genre)}
              className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                excludedGenres.includes(genre)
                  ? "bg-destructive text-destructive-foreground"
                  : "bg-secondary text-secondary-foreground hover:bg-accent"
              }`}
            >
              {genre}
            </button>
          ))}
        </div>
      </div>

      {/* Release year range */}
      <div className="mb-4">
        <label className="mb-2 block text-xs font-medium text-muted-foreground">
          Release Year
        </label>
        <div className="flex items-center gap-2">
          <input
            type="number"
            placeholder={String(MIN_YEAR)}
            min={MIN_YEAR}
            max={MAX_YEAR}
            value={minYear}
            onChange={(e) => setMinYear(e.target.value)}
            onBlur={() => setMinYear(clampYear(minYear))}
            className="w-full rounded-lg border border-input bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
          />
          <span className="text-xs text-muted-foreground">to</span>
          <input
            type="number"
            placeholder={String(MAX_YEAR)}
            min={MIN_YEAR}
            max={MAX_YEAR}
            value={maxYear}
            onChange={(e) => setMaxYear(e.target.value)}
            onBlur={() => setMaxYear(clampYear(maxYear))}
            className="w-full rounded-lg border border-input bg-background px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
          />
        </div>
      </div>

      {/* Min rating slider */}
      <div className="mb-4">
        <label className="mb-2 block text-xs font-medium text-muted-foreground">
          Minimum Rating
          <span className="ml-1 text-foreground">
            {minRating !== null ? minRating.toFixed(1) : "Any"}
          </span>
        </label>
        <input
          type="range"
          min={MIN_RATING}
          max={MAX_RATING}
          step={RATING_STEP}
          value={minRating ?? MIN_RATING}
          onChange={(e) => {
            const val = parseFloat(e.target.value);
            setMinRating(val === MIN_RATING ? null : val);
          }}
          className="w-full accent-primary"
        />
        <div className="flex justify-between text-[10px] text-muted-foreground">
          <span>Any</span>
          <span>10</span>
        </div>
      </div>

      {/* Adult content toggle */}
      <div className="mb-4 flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">
          Include Adult Content
        </label>
        <button
          onClick={() => setIncludeAdult((prev) => !prev)}
          className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${
            includeAdult ? "bg-primary" : "bg-secondary"
          }`}
        >
          <span
            className={`pointer-events-none inline-block h-4 w-4 rounded-full bg-background shadow-sm transition-transform ${
              includeAdult ? "translate-x-4" : "translate-x-0"
            }`}
          />
        </button>
      </div>

      {/* Streaming providers */}
      <div className="mb-4">
        <label className="mb-2 block text-xs font-medium text-muted-foreground">
          Streaming Providers
        </label>
        <div className="flex flex-wrap gap-1.5">
          {STREAMING_PROVIDERS.map((provider) => (
            <button
              key={provider}
              onClick={() => toggleProvider(provider)}
              className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                selectedProviders.includes(provider)
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground hover:bg-accent"
              }`}
            >
              {provider}
            </button>
          ))}
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-2">
        {hasFilters && (
          <button
            onClick={handleClear}
            className="flex-1 rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground hover:bg-accent"
          >
            Clear
          </button>
        )}
        <button
          onClick={handleApply}
          className="flex-1 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:opacity-90"
        >
          Apply
        </button>
      </div>
    </div>
  );
}
