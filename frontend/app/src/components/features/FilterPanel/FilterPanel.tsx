import { useState, useEffect } from "react";
import { X } from "lucide-react";
import type { UserPreferences } from "@/types/user";

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

const MIN_YEAR = 1900;
const MAX_YEAR = new Date().getFullYear();

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
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [minYear, setMinYear] = useState("");
  const [maxYear, setMaxYear] = useState("");

  useEffect(() => {
    if (currentPreferences) {
      setSelectedGenres(currentPreferences.included_genres);
      setMinYear(
        currentPreferences.min_release_year?.toString() ?? "",
      );
      setMaxYear(
        currentPreferences.max_release_year?.toString() ?? "",
      );
    }
  }, [currentPreferences]);

  const toggleGenre = (genre: string) => {
    setSelectedGenres((prev) =>
      prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre],
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

    onApply({
      included_genres: selectedGenres,
      excluded_genres: currentPreferences?.excluded_genres ?? [],
      min_release_year: clampedMin ? parseInt(clampedMin, 10) : null,
      max_release_year: clampedMax ? parseInt(clampedMax, 10) : null,
      min_rating: currentPreferences?.min_rating ?? null,
      include_adult: currentPreferences?.include_adult ?? false,
      movie_providers: currentPreferences?.movie_providers ?? [],
    });
  };

  const handleClear = () => {
    setSelectedGenres([]);
    setMinYear("");
    setMaxYear("");
    onApply({
      included_genres: [],
      excluded_genres: [],
      min_release_year: null,
      max_release_year: null,
      min_rating: null,
      include_adult: currentPreferences?.include_adult ?? false,
      movie_providers: currentPreferences?.movie_providers ?? [],
    });
  };

  const hasFilters =
    selectedGenres.length > 0 || minYear !== "" || maxYear !== "";

  return (
    <div className="absolute right-4 top-4 z-50 w-72 rounded-xl border border-border bg-card p-4 shadow-lg">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Filters</h3>
        <button
          onClick={onClose}
          className="rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <X size={16} />
        </button>
      </div>

      <div className="mb-4">
        <label className="mb-2 block text-xs font-medium text-muted-foreground">
          Genres
        </label>
        <div className="flex flex-wrap gap-1.5">
          {TMDB_GENRES.map((genre) => (
            <button
              key={genre}
              onClick={() => toggleGenre(genre)}
              className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                selectedGenres.includes(genre)
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground hover:bg-accent"
              }`}
            >
              {genre}
            </button>
          ))}
        </div>
      </div>

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
