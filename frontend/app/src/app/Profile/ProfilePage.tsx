import { useState } from "react";
import { AnimatePresence } from "framer-motion";
import { Heart, ThumbsDown, Eye, LogOut, Star, Sparkles } from "lucide-react";
import {
  MOCK_USER,
  MOCK_STATS,
  MOCK_WATCHED_MOVIES,
  MOCK_SUGGESTED_MOVIES,
} from "./mock-data";
import type { MovieDetails } from "@/types/movie";
import { MovieDetail } from "@/components/features/MovieDetail/MovieDetail";

export function ProfilePage() {
  const user = MOCK_USER;
  const stats = MOCK_STATS;
  const watchedMovies = MOCK_WATCHED_MOVIES;
  const suggestedMovies = MOCK_SUGGESTED_MOVIES;
  const [selectedMovie, setSelectedMovie] = useState<MovieDetails | null>(null);

  // Sort by rating desc, take max 10
  const sorted = [...watchedMovies]
    .sort((a, b) => b.rating - a.rating)
    .slice(0, 10);

  const podium = sorted.slice(0, 3); // top 3
  const rest = sorted.slice(3); // remaining rows

  // Podium order: 2nd, 1st, 3rd
  const podiumOrder = [podium[1], podium[0], podium[2]].filter(Boolean);

  return (
    <div className="min-h-screen px-4 pb-20 pt-10 md:pb-4">
      {/* Profile header: big pfp + name on left, stats on right */}
      <div className="mb-8 flex items-center justify-start gap-10">
        <div className="flex items-center gap-4">
          {user.photoURL ? (
            <img
              src={user.photoURL}
              alt="Avatar"
              className="h-20 w-20 rounded-full object-cover"
            />
          ) : (
            <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary text-3xl font-bold text-primary-foreground">
              {(user.displayName?.[0] || user.email?.[0] || "U").toUpperCase()}
            </div>
          )}
          <div>
            <h1 className="text-2xl font-bold">
              {user.displayName || "Movie Lover"}
            </h1>
            <p className="text-sm text-muted-foreground">{user.email}</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-5">
            <div className="text-center">
              <div className="flex items-center justify-center gap-1">
                <Heart className="h-4 w-4 text-green-500" />
                <span className="text-xl font-bold">{stats.liked}</span>
              </div>
              <p className="text-[10px] text-muted-foreground">Liked</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center gap-1">
                <ThumbsDown className="h-4 w-4 text-red-500" />
                <span className="text-xl font-bold">{stats.passed}</span>
              </div>
              <p className="text-[10px] text-muted-foreground">Passed</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center gap-1">
                <Eye className="h-4 w-4 text-blue-500" />
                <span className="text-xl font-bold">{stats.rated}</span>
              </div>
              <p className="text-[10px] text-muted-foreground">Rated</p>
            </div>
          </div>
        </div>
      </div>

      {/* Suggested movies */}
      {suggestedMovies.length > 0 && (
        <div className="mb-8">
          <div className="mb-3 flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-yellow-500" />
            <h2 className="text-lg font-semibold">Suggested for You</h2>
          </div>
          <div className="flex gap-4 overflow-x-auto pb-2">
            {suggestedMovies.map((s) => (
              <button
                key={s.movie.movie_id}
                onClick={() => setSelectedMovie(s.movie)}
                className="group flex shrink-0 gap-3 rounded-xl bg-card p-2.5 text-left transition hover:bg-secondary"
              >
                <div className="relative overflow-hidden rounded-lg">
                  <img
                    src={s.movie.poster_url}
                    alt={s.movie.title}
                    className="h-36 w-24 object-cover transition group-hover:scale-105"
                  />
                </div>
                <div className="flex w-36 flex-col justify-center">
                  <p className="text-sm font-medium leading-tight">{s.movie.title}</p>
                  <p className="mt-1 text-[11px] leading-snug text-muted-foreground line-clamp-3">
                    {s.reason}
                  </p>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Rated movies – podium */}
      {podiumOrder.length > 0 && (
        <div className="mb-6">
          <h2 className="mb-4 text-lg font-semibold">Your Top Rated</h2>
          <div className="flex items-end justify-center gap-3">
            {podiumOrder.map((wm, i) => {
              // i=0 → 2nd place, i=1 → 1st place, i=2 → 3rd place
              const place = i === 1 ? 1 : i === 0 ? 2 : 3;
              const isFirst = place === 1;
              return (
                <button
                  key={wm.movie.movie_id}
                  onClick={() => setSelectedMovie(wm.movie)}
                  className="group flex flex-col items-center"
                >
                  <div
                    className={`relative overflow-hidden rounded-lg ${
                      isFirst ? "ring-2 ring-yellow-500" : ""
                    }`}
                  >
                    <img
                      src={wm.movie.poster_url}
                      alt={wm.movie.title}
                      className={`object-cover transition group-hover:scale-105 ${
                        isFirst
                          ? "h-40 w-[106px]"
                          : "h-32 w-[85px]"
                      }`}
                    />
                    <div className="absolute bottom-0 inset-x-0 bg-black/70 px-1 py-0.5 flex items-center justify-center gap-0.5">
                      <Star className="h-3 w-3 fill-yellow-500 text-yellow-500" />
                      <span className="text-[10px] text-white">{wm.rating}</span>
                    </div>
                  </div>
                  <div
                    className={`mt-1 flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold ${
                      place === 1
                        ? "bg-yellow-500 text-black"
                        : place === 2
                          ? "bg-gray-300 text-black"
                          : "bg-amber-700 text-white"
                    }`}
                  >
                    {place}
                  </div>
                  <p className="mt-0.5 w-20 truncate text-center text-[11px] text-muted-foreground">
                    {wm.movie.title}
                  </p>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Rest of rated movies as rows */}
      {rest.length > 0 && (
        <div className="mb-8">
          <div className="space-y-2">
            {rest.map((wm, i) => (
              <button
                key={wm.movie.movie_id}
                onClick={() => setSelectedMovie(wm.movie)}
                className="flex w-full items-center gap-3 rounded-xl bg-card p-2 text-left transition hover:bg-secondary"
              >
                <span className="w-5 text-center text-xs font-semibold text-muted-foreground">
                  {i + 4}
                </span>
                <img
                  src={wm.movie.poster_url}
                  alt={wm.movie.title}
                  className="h-16 w-11 flex-shrink-0 rounded-md object-cover"
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{wm.movie.title}</p>
                  <p className="text-xs text-muted-foreground">
                    {wm.movie.release_year} &middot; {wm.movie.genres.slice(0, 2).join(", ")}
                  </p>
                </div>
                <div className="flex items-center gap-0.5">
                  <Star className="h-3.5 w-3.5 fill-yellow-500 text-yellow-500" />
                  <span className="text-sm font-semibold">{wm.rating}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {watchedMovies.length === 0 && (
        <div className="py-12 text-center">
          <p className="text-muted-foreground">
            Start swiping to build your movie collection!
          </p>
        </div>
      )}

      <AnimatePresence>
        {selectedMovie && (
          <MovieDetail
            movie={selectedMovie}
            onClose={() => setSelectedMovie(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
