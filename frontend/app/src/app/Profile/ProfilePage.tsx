import { MovieList } from "@/components/features/MovieList/MovieList";
import { Heart, ThumbsDown, Eye, LogOut } from "lucide-react";
import {
  MOCK_USER,
  MOCK_STATS,
  MOCK_LIKED_MOVIES,
  MOCK_WATCHED_MOVIES,
} from "./mock-data";
import type { WatchedMovie } from "@/types/movie";

export function ProfilePage() {
  const user = MOCK_USER;
  const stats = MOCK_STATS;
  const likedMovies = MOCK_LIKED_MOVIES;
  const watchedMovies = MOCK_WATCHED_MOVIES;

  const watchedMap = new Map<string, WatchedMovie>();
  for (const wm of watchedMovies) {
    watchedMap.set(wm.movie.movie_id, wm);
  }

  return (
    <div className="min-h-screen px-4 pb-20 pt-6 md:pb-4">
      {/* User info */}
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {user.photoURL ? (
            <img
              src={user.photoURL}
              alt="Avatar"
              className="h-12 w-12 rounded-full object-cover"
            />
          ) : (
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-lg font-bold text-primary-foreground">
              {(user.displayName?.[0] || user.email?.[0] || "U").toUpperCase()}
            </div>
          )}
          <div>
            <p className="font-semibold">
              {user.displayName || "Movie Lover"}
            </p>
            <p className="text-xs text-muted-foreground">{user.email}</p>
          </div>
        </div>
        <button
          onClick={() => console.log("Sign out (mock)")}
          className="rounded-lg p-2 text-muted-foreground transition hover:bg-secondary hover:text-foreground"
        >
          <LogOut className="h-5 w-5" />
        </button>
      </div>

      {/* Stats */}
      <div className="mb-8 grid grid-cols-3 gap-3">
        <div className="rounded-xl bg-card p-4 text-center">
          <Heart className="mx-auto mb-1 h-5 w-5 text-green-500" />
          <p className="text-2xl font-bold">{stats.liked}</p>
          <p className="text-xs text-muted-foreground">Liked</p>
        </div>
        <div className="rounded-xl bg-card p-4 text-center">
          <ThumbsDown className="mx-auto mb-1 h-5 w-5 text-red-500" />
          <p className="text-2xl font-bold">{stats.passed}</p>
          <p className="text-xs text-muted-foreground">Passed</p>
        </div>
        <div className="rounded-xl bg-card p-4 text-center">
          <Eye className="mx-auto mb-1 h-5 w-5 text-blue-500" />
          <p className="text-2xl font-bold">{stats.rated}</p>
          <p className="text-xs text-muted-foreground">Rated</p>
        </div>
      </div>

      {/* Watched movies */}
      {watchedMovies.length > 0 && (
        <div className="mb-8">
          <h2 className="mb-3 text-lg font-semibold">Rated Movies</h2>
          <MovieList
            movies={watchedMovies.map((w) => w.movie)}
            watchedMap={watchedMap}
            emptyMessage="No rated movies yet"
          />
        </div>
      )}

      {/* Liked movies */}
      {likedMovies.length > 0 && (
        <div className="mb-8">
          <h2 className="mb-3 text-lg font-semibold">Liked Movies</h2>
          <MovieList
            movies={likedMovies}
            emptyMessage="No liked movies yet"
          />
        </div>
      )}

      {likedMovies.length === 0 && watchedMovies.length === 0 && (
        <div className="py-12 text-center">
          <p className="text-muted-foreground">
            Start swiping to build your movie collection!
          </p>
        </div>
      )}
    </div>
  );
}
