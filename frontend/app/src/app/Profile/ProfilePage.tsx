import { useState, useEffect } from "react";
import { AnimatePresence } from "framer-motion";

import { Heart, ThumbsDown, Star, MousePointerClick } from "lucide-react";
import type { MovieDetails } from "@/types/movie";
import type { UserProfileSummary } from "@/types/user";
import { MovieDetail } from "@/components/features/MovieDetail/MovieDetail";
import { useAuthStore } from "@/stores/authStore";
import { useMovieStore } from "@/stores/movieStore";
import { getProfileSummary, getLikedMovies } from "@/services/api/user";

export function ProfilePage() {
  const { user, firebaseUid } = useAuthStore();
  const { likedMovies, setLikedMovies } = useMovieStore();
  const [profile, setProfile] = useState<UserProfileSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMovie, setSelectedMovie] = useState<MovieDetails | null>(null);

  useEffect(() => {
    if (!firebaseUid) {
      setLoading(false);
      return;
    }
    setLoading(true);
    Promise.all([getProfileSummary(firebaseUid), getLikedMovies(firebaseUid)])
      .then(([profileData, likedData]) => {
        setProfile(profileData);
        setLikedMovies(likedData.items);
      })
      .catch(() => setError("Failed to load profile"))
      .finally(() => setLoading(false));
  }, [firebaseUid]);

  if (loading) {
    return (
      <div className="flex h-[80vh] items-center justify-center">
        <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="flex h-[80vh] items-center justify-center px-6">
        <p className="text-sm text-muted-foreground">
          {error || "Sign in to view your profile."}
        </p>
      </div>
    );
  }

  const { profile: displayInfo, stats } = profile;
  const avatarUrl = displayInfo.avatar_url || user?.photoURL;

  return (
    <div className="min-h-screen px-4 pb-20 pt-10 md:pb-4">
      {/* Profile header */}
      <div className="mb-8 flex items-center justify-start gap-10">
        <div className="flex items-center gap-4">
          {avatarUrl ? (
            <img
              src={avatarUrl}
              alt="Avatar"
              className="h-20 w-20 rounded-full object-cover"
            />
          ) : (
            <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary text-3xl font-bold text-primary-foreground">
              {(displayInfo.username?.[0] || "U").toUpperCase()}
            </div>
          )}
          <div>
            <h1 className="text-2xl font-bold">
              {displayInfo.username || "Movie Lover"}
            </h1>
            <p className="text-sm text-muted-foreground">{user?.email}</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-5">
            <div className="text-center">
              <div className="flex items-center justify-center gap-1">
                <Heart className="h-4 w-4 text-green-500" />
                <span className="text-xl font-bold">{stats.total_likes}</span>
              </div>
              <p className="text-[10px] text-muted-foreground">Liked</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center gap-1">
                <ThumbsDown className="h-4 w-4 text-red-500" />
                <span className="text-xl font-bold">{stats.total_dislikes}</span>
              </div>
              <p className="text-[10px] text-muted-foreground">Passed</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center gap-1">
                <MousePointerClick className="h-4 w-4 text-blue-500" />
                <span className="text-xl font-bold">{stats.total_swipes}</span>
              </div>
              <p className="text-[10px] text-muted-foreground">Swipes</p>
            </div>
          </div>
        </div>
      </div>

      {/* Top genres */}
      {stats.top_genres.length > 0 && (
        <div className="mb-8">
          <h2 className="mb-3 text-lg font-semibold">Your Top Genres</h2>
          <div className="flex flex-wrap gap-2">
            {stats.top_genres.map((genre) => (
              <span
                key={genre}
                className="rounded-full bg-primary/10 px-3 py-1 text-sm font-medium text-primary"
              >
                {genre}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Recently liked movies from local store */}
      {likedMovies.length > 0 && (
        <div className="mb-8">
          <h2 className="mb-4 text-lg font-semibold">Recently Liked</h2>
          <div className="flex gap-4 overflow-x-auto pb-2">
            {likedMovies.slice(-10).reverse().map((movie) => (
              <button
                key={movie.movie_db_id}
                onClick={() => setSelectedMovie(movie)}
                className="group flex shrink-0 flex-col items-center"
              >
                <div className="relative overflow-hidden rounded-lg">
                  <img
                    src={movie.poster_url}
                    alt={movie.title}
                    className="h-36 w-24 object-cover transition group-hover:scale-105"
                  />
                  <div className="absolute bottom-0 inset-x-0 bg-black/70 px-1 py-0.5 flex items-center justify-center gap-0.5">
                    <Star className="h-3 w-3 fill-yellow-500 text-yellow-500" />
                    <span className="text-[10px] text-white">{movie.rating.toFixed(1)}</span>
                  </div>
                </div>
                <p className="mt-1 w-24 truncate text-center text-[11px] text-muted-foreground">
                  {movie.title}
                </p>
              </button>
            ))}
          </div>
        </div>
      )}

      {likedMovies.length === 0 && (
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
