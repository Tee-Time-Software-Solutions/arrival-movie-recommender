import { useState } from "react";
import { AnimatePresence } from "framer-motion";
import { Heart, ThumbsDown, Star } from "lucide-react";
import type { MovieDetails } from "@/types/movie";
import { MovieDetail } from "@/components/features/MovieDetail/MovieDetail";
import { useMovieStore } from "@/stores/movieStore";

export function ProfilePage() {
  const { likedMovies, dislikedMovies } = useMovieStore();
  const [selectedMovie, setSelectedMovie] = useState<MovieDetails | null>(null);

  const stats = {
    liked: likedMovies.length,
    passed: dislikedMovies.length,
  };

  return (
    <div className="min-h-screen px-4 pb-20 pt-10 md:pb-4">
      {/* Profile header: big pfp + name on left, stats on right */}
      <div className="mb-8 flex items-center justify-start gap-10">
        <div className="flex items-center gap-4">
          <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary text-3xl font-bold text-primary-foreground">
            D
          </div>
          <div>
            <h1 className="text-2xl font-bold">Demo User</h1>
            <p className="text-sm text-muted-foreground">demo@example.com</p>
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
          </div>
        </div>
      </div>

      {/* Liked movies */}
      {likedMovies.length > 0 && (
        <div className="mb-8">
          <h2 className="mb-4 text-lg font-semibold">Liked Movies</h2>
          <div className="flex gap-4 overflow-x-auto pb-2">
            {likedMovies.map((movie) => (
              <button
                key={movie.movie_id}
                onClick={() => setSelectedMovie(movie)}
                className="group flex shrink-0 flex-col items-center"
              >
                <div className="relative overflow-hidden rounded-lg">
                  <img
                    src={movie.poster_url}
                    alt={movie.title}
                    className="h-36 w-24 object-cover transition group-hover:scale-105"
                  />
                </div>
                <p className="mt-1 w-24 truncate text-center text-[11px] text-muted-foreground">
                  {movie.title}
                </p>
              </button>
            ))}
          </div>
        </div>
      )}

      {likedMovies.length === 0 && dislikedMovies.length === 0 && (
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
