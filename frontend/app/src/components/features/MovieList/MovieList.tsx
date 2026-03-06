import { useState } from "react";
import { AnimatePresence } from "framer-motion";
import { Star } from "lucide-react";
import type { MovieDetails, RatedMovie } from "@/types/movie";
import { MovieDetail } from "@/components/features/MovieDetail/MovieDetail";

interface MovieListProps {
  movies: MovieDetails[];
  watchedMap?: Map<string, RatedMovie>;
  emptyMessage?: string;
}

export function MovieList({
  movies,
  watchedMap,
  emptyMessage = "No movies yet",
}: MovieListProps) {
  const [selectedMovie, setSelectedMovie] = useState<MovieDetails | null>(null);

  if (movies.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        {emptyMessage}
      </p>
    );
  }

  return (
    <>
      <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
        {movies.map((movie) => {
          const watched = watchedMap?.get(movie.movie_id);
          return (
            <button
              key={movie.movie_id}
              onClick={() => setSelectedMovie(movie)}
              className="group relative overflow-hidden rounded-lg"
            >
              <img
                src={movie.poster_url}
                alt={movie.title}
                className="aspect-[2/3] w-full object-cover transition group-hover:scale-105"
              />
              {watched && (
                <div className="absolute bottom-0 inset-x-0 bg-black/70 px-1 py-0.5 flex items-center justify-center gap-0.5">
                  <Star className="h-3 w-3 fill-yellow-500 text-yellow-500" />
                  <span className="text-[10px] text-white">{watched.user_rating}</span>
                </div>
              )}
            </button>
          );
        })}
      </div>

      <AnimatePresence>
        {selectedMovie && (
          <MovieDetail
            movie={selectedMovie}
            onClose={() => setSelectedMovie(null)}
          />
        )}
      </AnimatePresence>
    </>
  );
}
