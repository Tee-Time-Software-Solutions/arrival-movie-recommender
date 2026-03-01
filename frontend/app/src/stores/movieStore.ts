import { create } from "zustand";
import type { MovieDetails } from "@/types/movie";

interface MovieState {
  queue: MovieDetails[];
  currentIndex: number;
  likedMovies: MovieDetails[];
  dislikedMovies: MovieDetails[];
  loading: boolean;
  error: string | null;

  addToQueue: (movies: MovieDetails[]) => void;
  nextMovie: () => void;
  likeMovie: (movie: MovieDetails) => void;
  dislikeMovie: (movie: MovieDetails) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  getCurrentMovie: () => MovieDetails | undefined;
}

export const useMovieStore = create<MovieState>((set, get) => ({
  queue: [],
  currentIndex: 0,
  likedMovies: [],
  dislikedMovies: [],
  loading: false,
  error: null,

  addToQueue: (movies) =>
    set((state) => {
      const existingIds = new Set(state.queue.map((m) => m.movie_id));
      const newMovies = movies.filter((m) => !existingIds.has(m.movie_id));
      return { queue: [...state.queue, ...newMovies] };
    }),

  nextMovie: () =>
    set((state) => ({ currentIndex: state.currentIndex + 1 })),

  likeMovie: (movie) =>
    set((state) => ({
      likedMovies: [...state.likedMovies, movie],
    })),

  dislikeMovie: (movie) =>
    set((state) => ({
      dislikedMovies: [...state.dislikedMovies, movie],
    })),

  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  getCurrentMovie: () => {
    const { queue, currentIndex } = get();
    return queue[currentIndex];
  },
}));
