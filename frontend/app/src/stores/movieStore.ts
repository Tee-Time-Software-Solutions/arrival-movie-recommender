import { create } from "zustand";
import type { MovieDetails } from "@/types/movie";

interface MovieState {
  queue: MovieDetails[];
  currentIndex: number;
  loading: boolean;
  error: string | null;
  exhausted: boolean;

  addToQueue: (movies: MovieDetails[]) => void;
  nextMovie: () => void;
  resetQueue: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setExhausted: (exhausted: boolean) => void;
  getCurrentMovie: () => MovieDetails | undefined;
}

export const useMovieStore = create<MovieState>((set, get) => ({
  queue: [],
  currentIndex: 0,
  loading: false,
  error: null,
  exhausted: false,

  addToQueue: (movies) =>
    set((state) => {
      const existingIds = new Set(state.queue.map((m) => m.movie_db_id));
      const newMovies = movies.filter((m) => !existingIds.has(m.movie_db_id));
      return { queue: [...state.queue, ...newMovies] };
    }),

  nextMovie: () =>
    set((state) => ({ currentIndex: state.currentIndex + 1 })),

  resetQueue: () => set({ queue: [], currentIndex: 0, exhausted: false }),

  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setExhausted: (exhausted) => set({ exhausted }),
  getCurrentMovie: () => {
    const { queue, currentIndex } = get();
    return queue[currentIndex];
  },
}));
