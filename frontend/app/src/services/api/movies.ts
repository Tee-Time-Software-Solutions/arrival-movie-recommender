import apiClient from "./client";
import type {
  MovieDetails,
  RatedMovie,
  SwipeAction,
  RegisteredFeedback,
} from "@/types/movie";

export async function getMovieFeed(): Promise<MovieDetails> {
  const { data } = await apiClient.get<MovieDetails>("movies/feed");
  return data;
}

export async function getMovieQueue(
  count: number = 5,
): Promise<MovieDetails[]> {
  const { data } = await apiClient.get<MovieDetails[]>("movies/feed/batch", {
    params: { count },
  });
  return data;
}

export async function flushMovieFeed(): Promise<void> {
  await apiClient.delete("movies/feed");
}

export async function registerSwipe(
  movieId: number,
  actionType: SwipeAction,
  isSupercharged: boolean = false,
): Promise<RegisteredFeedback> {
  const { data } = await apiClient.post<RegisteredFeedback>(
    `interactions/${movieId}/swipe`,
    {
      action_type: actionType,
      is_supercharged: isSupercharged,
    },
  );
  return data;
}
