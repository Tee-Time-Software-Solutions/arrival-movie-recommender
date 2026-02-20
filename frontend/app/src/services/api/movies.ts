import apiClient from "./client";
import type {
  MovieDetails,
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
  const promises = Array.from({ length: count }, () => getMovieFeed());
  const results = await Promise.allSettled(promises);
  return results
    .filter(
      (r): r is PromiseFulfilledResult<MovieDetails> =>
        r.status === "fulfilled",
    )
    .map((r) => r.value);
}

export async function registerSwipe(
  movieId: string,
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

// TODO: Connect to real endpoint — currently mocked
export async function rateMovie(
  _movieId: string,
  _rating: number,
): Promise<void> {
  // When a real rating endpoint exists, call it here.
  // For now, the rating is stored locally in the movieStore.
}
