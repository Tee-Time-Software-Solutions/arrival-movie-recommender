import apiClient from "./client";
import type {
  PaginatedMovieDetails,
  WatchlistAddResponse,
  WatchlistRemoveResponse,
} from "@/types/movie";

export async function getWatchlist(
  limit: number = 20,
  offset: number = 0,
): Promise<PaginatedMovieDetails> {
  const { data } = await apiClient.get<PaginatedMovieDetails>("watchlist", {
    params: { limit, offset },
  });
  return data;
}

export async function addToWatchlist(
  movieId: number,
): Promise<WatchlistAddResponse> {
  const { data } = await apiClient.post<WatchlistAddResponse>(
    `watchlist/${movieId}`,
  );
  return data;
}

export async function removeFromWatchlist(
  movieId: number,
): Promise<WatchlistRemoveResponse> {
  const { data } = await apiClient.delete<WatchlistRemoveResponse>(
    `watchlist/${movieId}`,
  );
  return data;
}
