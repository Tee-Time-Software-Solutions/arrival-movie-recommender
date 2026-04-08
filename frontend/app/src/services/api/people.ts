import apiClient from "./client";
import type { LinkedMovie } from "@/types/user";

export async function getPersonLinkedMovies(
  personTmdbId: number,
): Promise<LinkedMovie[]> {
  const { data } = await apiClient.get<LinkedMovie[]>(
    `people/${personTmdbId}/linked-movies`,
  );
  return data;
}
