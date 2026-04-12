import apiClient from "./client";
import type {
  OnboardingMovieCard,
  OnboardingSearchResult,
  OnboardingCompleteResponse,
} from "@/types/onboarding";

export async function getOnboardingMovies(): Promise<OnboardingMovieCard[]> {
  const { data } = await apiClient.get<OnboardingMovieCard[]>(
    "onboarding/movies",
  );
  return data;
}

export async function searchOnboardingMovies(
  query: string,
): Promise<OnboardingSearchResult[]> {
  const { data } = await apiClient.get<OnboardingSearchResult[]>(
    "onboarding/search",
    { params: { query } },
  );
  return data;
}

export async function completeOnboarding(
  gridMovieIds: number[],
  searchMovieTmdbIds: number[],
): Promise<OnboardingCompleteResponse> {
  const { data } = await apiClient.post<OnboardingCompleteResponse>(
    "onboarding/complete",
    {
      grid_movie_ids: gridMovieIds,
      search_movie_tmdb_ids: searchMovieTmdbIds,
    },
  );
  return data;
}
