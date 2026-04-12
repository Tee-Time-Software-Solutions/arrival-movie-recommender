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
  movieDbIds: number[],
): Promise<OnboardingCompleteResponse> {
  const { data } = await apiClient.post<OnboardingCompleteResponse>(
    "onboarding/complete",
    { movie_db_ids: movieDbIds },
  );
  return data;
}
