import apiClient from "./client";
import type {
  UserPreferences,
  UserProfileSummary,
  UserCreate,
  UserCreatedResponse,
} from "@/types/user";
import type { PaginatedMovieDetails } from "@/types/movie";

export async function registerUser(
  userData: UserCreate,
): Promise<UserCreatedResponse> {
  const { data } = await apiClient.post<UserCreatedResponse>(
    "users/register",
    userData,
  );
  return data;
}

export async function getProfileSummary(
  userId: string,
): Promise<UserProfileSummary> {
  const { data } = await apiClient.get<UserProfileSummary>(
    `users/${userId}/summary`,
  );
  return data;
}

export async function getLikedMovies(
  userId: string,
  limit: number = 20,
  offset: number = 0,
): Promise<PaginatedMovieDetails> {
  const { data } = await apiClient.get<PaginatedMovieDetails>(
    `users/${userId}/liked-movies`,
    { params: { limit, offset } },
  );
  return data;
}

export async function updatePreferences(
  userId: string,
  prefs: UserPreferences,
): Promise<UserPreferences> {
  const { data } = await apiClient.patch<UserPreferences>(
    `users/${userId}/preferences`,
    prefs,
  );
  return data;
}
