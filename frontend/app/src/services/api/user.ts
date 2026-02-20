import type { UserPreferences } from "@/types/user";
import { MOCK_PROFILE_SUMMARY, MOCK_WATCHED_MOVIES } from "@/mock-data";

// TODO: Connect to real endpoint — currently mocked
export async function getProfileSummary() {
  return MOCK_PROFILE_SUMMARY;
}

// TODO: Connect to real endpoint — currently mocked
export async function getWatchedMovies() {
  return MOCK_WATCHED_MOVIES;
}

// TODO: Connect to real endpoint — currently mocked
export async function updatePreferences(
  prefs: UserPreferences,
): Promise<UserPreferences> {
  return prefs;
}
