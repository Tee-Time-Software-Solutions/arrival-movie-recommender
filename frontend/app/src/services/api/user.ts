import type { UserPreferences } from "@/types/user";

// TODO: Connect to real endpoint — currently mocked
export async function updatePreferences(
  prefs: UserPreferences,
): Promise<UserPreferences> {
  return prefs;
}
