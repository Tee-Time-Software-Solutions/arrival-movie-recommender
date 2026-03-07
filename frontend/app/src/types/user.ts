import type { MovieProvider } from "./movie";

export interface UserAnalytics {
  total_swipes: number;
  total_likes: number;
  total_dislikes: number;
  top_genres: string[];
}

export interface UserPreferences {
  preferred_genres: string[];
  min_release_year: number;
  include_adult: boolean;
  movie_providers: MovieProvider[];
}

export interface UserDisplayInfo {
  username: string;
  avatar_url: string;
  joined_at: string;
}

export interface UserProfileSummary {
  profile: UserDisplayInfo;
  stats: UserAnalytics;
  preferences: UserPreferences;
}
