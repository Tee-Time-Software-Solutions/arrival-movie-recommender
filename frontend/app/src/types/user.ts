import type { MovieProvider } from "./movie";

export interface UserAnalytics {
  total_swipes: number;
  total_likes: number;
  total_dislikes: number;
  total_seen: number;
  top_genres: string[];
}

export interface UserPreferences {
  included_genres: string[];
  excluded_genres: string[];
  min_release_year: number | null;
  max_release_year: number | null;
  min_rating: number | null;
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

export interface UserCreate {
  firebase_uid: string;
  profile_image_url: string;
  email: string;
}

export interface UserCreatedResponse {
  id: number;
  firebase_uid: string;
  profile_image_url: string;
  email: string;
  created_at: string;
  updated_at: string;
}
