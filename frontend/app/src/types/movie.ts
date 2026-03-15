export type ProviderType = "flatrate" | "rent" | "buy";

export interface CastMember {
  name: string;
  role_type: string;
  character_name: string | null;
  profile_path: string | null;
}

export interface MovieProvider {
  name: string;
  provider_type: ProviderType;
}

export interface MovieCard {
  movie_db_id: number;
  tmdb_id: number;
  title: string;
  poster_url: string;
  release_year: number;
  rating: number;
  genres: string[];
  is_adult: boolean;
}

export interface MovieDetails extends MovieCard {
  synopsis: string;
  cast: CastMember[];
  trailer_url: string | null;
  runtime: number;
  movie_providers: MovieProvider[];
}

export interface PaginatedMovieDetails {
  items: MovieDetails[];
  total: number;
  limit: number;
  offset: number;
}

export type SwipeAction = "like" | "dislike" | "skip";

export interface SwipeRequest {
  action_type: SwipeAction;
  is_supercharged: boolean;
}

export interface RegisteredFeedback {
  interaction_id: number;
  movie_id: number;
  action_type: SwipeAction;
  is_supercharged: boolean;
  registered: boolean;
}

export interface WatchedMovie {
  movie: MovieDetails;
  rating: number;
  watchedAt: string;
}
