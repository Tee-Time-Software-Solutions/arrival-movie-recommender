export type ProviderType = "flatrate" | "rent" | "buy";

export interface CastMember {
  name: string;
  role_type: string | null;
  profile_path: string | null;
}

export interface MovieProvider {
  name: string;
  provider_type: ProviderType;
}

export interface MovieCard {
  movie_id: string;
  tmdb_id: string;
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
  trailer_url: string;
  runtime: number;
  movie_providers: MovieProvider[];
}

export type SwipeAction = "like" | "dislike" | "skip";

export interface SwipeRequest {
  action_type: SwipeAction;
  is_supercharged: boolean;
}

export interface RegisteredFeedback {
  interaction_id: string;
  movie_id: number;
  action_type: SwipeAction;
  is_supercharged: boolean;
  registered: boolean;
}

export interface RatedMovie {
  movie: MovieDetails;
  user_rating: number;
  rated_at: string;
}
