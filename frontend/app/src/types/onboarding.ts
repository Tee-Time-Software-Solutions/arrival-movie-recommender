export interface OnboardingMovieCard {
  movie_db_id: number;
  tmdb_id: number;
  title: string;
  poster_url: string;
  release_year: number;
  tmdb_rating: number;
  genres: string[];
}

export interface OnboardingSearchResult {
  tmdb_id: number;
  title: string;
  poster_url: string | null;
  release_year: number | null;
}

export interface OnboardingCompleteResponse {
  onboarding_completed: boolean;
  movies_with_embeddings: number;
}
