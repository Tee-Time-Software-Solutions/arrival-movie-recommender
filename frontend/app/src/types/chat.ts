export interface ChatMessage {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: string;
  movieRecommendations?: ChatMovieRecommendation[];
}

export interface ChatMovieRecommendation {
  title: string;
  year: number;
  posterUrl: string;
  reason: string;
}
