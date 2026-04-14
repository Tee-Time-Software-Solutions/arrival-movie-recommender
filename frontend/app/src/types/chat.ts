import type { MovieDetails } from "./movie";

export interface ChatMessage {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: string;
  movieRecommendations?: MovieDetails[];
}
