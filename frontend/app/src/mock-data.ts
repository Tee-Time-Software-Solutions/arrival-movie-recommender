import type { UserProfileSummary } from "@/types/user";
import type { MovieDetails } from "@/types/movie";

// ── Chat mock data ──────────────────────────────────────────────────────────

export const MOCK_CHAT_RESPONSES = [
  {
    content:
      "Based on your taste, I'd recommend checking out **Blade Runner 2049**. It's a visually stunning sci-fi film with deep themes about identity and humanity.",
    movieRecommendations: [
      {
        title: "Blade Runner 2049",
        year: 2017,
        posterUrl: "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg",
        reason: "Stunning sci-fi with philosophical depth",
      },
    ],
  },
  {
    content:
      "If you enjoy thought-provoking cinema, **Arrival** is a must-watch. It's a unique take on first contact that focuses on language and time.",
    movieRecommendations: [
      {
        title: "Arrival",
        year: 2016,
        posterUrl: "https://image.tmdb.org/t/p/w500/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg",
        reason: "Intelligent sci-fi about communication",
      },
    ],
  },
  {
    content:
      "You might love **Everything Everywhere All at Once**! It's a wild, emotional ride through the multiverse with incredible performances.",
    movieRecommendations: [
      {
        title: "Everything Everywhere All at Once",
        year: 2022,
        posterUrl: "https://image.tmdb.org/t/p/w500/w3LxiVYdWWRvEVdn5RYq6jIqkb1.jpg",
        reason: "Mind-bending multiverse adventure",
      },
    ],
  },
  {
    content:
      "For something different, try **The Grand Budapest Hotel**. Wes Anderson's visual storytelling is pure cinema magic.",
    movieRecommendations: [
      {
        title: "The Grand Budapest Hotel",
        year: 2014,
        posterUrl: "https://image.tmdb.org/t/p/w500/eWDyYQ3TcHOLd5fLaNFQ1q6L4lp.jpg",
        reason: "Whimsical visual masterpiece",
      },
    ],
  },
];

// ── User / Profile mock data ────────────────────────────────────────────────

export const MOCK_PROFILE_SUMMARY: UserProfileSummary = {
  profile: {
    username: "MovieLover",
    avatar_url: "https://via.placeholder.com/150",
    joined_at: new Date().toISOString(),
  },
  stats: {
    total_swipes: 0,
    total_likes: 0,
    total_dislikes: 0,
    top_genres: [],
  },
  preferences: {
    preferred_genres: [],
    min_release_year: 2000,
    include_adult: false,
    movie_providers: [],
  },
};

export const MOCK_WATCHED_MOVIES: MovieDetails[] = [];
