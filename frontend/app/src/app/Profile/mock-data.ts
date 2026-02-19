import type { MovieDetails, WatchedMovie } from "@/types/movie";

export const MOCK_USER = {
  displayName: "Bruce Wayne",
  email: "bwayne@waynecorp.com",
  photoURL: null as string | null,
};

export const MOCK_STATS = {
  liked: 24,
  passed: 18,
  rated: 12,
};

const ALL_MOVIES: MovieDetails[] = [
  {
    movie_id: "1",
    tmdb_id: "155",
    title: "The Dark Knight",
    poster_url: "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911BTUgMe1nNaD3.jpg",
    release_year: 2008,
    rating: 9.0,
    genres: ["Action", "Crime", "Drama"],
    is_adult: false,
    synopsis: "When the menace known as the Joker wreaks havoc on Gotham City, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
    cast: [],
    trailer_url: "",
    runtime: 152,
    movie_providers: [],
  },
  {
    movie_id: "2",
    tmdb_id: "550",
    title: "Fight Club",
    poster_url: "https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
    release_year: 1999,
    rating: 8.8,
    genres: ["Drama"],
    is_adult: false,
    synopsis: "An insomniac office worker and a devil-may-care soap maker form an underground fight club.",
    cast: [],
    trailer_url: "",
    runtime: 139,
    movie_providers: [],
  },
  {
    movie_id: "3",
    tmdb_id: "680",
    title: "Pulp Fiction",
    poster_url: "https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg",
    release_year: 1994,
    rating: 8.9,
    genres: ["Thriller", "Crime"],
    is_adult: false,
    synopsis: "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine.",
    cast: [],
    trailer_url: "",
    runtime: 154,
    movie_providers: [],
  },
  {
    movie_id: "4",
    tmdb_id: "27205",
    title: "Inception",
    poster_url: "https://image.tmdb.org/t/p/w500/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
    release_year: 2010,
    rating: 8.8,
    genres: ["Action", "Sci-Fi", "Adventure"],
    is_adult: false,
    synopsis: "A thief who steals corporate secrets through dream-sharing technology is given the task of planting an idea.",
    cast: [],
    trailer_url: "",
    runtime: 148,
    movie_providers: [],
  },
  {
    movie_id: "5",
    tmdb_id: "329865",
    title: "Arrival",
    poster_url: "https://image.tmdb.org/t/p/w500/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg",
    release_year: 2016,
    rating: 7.9,
    genres: ["Drama", "Sci-Fi"],
    is_adult: false,
    synopsis: "A linguist works with the military to communicate with alien lifeforms after twelve mysterious spacecraft appear around the world.",
    cast: [],
    trailer_url: "",
    runtime: 116,
    movie_providers: [],
  },
  {
    movie_id: "6",
    tmdb_id: "244786",
    title: "Whiplash",
    poster_url: "https://image.tmdb.org/t/p/w500/7fn624j5lj3xTme2SgiLCeuedos.jpg",
    release_year: 2014,
    rating: 8.5,
    genres: ["Drama", "Music"],
    is_adult: false,
    synopsis: "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential.",
    cast: [],
    trailer_url: "",
    runtime: 106,
    movie_providers: [],
  },
  {
    movie_id: "7",
    tmdb_id: "238",
    title: "The Godfather",
    poster_url: "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
    release_year: 1972,
    rating: 9.2,
    genres: ["Drama", "Crime"],
    is_adult: false,
    synopsis: "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant youngest son.",
    cast: [],
    trailer_url: "",
    runtime: 175,
    movie_providers: [],
  },
  {
    movie_id: "8",
    tmdb_id: "157336",
    title: "Interstellar",
    poster_url: "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
    release_year: 2014,
    rating: 8.7,
    genres: ["Adventure", "Drama", "Sci-Fi"],
    is_adult: false,
    synopsis: "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
    cast: [],
    trailer_url: "",
    runtime: 169,
    movie_providers: [],
  },
  {
    movie_id: "9",
    tmdb_id: "120",
    title: "The Lord of the Rings: The Fellowship of the Ring",
    poster_url: "https://image.tmdb.org/t/p/w500/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg",
    release_year: 2001,
    rating: 8.8,
    genres: ["Adventure", "Fantasy", "Action"],
    is_adult: false,
    synopsis: "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring.",
    cast: [],
    trailer_url: "",
    runtime: 178,
    movie_providers: [],
  },
  {
    movie_id: "10",
    tmdb_id: "278",
    title: "The Shawshank Redemption",
    poster_url: "https://image.tmdb.org/t/p/w500/9cjIGRQL0ElAgHMvQ3XL0XSBPHr.jpg",
    release_year: 1994,
    rating: 9.3,
    genres: ["Drama", "Crime"],
    is_adult: false,
    synopsis: "Over the course of several years, two convicts form a friendship, seeking consolation and eventual redemption through basic compassion.",
    cast: [],
    trailer_url: "",
    runtime: 142,
    movie_providers: [],
  },
];

// Rated movies sorted by rating desc (top 3 will be podium)
export const MOCK_WATCHED_MOVIES: WatchedMovie[] = [
  { movie: ALL_MOVIES[0], rating: 5, watchedAt: "2025-01-15T00:00:00Z" },  // The Dark Knight
  { movie: ALL_MOVIES[4], rating: 5, watchedAt: "2025-02-01T00:00:00Z" },  // Arrival
  { movie: ALL_MOVIES[1], rating: 4, watchedAt: "2025-01-20T00:00:00Z" },  // Fight Club
  { movie: ALL_MOVIES[6], rating: 5, watchedAt: "2025-02-10T00:00:00Z" },  // The Godfather
  { movie: ALL_MOVIES[2], rating: 4, watchedAt: "2025-01-25T00:00:00Z" },  // Pulp Fiction
  { movie: ALL_MOVIES[3], rating: 4, watchedAt: "2025-02-05T00:00:00Z" },  // Inception
  { movie: ALL_MOVIES[5], rating: 3, watchedAt: "2025-01-18T00:00:00Z" },  // Whiplash
  { movie: ALL_MOVIES[7], rating: 4, watchedAt: "2025-02-08T00:00:00Z" },  // Interstellar
  { movie: ALL_MOVIES[8], rating: 5, watchedAt: "2025-02-12T00:00:00Z" },  // LOTR
  { movie: ALL_MOVIES[9], rating: 5, watchedAt: "2025-02-15T00:00:00Z" },  // Shawshank
];

export interface SuggestedMovie {
  movie: MovieDetails;
  reason: string;
}

export const MOCK_SUGGESTED_MOVIES: SuggestedMovie[] = [
  {
    movie: {
      movie_id: "s1",
      tmdb_id: "603",
      title: "The Matrix",
      poster_url: "https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
      release_year: 1999,
      rating: 8.7,
      genres: ["Action", "Sci-Fi"],
      is_adult: false,
      synopsis: "A computer programmer discovers that reality as he knows it is a simulation created by machines.",
      cast: [],
      trailer_url: "",
      runtime: 136,
      movie_providers: [],
    },
    reason: "You loved Inception's mind-bending premise — this one goes even deeper.",
  },
  {
    movie: {
      movie_id: "s2",
      tmdb_id: "497",
      title: "The Green Mile",
      poster_url: "https://image.tmdb.org/t/p/w500/8VG8fDNiy50H4FedGwdSVUPoaJe.jpg",
      release_year: 1999,
      rating: 8.5,
      genres: ["Fantasy", "Drama", "Crime"],
      is_adult: false,
      synopsis: "A tale set on death row where gentle giant John possesses the mysterious power to heal people's ailments.",
      cast: [],
      trailer_url: "",
      runtime: 189,
      movie_providers: [],
    },
    reason: "Same emotional punch as Shawshank, from the same author.",
  },
  {
    movie: {
      movie_id: "s3",
      tmdb_id: "389",
      title: "12 Angry Men",
      poster_url: "https://image.tmdb.org/t/p/w500/ow3wq89wM8qd5X7hWKxiRfsFf9C.jpg",
      release_year: 1957,
      rating: 9.0,
      genres: ["Drama"],
      is_adult: false,
      synopsis: "The jury in a New York City murder trial is frustrated by a single member whose skeptical caution forces them to more carefully consider the evidence.",
      cast: [],
      trailer_url: "",
      runtime: 96,
      movie_providers: [],
    },
    reason: "Pure dialogue-driven tension — if you liked Whiplash, you'll love this.",
  },
];
