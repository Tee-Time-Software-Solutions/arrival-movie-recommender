import { useState } from "react";
import { motion } from "framer-motion";
import { X, Star } from "lucide-react";
import type { MovieDetails } from "@/types/movie";
import { getProviderLogoUrl } from "@/lib/constants";

interface MovieDetailContentProps {
  movie: MovieDetails;
}

function getYoutubeVideoId(url: string): string | null {
  const match = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&]+)/);
  return match ? match[1] : null;
}

function ProviderLogo({ name }: { name: string }) {
  const [failed, setFailed] = useState(false);
  const url = getProviderLogoUrl(name);

  if (failed) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-secondary text-xs font-bold text-secondary-foreground">
        {name.slice(0, 2).toUpperCase()}
      </div>
    );
  }

  return (
    <img
      src={url}
      alt={name}
      className="h-full w-full object-cover"
      onError={() => setFailed(true)}
    />
  );
}

export function MovieDetailContent({ movie }: MovieDetailContentProps) {
  const videoId = getYoutubeVideoId(movie.trailer_url);

  const streamingProviders = movie.movie_providers.filter(
    (p) => p.provider_type === "flatrate",
  );
  const paidProviders = movie.movie_providers.filter(
    (p) => p.provider_type === "rent" || p.provider_type === "buy",
  );

  return (
    <div className="space-y-6 p-4">
      {/* Meta info */}
      <div className="flex items-center gap-3 text-sm text-muted-foreground">
        <span>{movie.release_year}</span>
        <span className="flex items-center gap-1">
          <Star className="h-4 w-4 fill-yellow-500 text-yellow-500" />
          {movie.rating.toFixed(1)}
        </span>
        <span>{movie.runtime} min</span>
      </div>

      {/* Genres */}
      <div className="flex flex-wrap gap-2">
        {movie.genres.map((genre) => (
          <span
            key={genre}
            className="rounded-full bg-secondary px-3 py-1 text-xs"
          >
            {genre}
          </span>
        ))}
      </div>

      {/* Synopsis */}
      <div>
        <h3 className="mb-2 font-semibold">Synopsis</h3>
        <p className="text-sm leading-relaxed text-muted-foreground">
          {movie.synopsis}
        </p>
      </div>

      {/* Providers — inline after synopsis */}
      {streamingProviders.length > 0 && (
        <div>
          <h3 className="mb-2 font-semibold">Available on</h3>
          <div className="flex gap-4 overflow-x-auto pb-1">
            {streamingProviders.map((provider, idx) => (
              <div
                key={provider.name}
                className="flex shrink-0 flex-col items-center"
              >
                <div className="h-12 w-12 overflow-hidden rounded-xl shadow-sm">
                  <ProviderLogo name={provider.name} />
                </div>
                <span className="mt-1 max-w-[72px] truncate text-center text-[10px] text-muted-foreground">
                  {provider.name}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {paidProviders.length > 0 && (
        <div>
          <h3 className="mb-1 text-sm text-muted-foreground">Rent or Buy</h3>
          <div className="flex gap-3 overflow-x-auto pb-1">
            {paidProviders.map((provider, idx) => (
              <div
                key={provider.name}
                className="flex shrink-0 flex-col items-center opacity-70"
              >
                <div className="h-10 w-10 overflow-hidden rounded-xl shadow-sm">
                  <ProviderLogo name={provider.name} />
                </div>
                <span className="mt-1 max-w-[64px] truncate text-center text-[10px] text-muted-foreground">
                  {provider.name}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Trailer */}
      {videoId && (
        <div>
          <h3 className="mb-2 font-semibold">Trailer</h3>
          <div className="relative w-full" style={{ paddingBottom: "56.25%" }}>
            <iframe
              className="absolute inset-0 h-full w-full rounded-lg"
              src={`https://www.youtube.com/embed/${videoId}`}
              title="Trailer"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        </div>
      )}

      {/* Cast — horizontal scroll */}
      {movie.cast.length > 0 && (
        <div>
          <h3 className="mb-2 font-semibold">Cast</h3>
          <div className="flex gap-4 overflow-x-auto pb-2">
            {movie.cast.map((member) => (
              <div
                key={member.name}
                className="flex shrink-0 flex-col items-center"
              >
                {member.profile_path ? (
                  <img
                    src={member.profile_path}
                    alt={member.name}
                    className="h-12 w-12 rounded-full object-cover"
                  />
                ) : (
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-secondary text-xs">
                    {member.name.charAt(0)}
                  </div>
                )}
                <p className="mt-1 max-w-[72px] truncate text-center text-[11px] font-medium">
                  {member.name}
                </p>
                {member.role_type && (
                  <p className="max-w-[72px] truncate text-center text-[10px] text-muted-foreground">
                    {member.role_type}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface MovieDetailProps {
  movie: MovieDetails;
  onClose: () => void;
}

export function MovieDetail({ movie, onClose }: MovieDetailProps) {
  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-end justify-center"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <motion.div
        className="relative z-10 max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-t-2xl bg-card"
        initial={{ y: "100%" }}
        animate={{ y: 0 }}
        exit={{ y: "100%" }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
      >
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-card p-4">
          <h2 className="text-lg font-bold">{movie.title}</h2>
          <button
            onClick={onClose}
            className="rounded-full p-1 hover:bg-secondary"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <MovieDetailContent movie={movie} />
      </motion.div>
    </motion.div>
  );
}
