import { useState, useEffect } from "react";
import {
  motion,
  useMotionValue,
  useTransform,
  type PanInfo,
} from "framer-motion";
import { Star } from "lucide-react";
import type { MovieDetails } from "@/types/movie";
import { SWIPE_THRESHOLD, CARD_ROTATION_FACTOR } from "@/lib/constants";

interface MovieCardProps {
  movie: MovieDetails;
  onLike: () => void;
  onDislike: () => void;
  onExpand: () => void;
  onRate: () => void;
  isTop: boolean;
  forceSwipe?: "left" | "right" | null;
}

const CARD_SPRING = { type: "spring" as const, stiffness: 300, damping: 30 };

export function MovieCard({
  movie,
  onLike,
  onDislike,
  onExpand,
  onRate,
  isTop,
  forceSwipe,
}: MovieCardProps) {
  const [exitX, setExitX] = useState(0);
  const [swiped, setSwiped] = useState(false);

  // Trigger swipe animation from keyboard
  useEffect(() => {
    if (!forceSwipe || !isTop) return;
    setExitX(forceSwipe === "right" ? 500 : -500);
    setSwiped(true);
  }, [forceSwipe, isTop]);

  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const rotate = useTransform(
    x,
    [-300, 300],
    [-30 * CARD_ROTATION_FACTOR * 10, 30 * CARD_ROTATION_FACTOR * 10],
  );
  const likeOpacity = useTransform(x, [0, SWIPE_THRESHOLD], [0, 1]);
  const dislikeOpacity = useTransform(x, [-SWIPE_THRESHOLD, 0], [1, 0]);
  const detailsOpacity = useTransform(y, [-SWIPE_THRESHOLD, 0], [1, 0]);
  const rateOpacity = useTransform(y, [0, SWIPE_THRESHOLD], [0, 1]);

  const handleDragEnd = (_: unknown, info: PanInfo) => {
    const { offset, velocity } = info;
    const absX = Math.abs(offset.x);
    const absY = Math.abs(offset.y);

    if (absX > absY) {
      if (offset.x > SWIPE_THRESHOLD || velocity.x > 500) {
        setExitX(500);
        setSwiped(true);
        onLike();
      } else if (offset.x < -SWIPE_THRESHOLD || velocity.x < -500) {
        setExitX(-500);
        setSwiped(true);
        onDislike();
      }
    } else {
      if (offset.y < -SWIPE_THRESHOLD || velocity.y < -500) {
        onExpand();
      } else if (offset.y > SWIPE_THRESHOLD || velocity.y > 500) {
        onRate();
      }
    }
  };

  if (!isTop) return null;

  const exitRotation = exitX > 0 ? 20 : -20;

  return (
    <motion.div
      className="absolute inset-0 cursor-grab active:cursor-grabbing"
      style={{ x, y, rotate }}
      drag
      dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
      dragElastic={0.8}
      onDragEnd={handleDragEnd}
      initial={{ scale: 0.92, opacity: 0 }}
      animate={
        swiped
          ? { x: exitX, opacity: 0, rotate: exitRotation, scale: 1 }
          : { scale: 1, opacity: 1 }
      }
      transition={CARD_SPRING}
    >
      {/* LIKE stamp */}
      <motion.div
        className="pointer-events-none absolute left-6 top-1/2 z-10 -translate-y-1/2 rounded-xl border-4 border-green-500 bg-green-500/10 px-4 py-2"
        style={forceSwipe === null || forceSwipe === undefined ? { opacity: likeOpacity } : undefined}
        animate={forceSwipe ? { opacity: forceSwipe === "right" ? 1 : 0 } : undefined}
      >
        <span className="text-2xl font-bold text-green-500">LIKE</span>
      </motion.div>

      {/* NOPE stamp */}
      <motion.div
        className="pointer-events-none absolute right-6 top-1/2 z-10 -translate-y-1/2 rounded-xl border-4 border-red-500 bg-red-500/10 px-4 py-2"
        style={forceSwipe === null || forceSwipe === undefined ? { opacity: dislikeOpacity } : undefined}
        animate={forceSwipe ? { opacity: forceSwipe === "left" ? 1 : 0 } : undefined}
      >
        <span className="text-2xl font-bold text-red-500">NOPE</span>
      </motion.div>

      {/* INFO stamp */}
      <motion.div
        className="pointer-events-none absolute left-1/2 top-6 z-10 -translate-x-1/2 rounded-xl border-4 border-blue-500 bg-blue-500/10 px-4 py-2"
        style={{ opacity: detailsOpacity }}
      >
        <span className="text-2xl font-bold text-blue-500">INFO</span>
      </motion.div>

      {/* RATE stamp */}
      <motion.div
        className="pointer-events-none absolute bottom-6 left-1/2 z-10 -translate-x-1/2 rounded-xl border-4 border-yellow-500 bg-yellow-500/10 px-4 py-2"
        style={{ opacity: rateOpacity }}
      >
        <span className="text-2xl font-bold text-yellow-500">RATE</span>
      </motion.div>

      {/* Card content */}
      <div className="h-full overflow-hidden rounded-2xl bg-card shadow-2xl">
        <div className="relative h-[75%]">
          <img
            src={movie.poster_url}
            alt={movie.title}
            className="h-full w-full object-cover select-none"
            draggable={false}
          />
          <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-6 pt-20">
            <h2 className="text-2xl font-bold text-white">{movie.title}</h2>
          </div>
        </div>
        <div className="p-4">
          <div className="mb-3 flex items-center gap-3 text-sm text-muted-foreground">
            <span>{movie.release_year}</span>
            <span className="flex items-center gap-1">
              <Star className="h-4 w-4 fill-yellow-500 text-yellow-500" />
              {movie.rating.toFixed(1)}
            </span>
            <span>{movie.runtime} min</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {movie.genres.slice(0, 4).map((genre) => (
              <span
                key={genre}
                className="rounded-full bg-secondary px-2.5 py-0.5 text-xs text-secondary-foreground"
              >
                {genre}
              </span>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
