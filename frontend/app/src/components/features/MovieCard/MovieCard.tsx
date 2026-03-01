import { useState, useEffect, useRef, useCallback } from "react";
import {
  motion,
  useMotionValue,
  useTransform,
  type PanInfo,
} from "framer-motion";
import { Star, Zap } from "lucide-react";
import type { MovieDetails } from "@/types/movie";
import { SWIPE_THRESHOLD, CARD_ROTATION_FACTOR } from "@/lib/constants";

interface MovieCardProps {
  movie: MovieDetails;
  onLike: () => void;
  onDislike: () => void;
  onSuperLike: () => void;
  onSuperDislike: () => void;
  onExpand: () => void;
  onWatched: () => void;
  isTop: boolean;
  forceSwipe?: "left" | "right" | "down" | null;
}

const CARD_SPRING = { type: "spring" as const, stiffness: 300, damping: 30 };
const HOLD_DURATION = 600;

export function MovieCard({
  movie,
  onLike,
  onDislike,
  onSuperLike,
  onSuperDislike,
  onExpand,
  onWatched,
  isTop,
  forceSwipe,
}: MovieCardProps) {
  const [exitX, setExitX] = useState(0);
  const [exitY, setExitY] = useState(0);
  const [swiped, setSwiped] = useState(false);
  const [supercharged, setSupercharged] = useState(false);

  // Long-press detection
  const holdTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const superchargedRef = useRef(false);

  const clearHoldTimer = useCallback(() => {
    if (holdTimerRef.current) {
      clearTimeout(holdTimerRef.current);
      holdTimerRef.current = null;
    }
  }, []);

  // Trigger swipe animation from keyboard / force
  useEffect(() => {
    if (!forceSwipe || !isTop) return;
    if (forceSwipe === "down") {
      setExitY(500);
      setSwiped(true);
    } else {
      setExitX(forceSwipe === "right" ? 500 : -500);
      setSwiped(true);
    }
  }, [forceSwipe, isTop]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => clearHoldTimer();
  }, [clearHoldTimer]);

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
  const watchedOpacity = useTransform(y, [0, SWIPE_THRESHOLD], [0, 1]);

  const handlePointerDown = () => {
    holdTimerRef.current = setTimeout(() => {
      superchargedRef.current = true;
      setSupercharged(true);
    }, HOLD_DURATION);
  };

  const handlePointerUp = () => {
    clearHoldTimer();
  };

  const handleDragStart = () => {
    clearHoldTimer();
  };

  const handleDragEnd = (_: unknown, info: PanInfo) => {
    const { offset, velocity } = info;
    const absX = Math.abs(offset.x);
    const absY = Math.abs(offset.y);
    const charged = superchargedRef.current;

    if (absX > absY) {
      if (offset.x > SWIPE_THRESHOLD || velocity.x > 500) {
        setExitX(500);
        setSwiped(true);
        charged ? onSuperLike() : onLike();
      } else if (offset.x < -SWIPE_THRESHOLD || velocity.x < -500) {
        setExitX(-500);
        setSwiped(true);
        charged ? onSuperDislike() : onDislike();
      }
    } else {
      if (offset.y < -SWIPE_THRESHOLD || velocity.y < -500) {
        onExpand();
      } else if (offset.y > SWIPE_THRESHOLD || velocity.y > 500) {
        setExitY(500);
        setSwiped(true);
        onWatched();
      }
    }
  };

  if (!isTop) return null;

  const exitRotation = exitX > 0 ? 20 : exitX < 0 ? -20 : 0;

  const getAnimateState = () => {
    if (swiped) {
      return { x: exitX, y: exitY, opacity: 0, rotate: exitRotation, scale: 1 };
    }
    return { scale: 1, opacity: 1 };
  };

  return (
    <motion.div
      className="absolute inset-0 cursor-grab active:cursor-grabbing"
      style={{ x, y, rotate }}
      drag
      dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
      dragElastic={0.8}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
      initial={{ scale: 0.92, opacity: 0 }}
      animate={getAnimateState()}
      transition={CARD_SPRING}
    >
      {/* LIKE / SUPER LIKE stamp */}
      <motion.div
        className={`pointer-events-none absolute left-6 top-1/2 z-10 -translate-y-1/2 rounded-xl border-4 px-4 py-2 ${
          supercharged
            ? "border-yellow-400 bg-yellow-400/20"
            : "border-primary bg-primary/10"
        }`}
        style={forceSwipe === null || forceSwipe === undefined ? { opacity: likeOpacity } : undefined}
        animate={forceSwipe ? { opacity: forceSwipe === "right" ? 1 : 0 } : undefined}
      >
        {supercharged ? (
          <span className="flex items-center gap-1 text-2xl font-bold text-yellow-400">
            <Zap className="h-6 w-6 fill-yellow-400" />
            SUPER LIKE
          </span>
        ) : (
          <span className="text-2xl font-bold text-primary">LIKE</span>
        )}
      </motion.div>

      {/* NOPE / SUPER NOPE stamp */}
      <motion.div
        className={`pointer-events-none absolute right-6 top-1/2 z-10 -translate-y-1/2 rounded-xl border-4 px-4 py-2 ${
          supercharged
            ? "border-orange-500 bg-orange-500/20"
            : "border-red-500 bg-red-500/10"
        }`}
        style={forceSwipe === null || forceSwipe === undefined ? { opacity: dislikeOpacity } : undefined}
        animate={forceSwipe ? { opacity: forceSwipe === "left" ? 1 : 0 } : undefined}
      >
        {supercharged ? (
          <span className="flex items-center gap-1 text-2xl font-bold text-orange-500">
            <Zap className="h-6 w-6 fill-orange-500" />
            SUPER NOPE
          </span>
        ) : (
          <span className="text-2xl font-bold text-red-500">NOPE</span>
        )}
      </motion.div>

      {/* INFO stamp */}
      <motion.div
        className="pointer-events-none absolute left-1/2 top-6 z-10 -translate-x-1/2 rounded-xl border-4 border-blue-500 bg-blue-500/10 px-4 py-2"
        style={{ opacity: detailsOpacity }}
      >
        <span className="text-2xl font-bold text-blue-500">INFO</span>
      </motion.div>

      {/* WATCHED stamp */}
      <motion.div
        className="pointer-events-none absolute bottom-6 left-1/2 z-10 -translate-x-1/2 rounded-xl border-4 border-violet-500 bg-violet-500/10 px-4 py-2"
        style={forceSwipe === null || forceSwipe === undefined ? { opacity: watchedOpacity } : undefined}
        animate={forceSwipe ? { opacity: forceSwipe === "down" ? 1 : 0 } : undefined}
      >
        <span className="text-2xl font-bold text-violet-500">WATCHED</span>
      </motion.div>

      {/* Supercharged glow overlay */}
      <motion.div
        className="pointer-events-none absolute inset-0 z-0 rounded-2xl ring-4 ring-yellow-400/60"
        initial={{ opacity: 0 }}
        animate={{
          opacity: supercharged ? 1 : 0,
          boxShadow: supercharged
            ? "0 0 30px rgba(250, 204, 21, 0.4), 0 0 60px rgba(250, 204, 21, 0.2)"
            : "0 0 0px transparent",
        }}
        transition={{ duration: 0.3 }}
      />

      {/* Supercharged indicator */}
      {supercharged && (
        <motion.div
          className="pointer-events-none absolute left-1/2 top-1/2 z-20 -translate-x-1/2 -translate-y-1/2"
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 400, damping: 15 }}
        >
          <div className="flex items-center gap-2 rounded-full bg-black/60 px-5 py-2.5">
            <Zap className="h-6 w-6 fill-yellow-400 text-yellow-400" />
            <span className="text-lg font-bold text-yellow-400">SUPERCHARGED</span>
          </div>
        </motion.div>
      )}

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
              {(movie.rating ?? 0).toFixed(1)}
            </span>
            <span>{movie.runtime} min</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {(movie.genres ?? []).slice(0, 4).map((genre) => (
              <span
                key={genre}
                className="rounded-full bg-primary/10 px-2.5 py-0.5 text-xs text-primary"
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
