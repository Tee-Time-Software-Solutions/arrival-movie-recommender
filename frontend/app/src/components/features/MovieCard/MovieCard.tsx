import { useState, useEffect, useRef, useCallback } from "react";
import {
  motion,
  useMotionValue,
  useTransform,
  type PanInfo,
} from "framer-motion";
import { Star, Heart } from "lucide-react";
import type { MovieDetails } from "@/types/movie";
import { SWIPE_THRESHOLD, CARD_ROTATION_FACTOR } from "@/lib/constants";

interface MovieCardProps {
  movie: MovieDetails;
  onLike: () => void;
  onDislike: () => void;
  onExpand: () => void;
  onWatched: () => void;
  onLove: () => void;
  isTop: boolean;
  forceSwipe?: "left" | "right" | "down" | null;
}

const CARD_SPRING = { type: "spring" as const, stiffness: 300, damping: 30 };
const HOLD_DURATION = 600;

export function MovieCard({
  movie,
  onLike,
  onDislike,
  onExpand,
  onWatched,
  onLove,
  isTop,
  forceSwipe,
}: MovieCardProps) {
  const [exitX, setExitX] = useState(0);
  const [exitY, setExitY] = useState(0);
  const [swiped, setSwiped] = useState(false);
  const [loved, setLoved] = useState(false);

  // Long-press detection
  const holdTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isDraggingRef = useRef(false);
  const lovedRef = useRef(false);

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
    isDraggingRef.current = false;
    lovedRef.current = false;
    holdTimerRef.current = setTimeout(() => {
      if (!isDraggingRef.current) {
        lovedRef.current = true;
        setLoved(true);
        // After the love animation plays, trigger the callback
        setTimeout(() => {
          onLove();
        }, 500);
      }
    }, HOLD_DURATION);
  };

  const handlePointerUp = () => {
    clearHoldTimer();
  };

  const handleDragStart = () => {
    isDraggingRef.current = true;
    clearHoldTimer();
  };

  const handleDragEnd = (_: unknown, info: PanInfo) => {
    // If loved was triggered during hold, ignore drag end
    if (lovedRef.current) return;

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
        setExitY(500);
        setSwiped(true);
        onWatched();
      }
    }
  };

  if (!isTop) return null;

  const exitRotation = exitX > 0 ? 20 : exitX < 0 ? -20 : 0;

  // Build the animate target
  const getAnimateState = () => {
    if (loved) {
      return { scale: 0.8, opacity: 0 };
    }
    if (swiped) {
      return { x: exitX, y: exitY, opacity: 0, rotate: exitRotation, scale: 1 };
    }
    return { scale: 1, opacity: 1 };
  };

  return (
    <motion.div
      className="absolute inset-0 cursor-grab active:cursor-grabbing"
      style={{ x, y, rotate }}
      drag={!loved}
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
      {/* LIKE stamp */}
      <motion.div
        className="pointer-events-none absolute left-6 top-1/2 z-10 -translate-y-1/2 rounded-xl border-4 border-primary bg-primary/10 px-4 py-2"
        style={forceSwipe === null || forceSwipe === undefined ? { opacity: likeOpacity } : undefined}
        animate={forceSwipe ? { opacity: forceSwipe === "right" ? 1 : 0 } : undefined}
      >
        <span className="text-2xl font-bold text-primary">LIKE</span>
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

      {/* WATCHED stamp */}
      <motion.div
        className="pointer-events-none absolute bottom-6 left-1/2 z-10 -translate-x-1/2 rounded-xl border-4 border-violet-500 bg-violet-500/10 px-4 py-2"
        style={forceSwipe === null || forceSwipe === undefined ? { opacity: watchedOpacity } : undefined}
        animate={forceSwipe ? { opacity: forceSwipe === "down" ? 1 : 0 } : undefined}
      >
        <span className="text-2xl font-bold text-violet-500">WATCHED</span>
      </motion.div>

      {/* YOU LOVED IT overlay */}
      <motion.div
        className="pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center rounded-2xl bg-black/60"
        initial={{ opacity: 0 }}
        animate={{ opacity: loved ? 1 : 0 }}
        transition={{ duration: 0.2 }}
      >
        <Heart className="mb-2 h-16 w-16 fill-pink-500 text-pink-500" />
        <span className="text-2xl font-bold text-pink-500">YOU LOVED IT</span>
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
