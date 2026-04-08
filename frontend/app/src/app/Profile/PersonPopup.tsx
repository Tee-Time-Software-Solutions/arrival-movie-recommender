import { useEffect, useRef, useState, useLayoutEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { TopPerson } from "@/types/user";

const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185";
const POPUP_WIDTH = 288; // w-72
const POPUP_MARGIN = 12;

interface PersonPopupProps {
  person: TopPerson | null;
  anchorRect: DOMRect | null;
  onClose: () => void;
}

function computePosition(anchorRect: DOMRect, popupHeight: number) {
  // Horizontal: center on anchor, clamp to viewport
  let left = anchorRect.left + anchorRect.width / 2 - POPUP_WIDTH / 2;
  left = Math.max(POPUP_MARGIN, Math.min(left, window.innerWidth - POPUP_WIDTH - POPUP_MARGIN));

  // Vertical: prefer below, flip above if not enough space
  const spaceBelow = window.innerHeight - anchorRect.bottom - POPUP_MARGIN;
  const spaceAbove = anchorRect.top - POPUP_MARGIN;
  let top: number;
  let flipped = false;

  if (spaceBelow >= popupHeight || spaceBelow >= spaceAbove) {
    top = anchorRect.bottom + 8;
  } else {
    top = anchorRect.top - popupHeight - 8;
    flipped = true;
  }

  // Final clamp so it never goes off-screen vertically
  top = Math.max(POPUP_MARGIN, Math.min(top, window.innerHeight - popupHeight - POPUP_MARGIN));

  return { left, top, flipped };
}

export function PersonPopup({ person, anchorRect, onClose }: PersonPopupProps) {
  const popupRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ left: number; top: number; flipped: boolean } | null>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [onClose]);

  // Measure popup after render, then position
  useLayoutEffect(() => {
    if (!anchorRect || !popupRef.current) {
      setPosition(null);
      return;
    }
    const popupHeight = popupRef.current.offsetHeight;
    setPosition(computePosition(anchorRect, popupHeight));
  }, [anchorRect, person]);

  return (
    <AnimatePresence>
      {person && anchorRect && (
        <motion.div
          ref={popupRef}
          initial={{ opacity: 0, y: position?.flipped ? -4 : 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: position?.flipped ? -4 : 4 }}
          transition={{ duration: 0.15 }}
          className="fixed z-100 w-72 rounded-xl border border-border bg-card p-4 shadow-xl"
          style={{
            left: position?.left ?? -9999,
            top: position?.top ?? -9999,
            visibility: position ? "visible" : "hidden",
          }}
        >
          {/* Person header */}
          <div className="mb-3 flex items-center gap-3">
            {person.image_url ? (
              <img
                src={`${TMDB_IMAGE_BASE}${person.image_url}`}
                alt={person.name}
                className="h-12 w-12 rounded-full object-cover"
              />
            ) : (
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-lg font-semibold text-primary">
                {person.name[0]?.toUpperCase() ?? "?"}
              </div>
            )}
            <div>
              <p className="text-sm font-semibold">{person.name}</p>
              <p className="text-xs text-muted-foreground">
                {person.entity_type}
              </p>
            </div>
          </div>

          {/* Linked movies */}
          {person.linked_movies.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-medium text-muted-foreground">
                Appears in films you loved
              </p>
              <div className="flex gap-2 overflow-x-auto pb-1">
                {person.linked_movies.map((movie) => (
                  <div
                    key={movie.tmdb_id}
                    className="flex shrink-0 flex-col items-center"
                  >
                    {movie.poster_url ? (
                      <img
                        src={movie.poster_url}
                        alt={movie.title}
                        className="h-20 w-14 rounded-md object-cover"
                      />
                    ) : (
                      <div className="flex h-20 w-14 items-center justify-center rounded-md bg-secondary text-[10px] text-muted-foreground">
                        {movie.title.slice(0, 6)}
                      </div>
                    )}
                    <p className="mt-1 w-14 truncate text-center text-[10px] text-muted-foreground">
                      {movie.title}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {person.linked_movies.length === 0 && (
            <p className="text-xs text-muted-foreground">
              Based on your overall taste profile
            </p>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
