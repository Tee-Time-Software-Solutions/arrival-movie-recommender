import { useEffect, useRef, useState, useLayoutEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { EntityReference } from "@/types/movie";
import type { LinkedMovie } from "@/types/user";
import { getPersonLinkedMovies } from "@/services/api/people";

interface EntityPopupProps {
  entity: EntityReference | null;
  anchorRect: DOMRect | null;
  onClose: () => void;
}

const PERSON_TYPES = new Set(["Person", "Director", "Actor", "Writer"]);
const POPUP_WIDTH = 288;
const POPUP_MARGIN = 12;

const TYPE_LABELS: Record<string, string> = {
  Person: "Person",
  Director: "Director",
  Actor: "Actor",
  Writer: "Writer",
  Genre: "Genre",
  Movie: "Movie",
  Keyword: "Keyword",
  Collection: "Collection",
};

function computePosition(anchorRect: DOMRect, popupHeight: number) {
  let left = anchorRect.left + anchorRect.width / 2 - POPUP_WIDTH / 2;
  left = Math.max(
    POPUP_MARGIN,
    Math.min(left, window.innerWidth - POPUP_WIDTH - POPUP_MARGIN),
  );

  const spaceBelow = window.innerHeight - anchorRect.bottom - POPUP_MARGIN;
  const spaceAbove = anchorRect.top - POPUP_MARGIN;
  let top: number;
  let flipped = false;

  if (spaceBelow >= popupHeight || spaceBelow >= spaceAbove) {
    top = anchorRect.bottom + 6;
  } else {
    top = anchorRect.top - popupHeight - 6;
    flipped = true;
  }

  top = Math.max(
    POPUP_MARGIN,
    Math.min(top, window.innerHeight - popupHeight - POPUP_MARGIN),
  );

  return { left, top, flipped };
}

export function EntityPopup({ entity, anchorRect, onClose }: EntityPopupProps) {
  const popupRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{
    left: number;
    top: number;
    flipped: boolean;
  } | null>(null);
  const [linkedMovies, setLinkedMovies] = useState<LinkedMovie[]>([]);
  const [loadingMovies, setLoadingMovies] = useState(false);

  const isPerson = entity ? PERSON_TYPES.has(entity.entity_type) : false;

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

  // Fetch linked movies when a person entity is selected
  useEffect(() => {
    if (!entity || !isPerson) {
      setLinkedMovies([]);
      return;
    }

    setLoadingMovies(true);
    getPersonLinkedMovies(entity.tmdb_id)
      .then(setLinkedMovies)
      .catch(() => setLinkedMovies([]))
      .finally(() => setLoadingMovies(false));
  }, [entity, isPerson]);

  // Position after render
  useLayoutEffect(() => {
    if (!anchorRect || !popupRef.current) {
      setPosition(null);
      return;
    }
    const popupHeight = popupRef.current.offsetHeight;
    setPosition(computePosition(anchorRect, popupHeight));
  }, [anchorRect, entity, linkedMovies, loadingMovies]);

  return (
    <AnimatePresence>
      {entity && anchorRect && (
        <motion.div
          ref={popupRef}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 4 }}
          transition={{ duration: 0.15 }}
          className="fixed z-100 w-72 rounded-xl border border-border bg-card p-4 shadow-xl"
          style={{
            left: position?.left ?? -9999,
            top: position?.top ?? -9999,
            visibility: position ? "visible" : "hidden",
          }}
        >
          {/* Entity header */}
          <p className="text-sm font-semibold">{entity.name}</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {TYPE_LABELS[entity.entity_type] || entity.entity_type}
          </p>

          {/* Linked movies for person entities */}
          {isPerson && (
            <div className="mt-3">
              {loadingMovies ? (
                <div className="flex justify-center py-2">
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                </div>
              ) : linkedMovies.length > 0 ? (
                <>
                  <p className="mb-2 text-xs font-medium text-muted-foreground">
                    Appears in films you loved
                  </p>
                  <div className="flex gap-2 overflow-x-auto pb-1">
                    {linkedMovies.map((movie) => (
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
                </>
              ) : null}
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
