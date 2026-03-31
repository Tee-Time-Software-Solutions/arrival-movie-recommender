import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { EntityReference } from "@/types/movie";

interface EntityPopupProps {
  entity: EntityReference | null;
  anchorRect: DOMRect | null;
  onClose: () => void;
}

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

export function EntityPopup({ entity, anchorRect, onClose }: EntityPopupProps) {
  const popupRef = useRef<HTMLDivElement>(null);

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

  return (
    <AnimatePresence>
      {entity && anchorRect && (
        <motion.div
          ref={popupRef}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 4 }}
          transition={{ duration: 0.15 }}
          className="fixed z-[100] rounded-lg border border-border bg-card px-4 py-3 shadow-lg"
          style={{
            left: Math.min(anchorRect.left, window.innerWidth - 200),
            top: anchorRect.bottom + 6,
          }}
        >
          <p className="text-sm font-semibold">{entity.name}</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {TYPE_LABELS[entity.entity_type] || entity.entity_type}
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
