import { useState, useCallback, useRef } from "react";
import { motion } from "framer-motion";
import { EntityPopup } from "./EntityPopup";
import type { ExplanationResponse, EntityReference } from "@/types/movie";

interface ExplanationBannerProps {
  explanation: ExplanationResponse | null;
}

const MIN_DISPLAY_CONFIDENCE = 0.05;

export function ExplanationBanner({ explanation }: ExplanationBannerProps) {
  const [activeEntity, setActiveEntity] = useState<EntityReference | null>(null);
  const [anchorRect, setAnchorRect] = useState<DOMRect | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleEntityClick = useCallback(
    (entity: EntityReference, e: React.MouseEvent<HTMLSpanElement>) => {
      const rect = (e.target as HTMLElement).getBoundingClientRect();
      if (activeEntity?.tmdb_id === entity.tmdb_id) {
        setActiveEntity(null);
        setAnchorRect(null);
      } else {
        setActiveEntity(entity);
        setAnchorRect(rect);
      }
    },
    [activeEntity],
  );

  if (!explanation || explanation.confidence < MIN_DISPLAY_CONFIDENCE) {
    return null;
  }

  // Parse text: replace @EntityName with clickable spans
  const parts = parseExplanationText(explanation.text, explanation.entities);

  return (
    <>
      <motion.div
        ref={containerRef}
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="mb-2 rounded-lg bg-card/80 px-3 py-2 text-center text-xs backdrop-blur-sm"
      >
        {parts.map((part, i) =>
          part.entity ? (
            <span
              key={i}
              className="cursor-pointer font-bold text-blue-500 hover:underline"
              onClick={(e) => handleEntityClick(part.entity!, e)}
            >
              {part.text}
            </span>
          ) : (
            <span key={i}>{part.text}</span>
          ),
        )}
      </motion.div>

      <EntityPopup
        entity={activeEntity}
        anchorRect={anchorRect}
        onClose={() => {
          setActiveEntity(null);
          setAnchorRect(null);
        }}
      />
    </>
  );
}

interface TextPart {
  text: string;
  entity?: EntityReference;
}

function parseExplanationText(
  text: string,
  entities: EntityReference[],
): TextPart[] {
  const parts: TextPart[] = [];
  // Match @EntityName patterns — entity names from the entities list
  let remaining = text;

  // Sort entities by name length descending to match longest first
  const sortedEntities = [...entities].sort(
    (a, b) => b.name.length - a.name.length,
  );

  // Build a regex that matches any @EntityName
  const escapedNames = sortedEntities.map((e) =>
    e.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
  );

  if (escapedNames.length === 0) {
    return [{ text }];
  }

  const pattern = new RegExp(`@(${escapedNames.join("|")})`, "g");
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(remaining)) !== null) {
    // Add text before the match
    if (match.index > lastIndex) {
      parts.push({ text: remaining.slice(lastIndex, match.index) });
    }

    // Find the matching entity
    const matchedName = match[1];
    const entity = entities.find((e) => e.name === matchedName);
    parts.push({ text: matchedName, entity });

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < remaining.length) {
    parts.push({ text: remaining.slice(lastIndex) });
  }

  return parts;
}
