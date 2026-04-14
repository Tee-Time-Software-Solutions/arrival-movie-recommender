import { useState, useCallback } from "react";
import type { TopPerson, TopPeopleResponse } from "@/types/user";
import { PersonPopup } from "./PersonPopup";

const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185";

function PersonCard({
  person,
  onClick,
}: {
  person: TopPerson;
  onClick: (person: TopPerson, rect: DOMRect) => void;
}) {
  const imageUrl = person.image_url
    ? `${TMDB_IMAGE_BASE}${person.image_url}`
    : null;

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    onClick(person, rect);
  };

  return (
    <button
      onClick={handleClick}
      className="flex shrink-0 cursor-pointer flex-col items-center gap-1.5 transition hover:opacity-80"
    >
      {imageUrl ? (
        <img
          src={imageUrl}
          alt={person.name}
          className="h-16 w-16 rounded-full object-cover"
        />
      ) : (
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-lg font-semibold text-primary">
          {person.name[0]?.toUpperCase() ?? "?"}
        </div>
      )}
      <p className="w-20 truncate text-center text-xs text-foreground">
        {person.name}
      </p>
    </button>
  );
}

function PeopleRow({
  label,
  people,
  onPersonClick,
}: {
  label: string;
  people: TopPerson[];
  onPersonClick: (person: TopPerson, rect: DOMRect) => void;
}) {
  if (people.length === 0) return null;
  return (
    <div className="mb-4">
      <h3 className="mb-2 text-sm font-medium text-muted-foreground">
        {label}
      </h3>
      <div className="flex gap-4 overflow-x-auto pb-2">
        {people.map((person) => (
          <PersonCard
            key={person.tmdb_id}
            person={person}
            onClick={onPersonClick}
          />
        ))}
      </div>
    </div>
  );
}

export function TopPeopleSection({ data }: { data: TopPeopleResponse }) {
  const [selectedPerson, setSelectedPerson] = useState<TopPerson | null>(null);
  const [anchorRect, setAnchorRect] = useState<DOMRect | null>(null);

  const hasData =
    data.directors.length > 0 ||
    data.actors.length > 0 ||
    data.writers.length > 0;

  const handlePersonClick = useCallback(
    (person: TopPerson, rect: DOMRect) => {
      if (selectedPerson?.tmdb_id === person.tmdb_id) {
        setSelectedPerson(null);
        setAnchorRect(null);
      } else {
        setSelectedPerson(person);
        setAnchorRect(rect);
      }
    },
    [selectedPerson],
  );

  const handleClosePopup = useCallback(() => {
    setSelectedPerson(null);
    setAnchorRect(null);
  }, []);

  if (!hasData) return null;

  return (
    <div>
      <h2 className="mb-3 text-lg font-semibold">Your Favorite People</h2>
      <PeopleRow
        label="Directors"
        people={data.directors}
        onPersonClick={handlePersonClick}
      />
      <PeopleRow
        label="Actors"
        people={data.actors}
        onPersonClick={handlePersonClick}
      />
      <PeopleRow
        label="Writers"
        people={data.writers}
        onPersonClick={handlePersonClick}
      />
      <PersonPopup
        person={selectedPerson}
        anchorRect={anchorRect}
        onClose={handleClosePopup}
      />
    </div>
  );
}
