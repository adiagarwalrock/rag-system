"use client";

export function RetrievalModeSelect({
  value,
  onChange,
}: {
  value: "auto" | "hybrid" | "dense_only";
  onChange: (value: "auto" | "hybrid" | "dense_only") => void;
}) {
  return (
    <select className="control w-full" value={value} onChange={(event) => onChange(event.target.value as "auto" | "hybrid" | "dense_only")} aria-label="Retrieval mode">
      <option value="auto">auto</option>
      <option value="hybrid">hybrid</option>
      <option value="dense_only">dense_only</option>
    </select>
  );
}
