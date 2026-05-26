"use client";

export function ReasoningEffortSelect({
  value,
  onChange,
}: {
  value: "low" | "medium" | "high";
  onChange: (value: "low" | "medium" | "high") => void;
}) {
  return (
    <select className="control w-full" value={value} onChange={(event) => onChange(event.target.value as "low" | "medium" | "high")} aria-label="Reasoning effort">
      <option value="low">low</option>
      <option value="medium">medium</option>
      <option value="high">high</option>
    </select>
  );
}
