"use client";

import { DarkSelect } from "@/components/common/dark-select";

export function RetrievalModeSelect({
  value,
  onChange,
}: {
  value: "auto" | "hybrid" | "dense_only";
  onChange: (value: "auto" | "hybrid" | "dense_only") => void;
}) {
  return (
    <DarkSelect
      label="Retrieval mode"
      value={value}
      onChange={onChange}
      options={[
        { value: "auto", label: "auto" },
        { value: "hybrid", label: "hybrid" },
        { value: "dense_only", label: "dense_only" },
      ]}
    />
  );
}
