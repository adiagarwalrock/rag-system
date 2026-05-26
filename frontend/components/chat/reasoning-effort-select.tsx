"use client";

import { DarkSelect } from "@/components/common/dark-select";

export function ReasoningEffortSelect({
  value,
  onChange,
}: {
  value: "low" | "medium" | "high";
  onChange: (value: "low" | "medium" | "high") => void;
}) {
  return (
    <DarkSelect
      label="Reasoning effort"
      value={value}
      onChange={onChange}
      options={[
        { value: "low", label: "low" },
        { value: "medium", label: "medium" },
        { value: "high", label: "high" },
      ]}
    />
  );
}
