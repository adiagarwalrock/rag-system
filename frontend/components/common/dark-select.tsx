"use client";

import { Check, ChevronDown } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

export type DarkSelectOption<T extends string = string> = {
  value: T;
  label: string;
  disabled?: boolean;
  hint?: string;
};

export function DarkSelect<T extends string>({
  label,
  value,
  options,
  onChange,
  placeholder = "Select",
  disabled,
  className,
  buttonClassName,
  menuClassName,
  menuSide = "bottom",
}: {
  label: string;
  value: T;
  options: Array<DarkSelectOption<T>>;
  onChange: (value: T) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  buttonClassName?: string;
  menuClassName?: string;
  menuSide?: "top" | "bottom";
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const activeOption = options.find((option) => option.value === value);

  useEffect(() => {
    function handlePointerDown(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  return (
    <div ref={rootRef} className={cn("relative", className)}>
      <button
        type="button"
        className={cn(
          "inline-flex h-9 w-full items-center justify-between gap-2 rounded-full border border-border bg-background px-3 text-left text-xs text-muted-foreground outline-none transition hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50",
          open && "border-primary/50 bg-muted text-foreground",
          buttonClassName,
        )}
        aria-label={label}
        aria-expanded={open}
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
      >
        <span className="min-w-0 truncate">{activeOption?.label ?? placeholder}</span>
        <ChevronDown className={cn("h-3.5 w-3.5 shrink-0 transition", open && "rotate-180")} />
      </button>
      {open ? (
        <div
          className={cn(
            "absolute z-50 max-h-72 min-w-full overflow-y-auto rounded-xl border border-border bg-[#111318] p-1 shadow-2xl shadow-black/50",
            menuSide === "top" ? "bottom-11 right-0" : "left-0 top-11",
            menuClassName,
          )}
        >
          {options.map((option) => (
            <button
              key={option.value}
              type="button"
              disabled={option.disabled}
              className={cn(
                "flex w-full items-center justify-between gap-3 rounded-lg px-3 py-2 text-left text-xs text-muted-foreground transition hover:bg-muted hover:text-foreground",
                option.value === value && "bg-muted text-foreground",
                option.disabled && "cursor-not-allowed opacity-40 hover:bg-transparent hover:text-muted-foreground",
              )}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
            >
              <span className="min-w-0 truncate">{option.label}</span>
              <span className="shrink-0">
                {option.value === value ? <Check className="h-3.5 w-3.5 text-blue-300" /> : null}
                {option.disabled && option.hint ? <span className="text-[10px] text-muted-foreground/60">{option.hint}</span> : null}
              </span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
