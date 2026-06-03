import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

function parseTimestamp(value: string): Date {
  // If the string has no timezone info, treat it as UTC (the server stores UTC).
  const hasOffset = /[Z+\-]\d*$/.test(value.trimEnd());
  return new Date(hasOffset ? value : `${value}Z`);
}

export function formatDate(value?: string | null) {
  if (!value) return "-";
  const date = parseTimestamp(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function formatMessageTimestamp(value?: string) {
  if (!value) return "";
  const date = parseTimestamp(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatNumber(value?: number | null) {
  if (value === null || value === undefined) return "-";
  return new Intl.NumberFormat().format(value);
}

export function truncate(value: string, max = 96) {
  const clean = value.replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}...` : clean;
}

export function scoreLabel(value?: number | null) {
  if (typeof value !== "number") return "-";
  return value.toFixed(value >= 10 ? 0 : 3);
}

export function firstParam(value: string | string[] | undefined) {
  return Array.isArray(value) ? value[0] ?? "" : value ?? "";
}

export function arrayParam(value: string | string[] | undefined) {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

export function downloadJson(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
