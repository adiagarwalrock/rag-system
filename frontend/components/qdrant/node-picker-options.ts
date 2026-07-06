export type SelectOption = { value: string; label: string; disabled?: boolean };

export function parserOptions(parsers?: Array<{ id: string; label: string; available: boolean }>) {
  const seen = new Set<string>();
  const options: SelectOption[] = [];
  for (const parser of parsers ?? []) {
    if (parser.id === "auto" || seen.has(parser.id)) continue;
    seen.add(parser.id);
    options.push({
      value: parser.id,
      label: parser.available ? parser.label : `${parser.label} unavailable`,
      disabled: !parser.available,
    });
  }
  return options;
}

export function facetOptions(facets?: Array<{ name: string; count: number }>, currentValue?: string) {
  const seen = new Set<string>();
  const options: SelectOption[] = [];
  for (const item of facets ?? []) {
    if (!item.name || seen.has(item.name)) continue;
    seen.add(item.name);
    options.push({ value: item.name, label: `${item.name} (${item.count})` });
  }
  if (currentValue && !seen.has(currentValue)) {
    options.unshift({ value: currentValue, label: currentValue });
  }
  return options;
}

export function mergeOptions(options: SelectOption[]) {
  const seen = new Set<string>();
  const merged: SelectOption[] = [];
  for (const option of options) {
    if (seen.has(option.value)) continue;
    seen.add(option.value);
    merged.push(option);
  }
  return merged;
}
