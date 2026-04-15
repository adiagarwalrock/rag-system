from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pymupdf as fitz


class PDFPipelineHelper:
    @staticmethod
    def dump_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                PDFPipelineHelper.to_jsonable(payload),
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def to_jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): PDFPipelineHelper.to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [PDFPipelineHelper.to_jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def normalize_block_text(text: str) -> str:
        lines = [
            PDFPipelineHelper.normalize_whitespace(line)
            for line in (text or "").splitlines()
        ]
        return "\n".join(line for line in lines if line)

    @staticmethod
    def has_numeric_data(text: str) -> bool:
        return len(re.findall(r"\b\d+[.,]?\d*%?\b", text or "")) >= 3

    @staticmethod
    def has_table_signals(text: str) -> bool:
        if not text:
            return False
        lines = text.strip().split("\n")
        structured_lines = sum(
            1 for line in lines if "\t" in line or line.count("|") >= 2
        )
        if structured_lines >= 2:
            return True
        if structured_lines == 1 and PDFPipelineHelper.has_numeric_data(text):
            return True
        lowered = (text or "").lower()
        if structured_lines >= 1 and re.search(
            r"\b(q[1-4]|quarter|year|total|revenue|cost|amount|value)\b", lowered
        ):
            return True
        return False

    @staticmethod
    def has_chart_signals(text: str) -> bool:
        lowered = (text or "").lower()
        return any(
            token in lowered
            for token in ("figure", "chart", "graph", "plot", "diagram", "trend")
        )

    @staticmethod
    def numeric_density(text: str) -> float:
        if not text:
            return 0.0
        tokens = re.findall(r"\S+", text)
        if not tokens:
            return 0.0
        numeric_tokens = len(re.findall(r"\b\d+[.,]?\d*%?\b", text))
        return round(min(1.0, numeric_tokens / len(tokens)), 4)

    @staticmethod
    def parse_table_like_text(text: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                cells = [
                    PDFPipelineHelper.normalize_whitespace(cell)
                    for cell in line.split("|")
                    if cell.strip()
                ]
            elif "\t" in line:
                cells = [
                    PDFPipelineHelper.normalize_whitespace(cell)
                    for cell in line.split("\t")
                    if cell.strip()
                ]
            else:
                cells = [
                    PDFPipelineHelper.normalize_whitespace(cell)
                    for cell in re.split(r"\s{2,}", line)
                    if cell
                ]
            if len(cells) >= 2:
                rows.append(cells)
        return rows

    @staticmethod
    def normalize_table_rows(rows: list[list[str]]) -> str:
        if not rows:
            return ""
        header = [cell.strip() for cell in rows[0]]
        lines = []
        for idx, row in enumerate(rows[1:], start=1):
            pairs = []
            for col_idx, value in enumerate(row):
                key = (
                    header[col_idx]
                    if col_idx < len(header)
                    else f"column_{col_idx + 1}"
                )
                pairs.append(f"{key}={value}")
            lines.append(f"row_{idx}: " + "; ".join(pairs))
        if not lines:
            lines.append("header: " + "; ".join(header))
        return "\n".join(lines)

    @staticmethod
    def table_to_markdown(rows: list[list[str]]) -> str:
        if not rows:
            return ""
        header = rows[0]
        output = [
            "|" + "|".join(cell or "" for cell in header) + "|",
            "|" + "|".join("---" for _ in header) + "|",
        ]
        for row in rows[1:]:
            padded = list(row) + [""] * max(0, len(header) - len(row))
            output.append("|" + "|".join(padded[: len(header)]) + "|")
        return "\n".join(output)

    @staticmethod
    def table_rows_to_html(rows: list[list[str]]) -> str:
        if not rows:
            return ""
        header = rows[0]
        body = rows[1:]
        head_html = "".join(
            f"<th>{PDFPipelineHelper.escape_html(cell)}</th>" for cell in header
        )
        body_rows = []
        for row in body:
            padded = list(row) + [""] * max(0, len(header) - len(row))
            body_rows.append(
                "<tr>"
                + "".join(
                    f"<td>{PDFPipelineHelper.escape_html(cell)}</td>"
                    for cell in padded[: len(header)]
                )
                + "</tr>"
            )
        return (
            "<table><thead><tr>"
            + head_html
            + "</tr></thead><tbody>"
            + "".join(body_rows)
            + "</tbody></table>"
        )

    @staticmethod
    def escape_html(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def extract_units(text: str) -> list[str]:
        units = set()
        lowered = text.lower()
        if "usd" in lowered:
            units.add("USD")
        if "$" in text:
            units.add("$")
        if "%" in text:
            units.add("%")
        if "million" in lowered:
            units.add("million")
        if "billion" in lowered:
            units.add("billion")
        return sorted(units)

    @staticmethod
    def to_float_bbox(bbox: Any) -> list[float]:
        if bbox is None:
            return [0.0, 0.0, 0.0, 0.0]
        if isinstance(bbox, fitz.Rect):
            return [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        return [0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def union_bbox(bboxes: list[list[float]]) -> list[float] | None:
        if not bboxes:
            return None
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        return [float(x0), float(y0), float(x1), float(y1)]

    @staticmethod
    def header_signature(headers: list[str]) -> str:
        return "|".join(
            PDFPipelineHelper.normalize_whitespace(header).lower() for header in headers
        )


def dump_json(path: Path, payload: Any) -> None:
    PDFPipelineHelper.dump_json(path, payload)


def normalize_whitespace(text: str) -> str:
    return PDFPipelineHelper.normalize_whitespace(text)


def normalize_block_text(text: str) -> str:
    return PDFPipelineHelper.normalize_block_text(text)


def has_numeric_data(text: str) -> bool:
    return PDFPipelineHelper.has_numeric_data(text)


def has_table_signals(text: str) -> bool:
    return PDFPipelineHelper.has_table_signals(text)


def has_chart_signals(text: str) -> bool:
    return PDFPipelineHelper.has_chart_signals(text)


def numeric_density(text: str) -> float:
    return PDFPipelineHelper.numeric_density(text)


def parse_table_like_text(text: str) -> list[list[str]]:
    return PDFPipelineHelper.parse_table_like_text(text)


def normalize_table_rows(rows: list[list[str]]) -> str:
    return PDFPipelineHelper.normalize_table_rows(rows)


def table_to_markdown(rows: list[list[str]]) -> str:
    return PDFPipelineHelper.table_to_markdown(rows)


def table_rows_to_html(rows: list[list[str]]) -> str:
    return PDFPipelineHelper.table_rows_to_html(rows)


def extract_units(text: str) -> list[str]:
    return PDFPipelineHelper.extract_units(text)


def to_float_bbox(bbox: Any) -> list[float]:
    return PDFPipelineHelper.to_float_bbox(bbox)


def union_bbox(bboxes: list[list[float]]) -> list[float] | None:
    return PDFPipelineHelper.union_bbox(bboxes)


def header_signature(headers: list[str]) -> str:
    return PDFPipelineHelper.header_signature(headers)
