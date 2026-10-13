"""Offline command-line masking for UTF-8 text, JSON and CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from ko_pii_guard.analyzer import SUPPORTED_ENTITIES, KoreanPIIGuard


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _process_text(input_stream: TextIO, output_stream: TextIO,
                  guard: KoreanPIIGuard, style: str, strict_fields: bool) -> None:
    output_stream.write(guard.mask(input_stream.read(), style=style))


def _process_jsonl(input_stream: TextIO, output_stream: TextIO,
                   guard: KoreanPIIGuard, style: str, strict_fields: bool) -> None:
    for number, line in enumerate(input_stream, 1):
        if number == 1:
            line = line.removeprefix("\ufeff")
        if not line.strip():
            raise ValueError(f"invalid JSON Lines record at line {number}: empty line")
        try:
            value = json.loads(line, parse_constant=_reject_constant,
                               object_pairs_hook=_unique_object)
            masked = guard.mask_json(value, style=style, strict_fields=strict_fields)
            output_stream.write(json.dumps(masked, ensure_ascii=False, allow_nan=False) + "\n")
        except (ValueError, TypeError, RecursionError, json.JSONDecodeError) as exc:
            # Do not include exception strings: upstream errors can contain PII.
            raise ValueError(f"invalid or unmaskable JSON Lines record at line {number}") from exc


def _process_json(input_stream: TextIO, output_stream: TextIO,
                  guard: KoreanPIIGuard, style: str, strict_fields: bool) -> None:
    try:
        contents = input_stream.read().removeprefix("\ufeff")
        value = json.loads(contents, parse_constant=_reject_constant,
                           object_pairs_hook=_unique_object)
        masked = guard.mask_json(value, style=style, strict_fields=strict_fields)
        output_stream.write(json.dumps(masked, ensure_ascii=False, allow_nan=False) + "\n")
    except (ValueError, TypeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid or unmaskable JSON document") from exc


def _process_csv(input_stream: TextIO, output_stream: TextIO,
                 guard: KoreanPIIGuard, style: str, strict_fields: bool) -> None:
    # The csv module defaults to 128 KiB per field, smaller than many notes
    # columns. Use the largest supported C long on this platform.
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10
    reader = csv.reader(input_stream, strict=True)
    writer = csv.writer(output_stream)
    try:
        headers = next(reader)
    except StopIteration as exc:
        raise ValueError("CSV needs a header row") from exc
    if headers and headers[0].startswith("\ufeff"):
        headers[0] = headers[0][1:]
    if not headers or any(not header for header in headers) or len(set(headers)) != len(headers):
        raise ValueError("CSV headers must be nonempty and unique")
    masked_headers = [guard.mask(header, style=style) for header in headers]
    if len(set(masked_headers)) != len(masked_headers):
        raise ValueError("Masking would create duplicate CSV headers")
    writer.writerow(masked_headers)
    rows: list[dict[str, str]] = []
    row_characters = 0

    def flush_rows() -> None:
        if not rows:
            return
        # A list of rows shares the bounded JSON key cache while bank context
        # and explicit PII labels are still resolved for each individual row.
        masked_rows = guard.mask_json(rows, style=style, strict_fields=strict_fields)
        for masked in masked_rows:
            writer.writerow([masked[header] for header in masked_headers])
        rows.clear()

    for row in reader:
        if len(row) != len(headers):
            raise ValueError("CSV record has a different number of fields")
        characters = sum(map(len, row))
        if rows and (len(rows) >= 64 or row_characters + characters > 262_144):
            flush_rows()
            row_characters = 0
        rows.append(dict(zip(headers, row, strict=True)))
        row_characters += characters
    flush_rows()


@contextmanager
def _input(path: str):
    if path == "-":
        yield sys.stdin
    else:
        with open(path, encoding="utf-8", newline="") as stream:
            yield stream


def _output_path(input_path: str, output_path: str) -> Path:
    path = Path(output_path)
    if path.is_dir():
        raise ValueError("output path is a directory")
    if path.is_symlink():
        raise ValueError("output path must not be a symlink")
    if input_path != "-" and path.exists() and os.path.samefile(input_path, path):
        raise ValueError("input and output must be different files")
    return path


@contextmanager
def _cleanup_on_interrupt():
    """Raise a normal exception for POSIX interrupts so temp files are removed."""
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}

    def interrupted(_signum, _frame):
        raise InterruptedError("screening interrupted")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _write_masked(input_path: str, output_path: str, guard: KoreanPIIGuard,
                  style: str, format_name: str, strict_fields: bool) -> None:
    process = {"text": _process_text, "json": _process_json,
               "jsonl": _process_jsonl, "csv": _process_csv}[format_name]
    if output_path == "-":
        # Do not send partially screened data to a downstream pipe when a
        # later record fails. TemporaryFile is unlinked and mode 0600.
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", newline="") as staged:
            with _input(input_path) as source:
                process(source, staged, guard, style, strict_fields)
            staged.seek(0)
            shutil.copyfileobj(staged, sys.stdout)
        return
    with _cleanup_on_interrupt():
        path = _output_path(input_path, output_path)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as staged:
                temporary_name = staged.name
                with _input(input_path) as source:
                    process(source, staged, guard, style, strict_fields)
                staged.flush()
                os.fsync(staged.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ko-pii-guard", description="Detect and mask Korean PII in local UTF-8 data."
    )
    parser.add_argument("command", choices=("mask", "analyze", "check"))
    parser.add_argument("input", nargs="?", default="-", help="input path (default: stdin)")
    parser.add_argument("-o", "--output", default="-", help="output path (default: stdout)")
    parser.add_argument("--format", choices=("text", "json", "jsonl", "csv"), default="text")
    parser.add_argument("--style", choices=("tag", "stars", "partial"), default="tag")
    parser.add_argument("--strict-fields", action="store_true",
                        help="mask nonblank string/integer values in selected explicit PII fields")
    parser.add_argument("--entity", action="append", choices=SUPPORTED_ENTITIES,
                        help="detect only these entities; repeat for multiple types")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--ner", action="store_true",
                        help="use a previously downloaded local name/address model")
    parser.add_argument("--ner-model", choices=("e5", "kcelectra"), default="e5")
    parser.add_argument("--ner-threshold", type=float, default=0.9)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command != "mask" and args.format != "text":
        parser.error("analyze/check currently require --format text")
    if args.command != "mask" and args.output != "-":
        parser.error("analyze/check write to stdout only")
    if args.strict_fields and (args.command != "mask" or args.format == "text"):
        parser.error("--strict-fields requires mask --format json, jsonl or csv")
    if args.ner and not {"KR_NAME", "KR_ADDRESS"}.intersection(args.entity or ()):
        parser.error("--ner requires --entity KR_NAME or --entity KR_ADDRESS")
    if not args.ner and (args.ner_model != "e5" or args.ner_threshold != 0.9):
        parser.error("--ner-model and --ner-threshold require --ner")
    if not 0.0 <= args.ner_threshold <= 1.0:
        parser.error("--ner-threshold must be between 0 and 1")
    try:
        ner = None
        if args.ner:
            from ko_pii_guard.ner import KoreanNER

            ner = KoreanNER.from_pretrained(model=args.ner_model,
                                           score_threshold=args.ner_threshold)
        guard = KoreanPIIGuard(entities=args.entity, score_threshold=args.threshold,
                               ner=ner)
        if args.command == "mask":
            _write_masked(args.input, args.output, guard, args.style, args.format,
                          args.strict_fields)
            return 0
        with _input(args.input) as source:
            findings = guard.analyze(source.read())
        if args.command == "check":
            return 1 if findings else 0
        # Findings.text contains raw PII; never include it in CLI reports.
        json.dump([
            {"entity": f.entity, "start": f.start, "end": f.end, "score": f.score}
            for f in findings
        ], sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, TimeoutError,
            ImportError, RuntimeError, csv.Error):
        # Do not echo arbitrary error strings or paths: a failure may include
        # sensitive source data. Exit 2 distinguishes error from PII found.
        print("ko-pii-guard: screening failed; no complete result was produced", file=sys.stderr)
        return 2
