#!/usr/bin/env python3
"""
Continuously sends a prompt to the `claude` CLI, stores the streamed JSON
responses in /dev/shm/output.json, and respects any reported rate-limit reset
time by sleeping until the limit expires. The script mirrors the tool’s output
both to stdout and to the JSON log file so existing log-processing workflows
continue to work.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

OUTPUT_PATH = Path("/dev/shm/output.json")

# Exact prompt text from the original loop3.source script
PROMPT = (
    "Read @PRD.md , @README.md first, git then git log -10.\n\n"
    "You are an senior software engineer who likes correctness and Nix.\n"
    "Run nix-shell --run 'just build test' to check what works and what needs to be fixed.\n\n"
    "Run tests often, don't assume your code is working.\n\n"
    "Fix root causes. Dont make assumptions, be general. Be correct and robust.\n"
    "NEVER make workarounds.\n"
    "NEVER use emojis.\n"
    "Make conventional git commits between code changes.\n"
    "Iterate until 'just build test' succeeds without failure.\n"
    "Be self-critical, terse, clear, and concise.\n"
    "Do not overcomplicate, reason from first principles.\n"
    "Add traces to code for debugging, under debug flags.\n"
    "Run commands with timeout of 10 minutes.\n\n"
    "You shall read @todo.md and fix the issues mentioned there.\n"
    "Once done, prefix with DONE.\n"
    "Once all is done, look for code smells and bugs, add missing tests -- add them to todo.md.\n"
)

# Patterns reproducing the original bash greps
RATE_LIMIT_MSG = re.compile(r"Claude.*(?:usage|use|limit).*reach", re.IGNORECASE)
FIRST_INT = re.compile(r"(\d+)")


def last_json_line(path: Path) -> str:
    """Return the last non-empty line from *path* (as text)."""
    if not path.exists():
        return ""
    # Read file from the end without loading entire file into memory
    with path.open("rb") as fp:
        fp.seek(0, os.SEEK_END)
        pos = fp.tell() - 1
        buf = bytearray()
        while pos >= 0:
            fp.seek(pos)
            char = fp.read(1)
            if char == b"\n":
                # If this is the very first char we seek at, skip it
                if pos == fp.tell() - 1 and not buf:
                    pos -= 1
                    continue
                if buf:
                    break  # We have the last line
            else:
                buf.extend(char)
            pos -= 1
        buf.reverse()
        return buf.decode()


def rate_limit_reset_epoch(raw_json: str) -> Optional[int]:
    """Return epoch reset time if *raw_json* signals a rate-limit, else None."""
    if not raw_json:
        return None

    try:
        # Parse the JSON to extract the result field
        data = json.loads(raw_json.strip())
        if data.get("is_error") and "result" in data:
            result = data["result"]
            # Check if it's a rate limit message using the regex
            if RATE_LIMIT_MSG.search(result) and "|" in result:
                # Extract the timestamp after the pipe character
                timestamp_str = result.split("|", 1)[1]
                return int(timestamp_str)
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr, flush=True)

    return None


def claude_cmd(continue_flag: bool) -> list[str]:
    """Build the command list for subprocess based on *continue_flag*."""
    cmd = ["claude"]
    if continue_flag:
        cmd.append("--continue")
    cmd.extend(
        [
            "--dangerously-skip-permissions",
            "--verbose",
            "--output-format",
            "stream-json",
            "-p",
            PROMPT,
        ]
    )
    return cmd


def main() -> None:
    continue_next = False

    while True:
        cmd = claude_cmd(continue_next)
        print("Running:", " ".join(cmd), file=sys.stderr, flush=True)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output to both stdout and the log file (like `tee -a`)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("a", buffering=1) as log_fp:
            for line in process.stdout:  # type: ignore[attr-defined]
                print(line, end="", flush=True)
                log_fp.write(line)

        process.wait()

        continue_next = False  # Reset; will be re-enabled if rate-limited

        last_line = last_json_line(OUTPUT_PATH)
        reset_epoch = rate_limit_reset_epoch(last_line)
        if reset_epoch is not None:
            continue_next = True
            current_epoch = int(time.time())
            print(f"Current epoch: {current_epoch}, reset epoch: {reset_epoch}", file=sys.stderr, flush=True)
            sleep_seconds = max(0, reset_epoch - current_epoch)
            reset_time = datetime.fromtimestamp(reset_epoch).isoformat(timespec="seconds")
            print(
                f"Sleeping for {sleep_seconds} seconds (until {reset_time}) to wait for rate limit reset",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_seconds)

        # Always pause briefly to avoid spamming requests
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
