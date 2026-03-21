# SPDX-License-Identifier: Apache-2.0
"""Cursor debug NDJSON (session 7ce62a). Remove after investigation."""

from __future__ import annotations

import json
import os
import time
from typing import Any

_SESSION = "7ce62a"
_LOG_PATH = "/home/prushto/.cursor/debug-7ce62a.log"


def agent_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> None:
    payload = {
        "sessionId": _SESSION,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": {**(data or {}), "pid": os.getpid()},
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, default=str)
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        print(f"AGENT_DEBUG_NDJSON {line}", flush=True)
