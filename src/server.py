"""Phase 6 -- FastAPI service exposing the Phase 1-5 DSP pipeline.

Endpoints
---------
GET  /health     liveness probe
GET  /           endpoint index (JSON)
POST /process    body {"path": str, "fs": float|null, "datatype": str|null,
                       "center_hz": float|null}
                 -> full JSON report (spectral metrics, per-detection AMC,
                    constellation vectors, protocol/bitstream probe)
WS   /ws/stream  same computation, but stage progress events are streamed
                 as they happen: {"type": "progress", stage, pct, detail}
                 then a single {"type": "result"|"error", ...} terminator.

Run:  uvicorn src.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.pipeline import run_pipeline

app = FastAPI(title="NTRO RF MVP DSP service", version="0.6.0")


class ProcessRequest(BaseModel):
    """POST /process body. `fs` is mandatory for headerless raw captures."""

    path: str = Field(..., description="Capture file path (any IQ format)")
    fs: Optional[float] = Field(None, description="Sample rate [Hz]")
    datatype: Optional[str] = Field(
        None, description="cf32_le | cs16 | cu8 | ci16 | ... (auto-guessed)")
    center_hz: Optional[float] = Field(
        None, description="Disambiguate one burst in multi-burst captures")


def _resolve_path(raw: str) -> Path:
    """Validate the requested capture path (robust file-loading errors)."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path(os.environ.get("NTRO_DATA_ROOT", Path.cwd())) / p
    if not p.exists():
        raise FileNotFoundError(f"capture not found: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"not a file: {p}")
    if p.stat().st_size == 0:
        raise ValueError(f"empty capture: {p}")
    return p


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "ntro-rf-mvp"}


@app.get("/")
def index() -> Dict[str, Any]:
    return {
        "service": "NTRO RF MVP DSP",
        "endpoints": {
            "GET /health": "liveness",
            "POST /process": {"path": "str", "fs": "float|null",
                              "datatype": "str|null",
                              "center_hz": "float|null"},
            "WS /ws/stream": "streaming progress + final result",
        },
    }


@app.post("/process")
def process(req: ProcessRequest) -> Dict[str, Any]:
    """Run the full Phase 1-5 chain and return the JSON report."""
    try:
        path = _resolve_path(req.path)
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return run_pipeline(str(path), fs=req.fs, datatype=req.datatype,
                            center_hz=req.center_hz)
    except Exception as exc:  # ingestion/spectral errors -> 422 w/ reason
        raise HTTPException(
            status_code=422,
            detail=f"{type(exc).__name__}: {exc}") from exc


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    """Same pipeline with live progress: compute in a worker thread, forward
    progress events through an asyncio queue onto the socket."""
    await ws.accept()
    try:
        req = await ws.receive_json()
        raw_path = str(req.get("path", ""))
        try:
            path = str(_resolve_path(raw_path))
        except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
            await ws.send_json({"type": "error", "error": str(exc)})
            await ws.close()
            return

        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress(stage: str, pct: float, detail: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(q.put_nowait, {
                "type": "progress", "stage": stage, "pct": round(pct, 3),
                "detail": detail})

        def work() -> None:
            try:
                res = run_pipeline(path, fs=req.get("fs"),
                                   datatype=req.get("datatype"),
                                   center_hz=req.get("center_hz"),
                                   progress=progress)
                msg: Dict[str, Any] = {"type": "result", "result": res}
            except Exception as exc:
                msg = {"type": "error",
                       "error": f"{type(exc).__name__}: {exc}"}
            loop.call_soon_threadsafe(q.put_nowait, msg)   # terminator

        loop.run_in_executor(None, work)
        while True:
            msg = await q.get()
            await ws.send_json(msg)
            if msg["type"] in ("result", "error"):
                break
        await ws.close()
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
