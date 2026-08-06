"""Grabette SLAM → LeRobot pipeline — HuggingFace Space (Gradio + HF OAuth).

The user signs in with their HF account; the OAuth token is used to download the
source dataset and push the generated LeRobot dataset under their account.
Gradio auto-injects the gr.OAuthToken parameter — it is NOT a UI input.

This module is just the UI: it builds the Blocks layout and wires events. The
rendering helpers live in views.py, the streaming run generators + cooperative
flags in controller.py, and the episode-review state + reactive panel in review.py.

REST API (called by grabette devices):
  POST /api/process  {source_repo, target_repo, task}  Authorization: Bearer <hf_token>  → {job_id}
  GET  /api/status/{job_id}                                      → {status, log, error}
"""

import contextlib
import os
import tempfile
import threading
import uuid
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, Header, HTTPException
from huggingface_hub import HfApi, snapshot_download
from pydantic import BaseModel

import review
from views import CSS, THEME, bar
from controller import (
    confirm_cancel,
    keep_running,
    request_cancel,
    reset,
    retry_push,
    run_pipeline,
    run_pipeline_branch,
    toggle_pause,
)

# ── Standalone FastAPI app — Gradio is mounted onto it so custom routes survive.
# Registering routes on demo.app before launch() is unreliable in Gradio 6: the
# internal FastAPI instance may be replaced, dropping the handlers.
fastapi_app = FastAPI()

with gr.Blocks(title="Grabette SLAM → LeRobot") as demo:
    gr.HTML(
        "<div id='app-header'><h1>Grabette post-processing</h1>"
        "<p>Raw Grabette recording → SLAM → LeRobot dataset on the Hub</p></div>"
    )
    with gr.Row():
        # ---- Inputs (left) ----
        with gr.Column(scale=2, min_width=280):
            gr.LoginButton(size="sm")
            source = gr.Textbox(label="Source dataset", info="raw Grabette repo on the Hub",
                                placeholder="pollen-robotics/grabette-raw")
            target = gr.Textbox(label="Target dataset", info="repo to create or contribute to",
                                placeholder="your-account/grabette-dataset")
            task = gr.Textbox(label="Task description", placeholder="cup grasping")
            # Sync checks (OAK cam↔gyro pre-SLAM, image↔trajectory post-SLAM) are a
            # debug-only diagnostic — off by default so a normal run skips them.
            debug_sync = gr.Checkbox(
                label="Run sync checks (debug)", value=False,
                info="Extra camera↔IMU / image↔trajectory timing diagnostics")
            # Read-only recap of the inputs, shown in their place while running.
            selection_card = gr.HTML(visible=False)
            # Idle: only Run. Running: Run greyed to 'Running', + Pause + Cancel.
            # Finished: only Reset. All visibility is driven by the generators.
            with gr.Row():
                run_btn = gr.Button("▶  Run", variant="primary", scale=2)
                pause_btn = gr.Button("⏸  Pause", variant="secondary",
                                      scale=1, visible=False)
                cancel_btn = gr.Button("✕  Cancel", variant="stop",
                                       scale=1, visible=False)
            # Cancel confirmation (revealed by Cancel; replaces Pause/Cancel).
            with gr.Row():
                confirm_cancel_btn = gr.Button("✓  Confirm cancel", variant="stop",
                                               size="sm", visible=False)
                keep_btn = gr.Button("Keep running", variant="secondary",
                                     size="sm", visible=False)
            # Revealed (as the primary action, replacing Run) only when the target
            # dataset already exists (see the controller).
            confirm_branch_btn = gr.Button("⎇  Push to a new branch",
                                           variant="primary", visible=False)
            # Revealed only when a build succeeded but the push failed.
            retry_btn = gr.Button("⟳  Retry push", variant="secondary",
                                  size="sm", visible=False)
            # Shown once a run reaches a terminal state.
            reset_btn = gr.Button("↺  Reset", variant="secondary", visible=False)
            # Holds the built-but-unpushed dataset context for retry_push.
            retry_state = gr.State(None)
        # ---- Outputs (right) ----
        with gr.Column(scale=3, min_width=280):
            # Bar + log are revealed only while a run is in progress (running /
            # paused); the generators toggle their visibility via views.io().
            bar_out = gr.HTML(bar(0, "ready"), visible=False)
            log_out = gr.Textbox(label="Log", lines=15, max_lines=15, autoscroll=True,
                                 elem_id="logbox", visible=False)

            # Episode review (reactive): a gr.Timer mirrors the worker's review
            # flags into review_state, and review.build_panel draws one orange card
            # per still-kept flagged episode. Not part of the streaming outputs.
            review_state = gr.State({"open": False, "kind": "trajectory",
                                     "items": [], "dropped": []})
            review_timer = gr.Timer(0.5)
            review.build_panel(review_state)

            # line_breaks=True so a single "\n" in a summary/warning renders as a
            # line break (GFM soft breaks) — plain Markdown would collapse it to a
            # space and only "\n\n" (a full paragraph gap) would show.
            summary_out = gr.Markdown(line_breaks=True)

    # Order MUST match the generators' yields (views.io → bar, log; then summary,
    # retry_state; then views.btns → branch, retry, run, pause, cancel,
    # confirm_cancel, keep, reset; then views.inputs_view → source, target, task,
    # selection_card). The review panel is reactive, NOT in this list.
    outputs = [bar_out, log_out, summary_out, retry_state,
               confirm_branch_btn, retry_btn,
               run_btn, pause_btn, cancel_btn, confirm_cancel_btn, keep_btn, reset_btn,
               source, target, task, selection_card]
    run_event = run_btn.click(run_pipeline, inputs=[source, target, task, debug_sync],
                              outputs=outputs)
    branch_event = confirm_branch_btn.click(run_pipeline_branch,
                                            inputs=[source, target, task, debug_sync],
                                            outputs=outputs)
    retry_event = retry_btn.click(retry_push, inputs=[retry_state], outputs=outputs)
    reset_btn.click(reset, inputs=None, outputs=outputs)
    # Pause/Resume + cancel-confirm just flip cooperative flags; the running
    # generator re-renders the buttons from them on its next tick (outputs=None
    # → these never write the same components as the stream, so no conflict).
    pause_btn.click(toggle_pause, inputs=None, outputs=None)
    cancel_btn.click(request_cancel, inputs=None, outputs=None)
    keep_btn.click(keep_running, inputs=None, outputs=None)
    # Mirror the worker's review flags into review_state every tick; review.poll
    # returns gr.skip() when nothing changed, so the @gr.render block only redraws
    # on a real open/close or drop-set change (no churn while idle or running).
    review_timer.tick(review.poll, inputs=review_state, outputs=review_state)
    # Confirm cancel: first cancel the streaming generator + signal the worker to
    # abandon (outputs=None so it preempts at once), THEN reset the page to idle.
    # Split into click(cancel) → .then(reset) because a single function can't both
    # cancel a generator and reliably write that generator's outputs.
    cancel_evt = confirm_cancel_btn.click(
        confirm_cancel, inputs=None, outputs=None,
        cancels=[run_event, branch_event, retry_event])
    cancel_evt.then(reset, inputs=None, outputs=outputs)


# ── REST API for grabette devices ────────────────────────────────────────────
# Headless pipeline calls: grabette uploads a raw dataset then POSTs here.
# No review callbacks — unattended run, bad episodes are skipped automatically.

_rest_jobs: dict[str, dict] = {}  # job_id → {status, log, error}


class _ProcessReq(BaseModel):
    source_repo: str
    target_repo: str
    task: str
    exclude_fail: bool = False
    exclude_bad: bool = False
    private: bool = False
    run_sync: bool = False
    # Device roles present in the raw dataset (e.g. ["left","right"]). >1 role
    # → bimanual build (each episode combines the arms, uploaded by the fleet to
    # {episode_id}/{role}/); ≤1 role (or empty) → the classic single-arm build.
    roles: list[str] = []


def _run_pipeline_headless(job_id: str, source_repo: str, target_repo: str,
                            task: str, token: str,
                            exclude_fail: bool = False,
                            exclude_bad: bool = False,
                            private: bool = False,
                            run_sync: bool = False,
                            roles: list[str] | None = None) -> None:
    """Run the pipeline synchronously in a background thread.

    exclude_fail: drop episodes with trajectory verdict FAIL.
    exclude_bad:  drop all BAD episodes.
    private:      create the LeRobot dataset repo as private.
    run_sync:     enable the debug-only sync diagnostics (off by default).
    All episodes appear in the returned quality data (including excluded ones)
    so the caller can display a full recap.
    """
    logs: list[str] = []
    quality_episodes: list[dict] = []

    def log(msg: str) -> None:
        logs.append(msg)
        _rest_jobs[job_id]["log"] = "\n".join(logs)

    result_link: str | None = None
    try:
        # No global os.environ["HF_TOKEN"] here — every HF call below passes the
        # token explicitly (HfApi / snapshot_download / delete_repo), and the push
        # scopes it under a lock itself (see push_lerobot). Setting it globally
        # would let concurrent conversions clobber each other's token.
        api = HfApi(token=token)
        log(f"Downloading {source_repo}…")
        work = Path(tempfile.mkdtemp())
        raw = snapshot_download(source_repo, repo_type="dataset",
                                local_dir=work / "raw", token=token,
                                max_workers=48)
        log("Download complete. Running SLAM + LeRobot conversion…")
        ds_root = work / "lerobot"

        with contextlib.redirect_stdout(type("W", (), {"write": lambda s, m: log(m.rstrip()) or None, "flush": lambda s: None})()): # noqa: E501
            from pipeline import (build_recordings, episode_label,
                                  find_episode_dirs, precheck_episodes, run_slam,
                                  push_lerobot)

            episodes = find_episode_dirs(Path(raw))
            if not episodes:
                raise ValueError(f"No episodes found under {raw}")
            log(f"Found {len(episodes)} episode(s).")

            # Pre-check: record quality data for all episodes; exclude those with
            # recording errors unconditionally, and optionally exclude those with
            # recording warnings or poor sync depending on caller flags.
            checks = precheck_episodes(episodes, log=log, run_sync=run_sync)
            for c in checks:
                sync = c.get("sync") or {}
                sync_verdict = sync.get("verdict", "GOOD")
                sync_msg = sync.get("message", "")
                sync_flagged = sync_verdict != "GOOD"
                if c["errors"]:
                    quality_episodes.append({
                        "name": c["name"], "kind": "pre_check", "verdict": "ERROR",
                        "errors": list(c["errors"]),
                        "warnings": list(c["warnings"]) + ([sync_msg] if sync_flagged and sync_msg else []),
                        "excluded": True,
                    })
                elif c["warnings"] or sync_flagged:
                    quality_episodes.append({
                        "name": c["name"], "kind": "pre_check", "verdict": "WARN",
                        "errors": [],
                        "warnings": list(c["warnings"]) + ([sync_msg] if sync_flagged and sync_msg else []),
                        "excluded": False,
                    })
            slam_input = [c["ep"] for c in checks if not c["errors"]]
            if not slam_input:
                log("⚠ All episodes failed the recording check — nothing to process.")
            else:
                # SLAM: convert + trajectory per episode.
                slam_results = run_slam(slam_input, log=log, run_sync=run_sync)
                slam_succeeded = {ep for ep, _ in slam_results}

                # Record SLAM failures (no trajectory produced at all).
                for ep in slam_input:
                    if ep not in slam_succeeded:
                        quality_episodes.append({
                            "name": episode_label(ep), "kind": "slam", "verdict": "FAIL",
                            "tracking_pct": 0.0, "n_jumps": 0, "total_distance_m": 0.0,
                            "errors": ["SLAM failed: no trajectory produced"],
                            "warnings": [], "excluded": True,
                        })

                # Apply verdict filter and record trajectory quality.
                kept = []
                for ep, report in slam_results:
                    excluded = (
                        (exclude_fail and report.verdict == "FAIL")
                        or (exclude_bad and report.verdict == "BAD")
                    )
                    if excluded:
                        log(f"  • Excluded ({report.verdict}): {episode_label(ep)}")
                    else:
                        kept.append(ep)
                    quality_episodes.append({
                        "name": episode_label(ep), "kind": "trajectory",
                        "verdict": report.verdict,
                        "tracking_pct": round(report.tracking_pct, 1),
                        "n_jumps": report.n_jumps,
                        "total_distance_m": round(report.total_distance_m, 2),
                        "errors": list(report.errors),
                        "warnings": list(report.warnings),
                        "excluded": excluded,
                    })

                if not kept:
                    log("⚠ No episode produced an acceptable trajectory — dataset not pushed.")
                else:
                    # Same builder as the UI run: `roles` (given by the fleet, raw
                    # uploaded to {episode_id}/{role}) selects the uniform
                    # role-prefixed schema for any arm count; when absent it is
                    # inferred from the layout, so a legacy flat upload keeps the
                    # original single-arm schema.
                    report_by_ep = {ep: rep for ep, rep in slam_results}
                    built = build_recordings(target_repo, task, ds_root, kept,
                                             report_by_ep, roles=roles, log=log)
                    n_built = len(built)
                    n, link, _mode = push_lerobot(target_repo, task, ds_root,
                                                  n_built, token=token, log=log,
                                                  private=private)
                    result_link = link or f"https://huggingface.co/datasets/{target_repo}"
                    log(f"✅ Done — {n} episode(s).")

        _rest_jobs[job_id].update(
            status="done",
            result=result_link,
            quality=quality_episodes,
        )
    except Exception as e:
        _rest_jobs[job_id].update(status="error", error=str(e), quality=quality_episodes)
    finally:
        # Clean up the raw source ONLY on success. On failure keep it, so the
        # run can be inspected and retried WITHOUT re-uploading from the devices
        # (the upload is the slow part). `api` may be unset if we failed before
        # it was built — the success gate avoids touching it in that case.
        if _rest_jobs[job_id].get("status") == "done":
            try:
                api.delete_repo(source_repo, repo_type="dataset")
                log(f"Deleted raw dataset {source_repo}.")
            except Exception as e:
                log(f"Warning: could not delete raw dataset: {e}")
        else:
            log(f"Kept raw dataset {source_repo} (run did not complete) — inspect or retry.")


@fastapi_app.post("/api/process")
async def api_process(req: _ProcessReq, authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <hf_token>")
    token = authorization[len("Bearer "):].strip()
    job_id = uuid.uuid4().hex[:12]
    _rest_jobs[job_id] = {"status": "running", "log": "", "error": None, "result": None, "quality": None}
    t = threading.Thread(
        target=_run_pipeline_headless,
        args=(job_id, req.source_repo.strip(), req.target_repo.strip(),
              req.task.strip(), token),
        kwargs={"exclude_fail": req.exclude_fail, "exclude_bad": req.exclude_bad,
                "private": req.private, "run_sync": req.run_sync, "roles": req.roles},
        daemon=True,
    )
    t.start()
    return {"job_id": job_id}


@fastapi_app.get("/api/status/{job_id}")
async def api_status(job_id: str):
    job = _rest_jobs.get(job_id)
    if job is None:
        return {"status": "not_found"}
    return job


# Mount Gradio onto the FastAPI app. queue() is required for streaming generators
# and for the Stop button's cancels=[] to take effect.
app = gr.mount_gradio_app(fastapi_app, demo.queue(), path="/", theme=THEME, css=CSS)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
