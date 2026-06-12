"""Grabette SLAM → LeRobot pipeline — HuggingFace Space (Gradio + HF OAuth).

The user signs in with their HF account; the OAuth token is used to download the
source dataset and push the generated LeRobot dataset under their account.
Gradio auto-injects the gr.OAuthToken parameter — it is NOT a UI input.
"""

import contextlib
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
from huggingface_hub import HfApi, snapshot_download

from pipeline import process_dataset

VISUALIZER = "https://huggingface.co/spaces/lerobot/visualize_dataset"

# Cooperative stop flag, shared between the run() worker and the Stop button.
# The Space processes one run at a time, so a module-level flag is enough.
_stop = threading.Event()


def _request_stop():
    """Stop button handler: signal the worker to abandon the current run.

    Gradio's cancels= stops the streaming generator immediately; this flag makes
    the background worker actually stop (skip remaining episodes, skip the push).
    In-flight blocking calls (a running download, one SLAM subprocess) finish on
    their own, but nothing further is started and nothing is pushed.
    """
    _stop.set()


def _visualizer_url(repo_id: str) -> str:
    return f"{VISUALIZER}?dataset={repo_id}&episode=0"


class _LineQueueWriter:
    """A file-like object that pushes complete lines onto a queue.

    Used to redirect stdout: every print() from convert / SLAM / build_dataset
    becomes a streamed log line, not just the explicit log() callback.
    """

    def __init__(self, q: "queue.Queue"):
        self.q = q
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.q.put(("log", line))
        return len(s)

    def flush(self):
        if self._buf.strip():
            self.q.put(("log", self._buf))
        self._buf = ""


def _bar(frac: float, label: str) -> str:
    """An HTML progress bar updated via yield (coexists with the log textbox —
    unlike gr.Progress, which overlays the output and makes the UI flip)."""
    pct = max(0, min(100, int(frac * 100)))
    return (
        '<div style="background:#e5e7eb;border-radius:8px;height:24px;width:100%;'
        'overflow:hidden;font-family:sans-serif">'
        f'<div style="background:#10b981;height:100%;width:{pct}%;transition:width .3s;'
        'color:white;font-size:13px;line-height:24px;text-align:center;white-space:nowrap">'
        f'{pct}% · {label}</div></div>'
    )


def _preflight(api, source_repo, target_repo):
    """Quick access/existence checks. Returns (exists, writable, error_or_None)."""
    try:
        api.repo_info(source_repo, repo_type="dataset")
    except Exception as e:
        return None, None, f"Cannot access source dataset '{source_repo}': {e}"

    try:
        me = api.whoami()
        username = me.get("name")
        orgs = {o.get("name") for o in me.get("orgs", [])}
    except Exception:
        username, orgs = None, set()
    ns = target_repo.split("/")[0] if "/" in target_repo else username
    writable = ns is not None and (ns == username or ns in orgs)

    exists = api.repo_exists(target_repo, repo_type="dataset")
    return exists, writable, None


def _run(source_repo, target_repo, task, oauth_token, create_pr):
    """Streaming generator behind both buttons. Yields a 4-tuple matching
    [bar (HTML), log (Textbox), summary (Markdown), confirm-PR button (update)].

    A progress bar (gr.HTML, updated via yield — no gr.Progress overlay, so no
    UI flip) sits alongside the live log. The heavy work runs in a worker thread;
    its output (log() + every print()) streams through a queue. Pre-flight checks
    run synchronously up front: source read access, target write access, and
    target existence — an existing target reveals the "Push as PR" button.
    """
    if oauth_token is None:
        raise gr.Error("Sign in with your Hugging Face account first.")
    if not (source_repo and target_repo and task):
        raise gr.Error("Fill in source repo, target repo and task.")

    # OAuth access tokens aren't accepted by huggingface_hub.login() (it expects a
    # classic token's role). Expose the token via env + pass it explicitly.
    token = oauth_token.token
    os.environ["HF_TOKEN"] = token

    logs: list[str] = []
    start = time.time()
    frac, label = 0.02, "checking"
    NOBTN = gr.update()  # leave the confirm button as-is

    def render() -> str:
        return "\n".join(logs)

    def view(summary="", btn=NOBTN):
        return _bar(frac, f"{label} · {int(time.time() - start)}s"), render(), summary, btn

    # ---- Pre-flight (synchronous, fast) ---------------------------------
    logs.append("Checking repo access…")
    yield view()
    api = HfApi(token=token)
    exists, writable, err = _preflight(api, source_repo, target_repo)
    if err:
        logs.append(f"❌ {err}")
        yield _bar(0, "error"), render(), f"### ❌ {err}", gr.update(visible=False)
        raise gr.Error(err)
    logs.append("  ✓ source readable")

    do_pr = False
    if not exists:
        if not writable:
            msg = (f"You don't have write access to namespace "
                   f"'{target_repo.split('/')[0]}' to create '{target_repo}'.")
            logs.append(f"❌ {msg}")
            yield _bar(0, "error"), render(), f"### ❌ {msg}", gr.update(visible=False)
            raise gr.Error(msg)
        logs.append(f"  ✓ target '{target_repo}' is new and writable")
    else:
        if not create_pr:
            logs.append(f"⚠️ target '{target_repo}' already exists")
            warn = (
                f"### ⚠️ Target already exists\n"
                f"`{target_repo}` already exists on the Hub. To avoid overwriting it, "
                f"click **Push as PR to existing repo** to open a Pull Request instead.\n\n"
                f"_(Change the target above and re-run if you'd rather create a new dataset.)_"
            )
            frac, label = 0.05, "target exists"
            yield _bar(frac, "target exists"), render(), warn, gr.update(visible=True)
            return
        logs.append(f"  ✓ target '{target_repo}' exists — will open a Pull Request")
        do_pr = True
    yield view()

    # ---- Heavy work in a worker thread ----------------------------------
    q: "queue.Queue[tuple]" = queue.Queue()
    result: dict = {}
    _stop.clear()  # fresh run

    def on_progress(done, total, phase):
        if phase == "slam":
            f = 0.30 + 0.55 * (done / total if total else 1.0)
            lbl = f"SLAM {done}/{total}" if total else "SLAM"
        elif phase == "build":
            f, lbl = 0.88, "building dataset"
        elif phase == "push":
            f, lbl = 0.95, ("opening PR" if do_pr else "pushing")
        else:
            f, lbl = 0.30, phase
        q.put(("progress", f, lbl))

    def worker():
        writer = _LineQueueWriter(q)
        try:
            with contextlib.redirect_stdout(writer):
                q.put(("progress", 0.07, "download"))
                work = Path(tempfile.mkdtemp())
                try:
                    info = api.repo_info(source_repo, repo_type="dataset", files_metadata=True)
                    mb = sum((s.size or 0) for s in info.siblings) / 1e6
                    print(f"Downloading {source_repo} — {len(info.siblings)} files, {mb:.0f} MB…")
                except Exception:
                    print(f"Downloading {source_repo}…")
                raw = snapshot_download(source_repo, repo_type="dataset",
                                        local_dir=work / "raw", token=token)
                print("Download complete.")
                q.put(("progress", 0.30, "converting"))
                n, pr_url = process_dataset(
                    raw, target_repo, task, root=work / "lerobot",
                    log=print, should_stop=_stop.is_set, create_pr=do_pr,
                    on_progress=on_progress, token=token)
                writer.flush()
            result["n"], result["pr_url"] = n, pr_url
        except Exception as e:
            writer.flush()
            result["error"] = e
        finally:
            q.put(("done",))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        try:
            item = q.get(timeout=1.0)
        except queue.Empty:
            yield view()  # tick the elapsed counter even when nothing new arrives
            continue
        tag = item[0]
        if tag == "done":
            break
        if tag == "progress":
            frac, label = item[1], item[2]
        else:  # ("log", line)
            logs.append(item[1])
        yield view()

    t.join()
    if "error" in result:
        err = result["error"]
        yield _bar(frac, "failed"), render(), f"### ❌ Failed\n```\n{err}\n```", gr.update(visible=False)
        raise gr.Error(f"Pipeline failed: {err}")

    n = result["n"]
    pr_url = result.get("pr_url")
    ds_url = f"https://huggingface.co/datasets/{target_repo}"
    logs.append(f"✅ Done — {n} episode(s).")
    if pr_url:
        summary = (
            f"### ✅ Done — {n} episode(s) — Pull Request opened\n"
            f"- **PR:** 👉 [{pr_url}]({pr_url}) 👈\n"
            f"- **Target:** [{target_repo}]({ds_url})\n"
        )
    else:
        viz_url = _visualizer_url(target_repo)
        # The LeRobot visualizer sends X-Frame-Options: deny, so link out.
        summary = (
            f"### ✅ Done — {n} episode(s)\n"
            f"- **Dataset:** [{target_repo}]({ds_url})\n"
            f"- **Visualize:** 👉 [open in LeRobot visualizer]({viz_url}) 👈\n\n"
            f"_(The visualizer needs the dataset to be public.)_"
        )
    frac, label = 1.0, "done"
    yield _bar(1.0, "done"), render(), summary, gr.update(visible=False)


def run_pipeline(source_repo, target_repo, task, oauth_token: gr.OAuthToken | None = None):
    yield from _run(source_repo, target_repo, task, oauth_token, create_pr=False)


def run_pipeline_pr(source_repo, target_repo, task, oauth_token: gr.OAuthToken | None = None):
    yield from _run(source_repo, target_repo, task, oauth_token, create_pr=True)


with gr.Blocks(title="Grabette SLAM → LeRobot") as demo:
    gr.Markdown(
        "# Grabette post-processing\n"
        "Convert a raw OAK-D dataset from the Hub into a LeRobot dataset and push "
        "it back to the Hub. Each episode is checked, converted, run through SLAM, assembled "
        "into LeRobot v3 format, then uploaded."
    )
    gr.LoginButton()
    source = gr.Textbox(label="Source dataset (raw repo_id)",
                        placeholder="pollen-robotics/grabette-raw")
    target = gr.Textbox(label="Target dataset (repo_id to create)",
                        placeholder="your-account/grabette-demo")
    task = gr.Textbox(label="Task description", placeholder="cup manipulation")
    with gr.Row():
        btn = gr.Button("Run pipeline", variant="primary")
        stop_btn = gr.Button("Stop", variant="stop")
    # Revealed only when the target dataset already exists (see _run).
    confirm_pr_btn = gr.Button("Push as PR to existing repo", variant="secondary",
                               visible=False)

    bar = gr.HTML()
    log_out = gr.Textbox(label="Log", lines=20, max_lines=20, autoscroll=True)
    summary_out = gr.Markdown()

    outputs = [bar, log_out, summary_out, confirm_pr_btn]
    run_event = btn.click(run_pipeline, inputs=[source, target, task], outputs=outputs)
    pr_event = confirm_pr_btn.click(run_pipeline_pr, inputs=[source, target, task],
                                    outputs=outputs)
    # Stop: cancel whichever generator is streaming (frees the UI at once) and
    # signal the worker to abandon the run (skip remaining episodes + the push).
    stop_btn.click(_request_stop, inputs=None, outputs=None,
                   cancels=[run_event, pr_event])


if __name__ == "__main__":
    # queue() is required for streaming generators and for the Stop button's
    # cancels=[] to take effect.
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
