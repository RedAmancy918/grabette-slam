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
                self.q.put(line)
        return len(s)

    def flush(self):
        if self._buf.strip():
            self.q.put(self._buf)
        self._buf = ""


def run(source_repo, target_repo, task,
        oauth_token: gr.OAuthToken | None, progress=gr.Progress()):
    """Stream the pipeline log live.

    This is a generator: each yield updates the UI. The heavy work runs in a
    worker thread and all of its output (the log() callback + every print() in
    the pipeline) is funnelled through a queue, so the user sees progress as it
    happens instead of one dump at the end. A heartbeat keeps the UI moving
    during the long, silent download.
    """
    if oauth_token is None:
        raise gr.Error("Sign in with your Hugging Face account first.")
    if not (source_repo and target_repo and task):
        raise gr.Error("Fill in source repo, target repo and task.")

    # OAuth access tokens aren't accepted by huggingface_hub.login() (it expects a
    # classic token's role). Expose the token via env + pass it explicitly so
    # snapshot_download and push_to_hub pick it up.
    token = oauth_token.token
    os.environ["HF_TOKEN"] = token

    q: "queue.Queue[str | None]" = queue.Queue()
    result: dict = {}

    def worker():
        writer = _LineQueueWriter(q)
        try:
            # redirect_stdout funnels every print() in the pipeline into the
            # queue; log=print routes pipeline.py's own log lines through it too.
            with contextlib.redirect_stdout(writer):
                work = Path(tempfile.mkdtemp())
                try:
                    info = HfApi().repo_info(source_repo, repo_type="dataset",
                                             files_metadata=True, token=token)
                    n_files = len(info.siblings)
                    mb = sum((s.size or 0) for s in info.siblings) / 1e6
                    print(f"Downloading {source_repo} — {n_files} files, {mb:.0f} MB…")
                except Exception:
                    print(f"Downloading {source_repo}…")

                raw = snapshot_download(source_repo, repo_type="dataset",
                                        local_dir=work / "raw", token=token)
                print("Download complete.")

                n = process_dataset(raw, target_repo, task,
                                    root=work / "lerobot", log=print)
                writer.flush()
            result["n"] = n
        except Exception as e:  # surfaced to the UI after the loop
            writer.flush()
            result["error"] = e
        finally:
            q.put(None)  # sentinel: worker finished

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    logs: list[str] = []
    last_msg = "Starting…"
    start = time.time()
    frac = 0.03
    progress(frac, desc=last_msg)
    yield "Starting…", ""

    while True:
        try:
            msg = q.get(timeout=2.0)
        except queue.Empty:
            # No new line for a while (e.g. mid-download): keep the UI alive.
            progress(min(frac, 0.95), desc=f"{last_msg[:60]}  ({int(time.time()-start)}s)")
            yield "\n".join(logs), ""
            continue
        if msg is None:
            break
        logs.append(msg)
        last_msg = msg
        frac = min(0.9, frac + 0.01)
        progress(frac, desc=msg[:80])
        yield "\n".join(logs), ""

    t.join()
    if "error" in result:
        yield "\n".join(logs), f"### ❌ Failed\n```\n{result['error']}\n```"
        raise gr.Error(f"Pipeline failed: {result['error']}")

    n = result["n"]
    ds_url = f"https://huggingface.co/datasets/{target_repo}"
    viz_url = _visualizer_url(target_repo)
    logs.append(f"✅ Pushed {n} episode(s).")

    # The LeRobot visualizer sends X-Frame-Options: deny, so it can't be embedded
    # in an iframe — link out (opens in a new tab) instead.
    summary = (
        f"### ✅ Done — {n} episode(s)\n"
        f"- **Dataset:** [{target_repo}]({ds_url})\n"
        f"- **Visualize:** 👉 [open in LeRobot visualizer]({viz_url}) 👈\n\n"
        f"_(The visualizer needs the dataset to be public.)_"
    )
    progress(1.0, desc="Done")
    yield "\n".join(logs), summary


with gr.Blocks(title="Grabette SLAM → LeRobot") as demo:
    gr.Markdown(
        "# 🤖 Grabette SLAM → LeRobot\n"
        "Convert a raw OAK-D dataset from the Hub into a LeRobot dataset and push "
        "it back to the Hub. Each episode is converted, run through SLAM, assembled "
        "into LeRobot v3 format, then uploaded."
    )
    gr.LoginButton()
    source = gr.Textbox(label="Source dataset (raw repo_id)",
                        placeholder="pollen-robotics/grabette-raw")
    target = gr.Textbox(label="Target dataset (repo_id to create)",
                        placeholder="your-account/grabette-demo")
    task = gr.Textbox(label="Task description", placeholder="cup manipulation")
    btn = gr.Button("Run pipeline", variant="primary")

    log_out = gr.Textbox(label="Log", lines=20, max_lines=20, autoscroll=True)
    summary_out = gr.Markdown()

    btn.click(run, inputs=[source, target, task],
              outputs=[log_out, summary_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
