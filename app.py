"""Grabette SLAM pipeline — HuggingFace Space (Gradio).

Test mode: upload a .zip of the dataset, run convert + SLAM on every episode,
and visualize the resulting trajectories. No LeRobot build, no Hub push yet.
"""

import tempfile
import zipfile
from pathlib import Path

import gradio as gr

from pipeline import run_slam_pipeline, plot_trajectory


def run(zip_path, progress=gr.Progress()):
    if not zip_path:
        raise gr.Error("Upload a .zip of the dataset first.")

    work = Path(tempfile.mkdtemp())
    dataset_dir = work / "dataset"
    progress(0.1, desc="Extracting zip…")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dataset_dir)

    logs: list[str] = []

    def log(msg: str):
        logs.append(msg)
        progress(0.5, desc=msg)  # keep the UI progress bar alive

    results = run_slam_pipeline(dataset_dir, log=log)

    gallery = [
        (plot_trajectory(r["traj_path"], f'{r["name"]} — {r["tracking_pct"]:.0f}% tracked'),
         r["name"])
        for r in results
    ]
    return "\n".join(logs), gallery


with gr.Blocks(title="Grabette SLAM Pipeline") as demo:
    gr.Markdown(
        "# 🤖 Grabette SLAM (test mode)\n"
        "Upload a `.zip` of a raw OAK-D dataset. Each episode is converted and run "
        "through SLAM; the trajectories are plotted below (gravity-aligned, Z-up)."
    )
    zip_in = gr.File(label="Dataset (.zip)", file_types=[".zip"], type="filepath")
    btn = gr.Button("Run convert + SLAM", variant="primary")
    log_out = gr.Textbox(label="Log", lines=15)
    gallery = gr.Gallery(label="Trajectories", columns=2, height="auto")

    btn.click(run, inputs=[zip_in], outputs=[log_out, gallery])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
