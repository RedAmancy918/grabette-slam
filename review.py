"""Episode review — the per-session state accessors + the reactive @gr.render panel
that lets the user drop flagged trajectories before the dataset is built.

When SLAM flags ≥1 non-GOOD episode, the run controller (worker thread) publishes
them on ITS OWN session (see session.RunSession) and blocks on `review_done` while
`review` is set. A gr.Timer mirrors that session's flags into the panel's gr.State
(poll), and @gr.render draws one card per still-kept episode.

Every handler below resolves the caller's session from the injected `gr.Request`,
so a flagged episode is only ever shown to the user whose run produced it — nothing
here is module-global, precisely so nothing leaks between concurrent users.
"""

import gradio as gr

import session
from views import episode_card


def drop_episode(name: str, request: gr.Request = None) -> dict:
    """🗑 clicked on a flagged episode: add it to this session's drop set and return
    the fresh state so the @gr.render panel re-runs and the card disappears."""
    sess = session.get(request)
    if name not in sess.review_drop:
        sess.review_drop.append(name)
    return sess.review_state()


def restore_episodes(request: gr.Request = None) -> dict:
    """'Restore' clicked: clear the drop set (un-hide every flagged card)."""
    sess = session.get(request)
    sess.review_drop.clear()
    return sess.review_state()


def submit_review(request: gr.Request = None) -> None:
    """'Continue' clicked: release this session's worker. The episodes to drop already
    live in review_drop (set as the user clicked 🗑). outputs=None — like Pause/Cancel,
    this only flips a flag; the worker clears `review` and the panel closes on the
    next poll."""
    session.get(request).review_done.set()


def poll(cur: dict, request: gr.Request = None):
    """gr.Timer tick: mirror this session's worker flags into its panel State so it
    opens/closes and its drop set stays in sync. Returns gr.skip() when nothing
    changed, so an idle session never churns the @gr.render block."""
    nxt = session.get(request).review_state()
    if nxt == cur:
        return gr.skip()
    return nxt


def build_panel(review_state) -> None:
    """Register the reactive review panel on the *current* Blocks. Call this inside
    `with gr.Blocks():` — @gr.render binds to the active Blocks. `review_state` is the
    gr.State the Timer feeds (see poll)."""
    @gr.render(inputs=review_state)
    def _render(st):
        if not st.get("open"):
            return
        items = st["items"]
        dropped = set(st["dropped"])
        kept = [it for it in items if it["name"] not in dropped]
        # Pre-SLAM gate (kind "input") flags incomplete/desynced recordings before
        # the slow SLAM; post-SLAM gate (kind "trajectory") flags bad odometry.
        if st.get("kind") == "input":
            source, action = "dataset / sync check", "run SLAM on"
        else:
            source, action = "trajectory check", "build & push"
        gr.Markdown(
            f"### ⚠️ {len(items)} episode(s) flagged by the {source}\n"
            f"Drop those to exclude with 🗑, then **Continue**. "
            f"The other episodes are kept. "
            f"**{len(kept)}** flagged and kept."
        )
        # The card list scrolls on its own (#review-cards) and the action row is
        # sticky (#review-actions), so Continue stays reachable however many
        # episodes are flagged — the run is BLOCKED on that click, and the page
        # itself can't be scrolled to chase it inside the Space's iframe (see the
        # scrolling note in views.CSS).
        with gr.Column(elem_id="review-cards"):
            for it in items:
                if it["name"] in dropped:
                    continue  # removed → its card has disappeared
                with gr.Row(equal_height=True):
                    gr.HTML(episode_card(it))
                    gr.Button("🗑", size="sm", scale=0, min_width=52,
                              elem_classes=["btn-red"]).click(
                        drop_episode, inputs=gr.State(it["name"]), outputs=review_state)
        with gr.Row(elem_id="review-actions"):
            gr.Button(f"▶  Continue ({action} kept episodes)",
                      size="sm", elem_classes=["btn-orange"]).click(
                submit_review, inputs=None, outputs=None)
            if dropped:
                gr.Button(f"↺  Restore {len(dropped)} removed",
                          variant="secondary", size="sm").click(
                    restore_episodes, inputs=None, outputs=review_state)
