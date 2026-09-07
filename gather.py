"""Community contributions → one raw dataset. Step 1 of building a community
LeRobot dataset; the conversion form below it is step 2.

A community dataset holds no recordings. Each accepted episode is one small JSON
under ``contrib/<contributor>/`` pointing at a recording that lives in the
CONTRIBUTOR's own raw dataset. Copying those into a single raw dataset is what
turns many contributors' uploads into something the converter can read — and it
belongs HERE rather than on a maintainer's laptop, because both ends are on the
Hub: run from home, every episode would come down a domestic link and go straight
back up again.

Deliberately NOT folded into the run generator in controller.py. This has no SLAM,
no episode review, no pause: it is a file copy, and giving it its own small panel
keeps both flows readable. It does reuse controller._preflight, which already
answers exactly the two questions asked here (can I read the source, can I create
or write the target).
"""

import queue
import threading

import gradio as gr
from huggingface_hub import HfApi

from controller import _preflight
from grabette_postprocess.community import gather as gather_contributions

# Whether the Gather button is usable, and what it says while it is not.
_IDLE = "⇩  Gather into a raw dataset"
_BUSY = "Gathering…"


def _view(log_lines, summary="", *, busy, source=None, chain=False):
    """The panel's five outputs: log, summary, button, the conversion form's
    source field (on success), and whether the chained conversion should run.

    `chain` defaults to False on EVERY yield, so a refusal, a dry run or a
    failure can never leave a stale True behind for the .then() to act on.
    """
    return (
        gr.update(value="\n".join(log_lines), visible=bool(log_lines)),
        summary,
        gr.update(value=_BUSY if busy else _IDLE, interactive=not busy),
        gr.skip() if source is None else gr.update(value=source),
        chain,
    )


def run_gather(community_repo, raw_repo, roles, dry_run, chain=False,
               conv_target="", conv_task="",
               oauth_token: gr.OAuthToken | None = None):
    """Streaming generator behind the Gather button.

    Yields (log, summary, button, source, chain). `source` is the point of doing
    this in the Space at all: on success the conversion form's Source dataset is
    filled in with the raw dataset just written, so step 2 is one click away.
    `chain` tells the conversion chained after this one whether to run.

    conv_target / conv_task are the conversion form's fields, read here ONLY to
    validate them: "and convert" with an empty target would copy every episode
    and then fail on a missing field, which is the worst possible ordering.
    """
    if oauth_token is None:
        raise gr.Error("Sign in with your Hugging Face account first.")
    community_repo = (community_repo or "").strip()
    raw_repo = (raw_repo or "").strip()
    if not (community_repo and raw_repo):
        raise gr.Error("Fill in the community dataset and the raw dataset to write.")
    if community_repo == raw_repo:
        # Gathering a repo into itself would write episode folders into the
        # community dataset, which is precisely what this design keeps out of it.
        raise gr.Error("The raw dataset must be a different repo from the community one.")
    chain = bool(chain) and not dry_run  # a dry run copies nothing to convert
    if chain and not ((conv_target or "").strip() and (conv_task or "").strip()):
        raise gr.Error("To gather and convert in one go, fill in Target dataset and "
                       "Task description in the form below first.")
    token = oauth_token.token  # local only — never os.environ, see controller._run
    scopes = frozenset((getattr(oauth_token, "scope", "") or "").split())

    logs = ["Checking repo access…"]
    yield _view(logs, busy=True)

    api = HfApi(token=token)
    # The target may well already exist here (a second gathering adds to it), so
    # only the error matters — unlike a conversion, where an existing target means
    # something.
    _exists, _writable, err = _preflight(api, community_repo, raw_repo, scopes)
    if err:
        logs.append(err)
        yield _view(logs, summary=f"### ⚠️ {err.splitlines()[0]}", busy=False)
        return
    logs.append("  ✓ community dataset readable")
    logs.append(f"  ✓ write access to '{raw_repo.split('/')[0]}'")
    yield _view(logs, busy=True)

    q: "queue.Queue[str]" = queue.Queue()
    result: dict = {}

    def work():
        try:
            result["plan"] = gather_contributions(
                community_repo, raw_repo, token=token,
                require_roles=list(roles or ()) or None,
                dry_run=bool(dry_run), log=q.put)
        except Exception as e:  # noqa: BLE001 — surfaced in the panel, never a traceback
            result["error"] = str(e)
        finally:
            q.put(None)  # sentinel: the worker is done

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while True:
        line = q.get()
        if line is None:
            break
        logs.append(line)
        yield _view(logs, busy=True)

    if "error" in result:
        logs.append(f"✗ {result['error']}")
        yield _view(logs, summary=f"### ✗ Gathering failed\n{result['error']}", busy=False)
        return

    plan = result["plan"]
    n, left_out = len(plan.gather), len(plan.skipped)
    url = f"https://huggingface.co/datasets/{raw_repo}"
    if dry_run:
        summary = (f"### Dry run — nothing copied\n{n} episode(s) would be gathered into "
                   f"`{raw_repo}`" + (f", {left_out} left out (see the log)." if left_out else "."))
        yield _view(logs, summary=summary, busy=False)
        return
    if not n:
        # Not a failure: everything merged so far is already there. Say which,
        # rather than leaving an empty panel to be read as a broken run.
        summary = (f"### Nothing new to gather\n"
                   f"Every contribution merged into `{community_repo}` is already in "
                   f"[`{raw_repo}`]({url})."
                   + (f" {left_out} left out (see the log)." if left_out else "")
                   + f"\n\n**Source dataset** in the form below is set to it: **Run** "
                     f"there to build the LeRobot dataset.")
        # Nothing new is not a failure, and there IS something to convert, so a
        # chained run still goes ahead on what is already gathered.
        yield _view(logs, summary=summary, busy=False, source=raw_repo, chain=chain)
        return
    summary = (
        f"### ✅ {n} episode(s) gathered\n"
        f"- **Raw dataset:** 👉 [`{raw_repo}`]({url}) 👈\n"
        + (f"- {left_out} contribution(s) left out — see the log for each reason.\n"
           if left_out else "")
        + (f"\n**Converting it now** into `{conv_target.strip()}` — the run panel on "
           f"the right takes over from here."
           if chain else
           f"\n**This is a raw dataset, not a LeRobot one.** **Source dataset** in "
           f"the form below is now set to `{raw_repo}`: fill in a target and a task "
           f"description, then **Run** to build the LeRobot dataset from it.")
    )
    yield _view(logs, summary=summary, busy=False, source=raw_repo, chain=chain)
