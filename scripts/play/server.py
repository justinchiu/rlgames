"""Local web page to watch training runs and play against their latest checkpoint.

    uv run python scripts/play/server.py --exps_dir exps --port 8000

Then open http://localhost:8000. Every subdirectory of `exps_dir` with a cfg.json
(written by scripts/train/selfplay_main.py) shows up as a run. Each new game
loads the run's latest checkpoint, so you can play while it is still training.
"""

import argparse
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import jax
import jax.numpy as jnp
import numpy as np

from rlgames import utils
from rlgames.games import make
from rlgames.networks.mlp import apply_mlp
from rlgames.play.views import make_view, who
from rlgames.utils.constants import CHANCE

STATIC = os.path.join(os.path.dirname(__file__), "..", "..", "rlgames", "play", "static")


class GameRunner:
    """Jitted functions for one game, shared by all sessions of that game."""

    def __init__(self, game):
        self.game = game
        self.view = make_view(game)
        self.step = jax.jit(game.step)
        self.observe = jax.jit(game.observe)
        self.policy = jax.jit(lambda params, obs: apply_mlp(params, obs)[0])


class Session:
    def __init__(self, runner: GameRunner, ckpt: dict, seat: int, greedy: bool, seed: int):
        self.runner = runner
        self.params = jax.tree.map(jnp.asarray, ckpt["params"])
        self.ckpt_iter = ckpt["iter"]
        self.seat = seat
        self.greedy = greedy
        self.key = jax.random.key(seed)
        self.state = runner.game.init()
        self.history: list[str] = []
        self.advance()

    def np_state(self):
        return jax.tree.map(np.asarray, self.state)

    def apply(self, action: int, note: str = ""):
        event = self.runner.view.event(self.np_state(), action, self.seat)
        if event is not None:
            self.history.append(event + note)
        self.state = self.runner.step(self.state, jnp.int32(action))

    def advance(self):
        """Play chance and AI moves until it is the human's turn or the game ends."""
        while not bool(self.state.terminal) and int(self.state.player) != self.seat:
            self.key, sub = jax.random.split(self.key)
            if int(self.state.player) == CHANCE:
                self.apply(int(jax.random.categorical(sub, jnp.log(self.state.chance))))
                continue
            obs = self.runner.observe(self.state, self.state.player)
            logits = self.runner.policy(self.params, obs)
            logits = jnp.where(self.state.legal, logits, -jnp.inf)
            probs = jax.nn.softmax(logits)
            if self.greedy:
                action = int(jnp.argmax(probs))
            else:
                action = int(jax.random.categorical(sub, logits))
            self.apply(action, note=f"  (p={float(probs[action]):.2f})")

    def act(self, action: int):
        if bool(self.state.terminal) or int(self.state.player) != self.seat:
            raise ValueError("not your turn")
        if not bool(self.state.legal[action]):
            raise ValueError("illegal action")
        self.apply(action)
        self.advance()

    def to_json(self, session_id: str) -> dict:
        view = self.runner.view
        state = self.np_state()
        terminal = bool(state.terminal)
        out = {
            "session": session_id,
            "ckpt_iter": self.ckpt_iter,
            "seat": self.seat,
            "history": self.history,
            "info": view.info(state, self.seat, reveal=terminal),
            "grid": view.grid(state),
            "terminal": terminal,
            "actions": [],
        }
        if terminal:
            ours = float(state.returns[self.seat]) + 0.0  # avoid printing -0
            verdict = "You win" if ours > 0 else "You lose" if ours < 0 else "Draw"
            out["result"] = f"{verdict} ({ours:+g})"
            out["return"] = ours
        else:
            labels = view.labels(state)
            out["actions"] = [
                {"id": a, "label": labels[a]} for a in np.nonzero(state.legal)[0].tolist()
            ]
            out["to_move"] = who(int(state.player), self.seat)
        return out


class App:
    def __init__(self, exps_dir: str):
        self.exps_dir = exps_dir
        self.runners: dict[str, GameRunner] = {}
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()

    def run_dir(self, run: str) -> str:
        path = os.path.realpath(os.path.join(self.exps_dir, run))
        if not path.startswith(os.path.realpath(self.exps_dir) + os.sep):
            raise ValueError("bad run name")
        return path

    def runs(self) -> list[dict]:
        out = []
        if not os.path.isdir(self.exps_dir):
            return out
        for root, _, files in os.walk(self.exps_dir):
            if "cfg.json" not in files:
                continue
            with open(os.path.join(root, "cfg.json")) as f:
                cfg = json.load(f)
            latest = os.path.join(root, "latest.pkl")
            out.append({
                "name": os.path.relpath(root, self.exps_dir),
                "game": cfg["game"],
                "game_kwargs": cfg.get("game_kwargs", {}),
                "has_ckpt": os.path.exists(latest),
                "mtime": os.path.getmtime(latest) if os.path.exists(latest) else 0,
            })
        return sorted(out, key=lambda r: -r["mtime"])

    def metrics(self, run: str, since: int) -> dict:
        path = os.path.join(self.run_dir(run), "metrics.jsonl")
        records = []
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
            for line in lines[since:]:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # a line still being written
        return {"records": records, "next": since + len(records)}

    def new_game(self, run: str, seat: int, greedy: bool) -> dict:
        ckpt = utils.load_pickle(os.path.join(self.run_dir(run), "latest.pkl"))
        game_key = json.dumps([ckpt["game"], ckpt.get("game_kwargs", {})], sort_keys=True)
        with self.lock:
            if game_key not in self.runners:
                game = make(ckpt["game"], **ckpt.get("game_kwargs", {}))
                self.runners[game_key] = GameRunner(game)
            runner = self.runners[game_key]
            session_id = uuid.uuid4().hex
            session = Session(runner, ckpt, seat, greedy, seed=int.from_bytes(os.urandom(4)))
            self.sessions[session_id] = session
            return session.to_json(session_id)

    def act(self, session_id: str, action: int) -> dict:
        with self.lock:
            session = self.sessions[session_id]
            session.act(action)
            return session.to_json(session_id)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj, code: int = 200):
            self.send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    with open(os.path.join(STATIC, "index.html"), "rb") as f:
                        self.send(200, f.read(), "text/html; charset=utf-8")
                elif url.path == "/api/runs":
                    self.send_json(app.runs())
                elif url.path == "/api/metrics":
                    since = int(query.get("since", ["0"])[0])
                    self.send_json(app.metrics(query["run"][0], since))
                else:
                    self.send_json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001
                self.send_json({"error": str(e)}, 400)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            try:
                if self.path == "/api/new":
                    out = app.new_game(body["run"], int(body["seat"]), bool(body.get("greedy")))
                elif self.path == "/api/act":
                    out = app.act(body["session"], int(body["action"]))
                else:
                    return self.send_json({"error": "not found"}, 404)
                self.send_json(out)
            except Exception as e:  # noqa: BLE001
                self.send_json({"error": str(e)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps_dir", default="exps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(App(args.exps_dir)))
    print(f"Serving runs from {os.path.abspath(args.exps_dir)} at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
