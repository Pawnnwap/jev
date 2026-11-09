#!/usr/bin/env python3
"""军棋(暗棋)实时对局台: 浏览器里点"开始对局"就开局, 实时看棋 + 双方每一步用时。

双方各选一种棋手, 任意组合: JEV(决策模型) / RULE-AI(树搜索) / 人类(点子走子)。
  jev vs rule   —— 默认对局
  jev  vs jev   —— 两个决策模型互打(需 --api-twice, 见下)
  rule vs rule  —— 纯本地, 不需要 GPU/模型
  human vs 任意  —— 浏览器里点子走子, 裁判(Game.legal_moves)负责合法性, 不可能违规
暗棋对人生效: 页面按"该玩家视角"渲染, 敌子未确证时显示 ?; 视角可随时切到上帝视角观战。

  python junqi/live_play.py                 # 起服务(默认 8098), 浏览器打开该地址
  python junqi/live_play.py --selftest      # 代码自检: 只用 RULE-AI/人类, 不碰 GPU

一键脚本(仓库根目录): start_junqi_live.bat —— 起 llama-server + jev_adapter + 本服务并自动开浏览器。
"""
import argparse
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import junqi as J                                              # noqa: E402  (同目录裁判引擎)

SIDES = J.SIDES
SIDE_CN = J.SIDE_CN
MODES = ("jev", "ai", "human")
MODE_CN = {"jev": "JEV 决策模型", "ai": "RULE-AI 树搜索", "human": "人类(点子走子)"}
VIEWS = ("god", "red", "blue")
VIEW_CN = {"god": "上帝视角(观战)", "red": "红方视角", "blue": "蓝方视角"}


def other(side):
    return "blue" if side == "red" else "red"


def board_view(game, view):
    """按视角出棋盘: view=None/'god' 上帝真相; 否则己方真名, 敌子只在 belief 收窄到唯一时给名。"""
    out = []
    for r in range(12):
        row = []
        for c in range(5):
            p = game.cells.get((r, c))
            if p is None:
                row.append(None)
                continue
            name = p[1]
            if view in SIDES and p[0] != view:
                poss = game.knowledge[view].at((r, c))
                name = next(iter(poss)) if len(poss) == 1 else "?"
            row.append({"side": p[0], "name": name, "hq": J.cell_kind(r, c) == "hq",
                        "camp": J.cell_kind(r, c) == "camp", "rail": J.is_rail(r, c)})
        out.append(row)
    return out


def jev_alive(url, timeout=2.5):
    """JEV 决策服务是否在跑(只探 /health, 不发决策请求, 不占 GPU)。"""
    try:
        requests.get(url.rstrip("/") + "/health", timeout=timeout)
        return True
    except Exception:
        return False


def classify_move(desc, victim, gone):
    """把一步棋归类成音效事件: move/capture/trade/flag/loss。"""
    if "夺旗" in desc:
        return "flag"
    if len(gone) >= 2 or "同尽" in desc or "互亡" in desc or "触雷" in desc or "阵亡" in desc:
        return "trade" if len(gone) >= 2 else "loss"
    if victim is not None:
        return "capture"
    return "move"


class HumanPlayer:
    """人类棋手: 由浏览器 POST /api/move 送入, 合法性由裁判校验。"""

    def __init__(self, side, name="人类"):
        self.side, self.name = side, name

    def choose(self, game, side, moves):
        raise RuntimeError("人类着法由 /api/move 直接提交, 不经过 choose()")


class Session:
    """一局对局: 棋手组合、走子循环、每步用时统计、人类回合等待。"""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.human_move = None
        self.status = "idle"          # idle | running | waiting | finished | error | stopped
        self.error = ""
        self.tb = ""
        self.game = None
        self.players = {}
        self.plies = 0
        self.turn = None
        self.moves = []               # [{ply, side, desc, dt, info}]
        self.timing = {s: {"plies": 0, "total": 0.0, "max": 0.0, "max_desc": ""} for s in SIDES}
        self.winner = None
        self.reason = ""
        self.seconds = 0.0
        self.pieces, self.id_at, self.replay = [], {}, None
        self._cert, self._cert_key, self.cert_threshold = {}, None, 0.7
        self.view = "god"
        self.thread = None
        self.started_at = 0.0

    # ---------------- 人类输入 ----------------

    def human_side(self):
        for s in SIDES:
            if self.cfg.get(s) == "human":
                return s
        return None

    def submit(self, fr, to):
        """提交人类着法; 返回空串表示成功, 否则是拒绝原因。"""
        with self.lock:
            if self.status != "waiting" or self.game is None:
                return "现在不等待人类着法"
            mv = ((fr[0], fr[1]), (to[0], to[1]))
            if mv not in self.game.legal_moves(self.human_side()):
                return "非法着法(裁判拒绝)"
            self.human_move = mv
        self.wake.set()
        return ""

    def stop(self):
        with self.lock:
            if self.status in ("running", "waiting"):
                self.status = "stopped"
        self.wake.set()

    def _wait_human(self):
        with self.lock:                      # 先清空再置 waiting, 避免"已提交的着法"被清掉
            if self.status == "stopped":
                return None
            self.human_move = None
            self.wake.clear()
            self.status = "waiting"
        while True:
            if self.wake.wait(0.3):
                with self.lock:
                    if self.status == "stopped":
                        return None
                    if self.human_move is not None:
                        self.status = "running"
                        return self.human_move

    # ---------------- 对局循环 ----------------

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _build_players(self, cfg):
        players = {}
        for side in SIDES:
            mode = cfg[side]
            if mode == "ai":
                players[side] = J.RuleSearchAI(float(cfg["ai_time"]), int(cfg["ai_depth"]),
                                              samples=int(cfg["ai_samples"]),
                                              refine=int(cfg.get("ai_refine", 3)),
                                              procs=int(cfg.get("ai_procs", 0)))
            elif mode == "human":
                players[side] = HumanPlayer(side)
            else:
                players[side] = J.JevPlayer(cfg["url"], max_plies=int(cfg["max_plies"]))
        return players

    def _run(self):
        cfg = self.cfg
        t0 = time.monotonic()
        self.started_at = t0
        try:
            seed = int(cfg.get("seed") or 0) or random.randrange(1, 1 << 30)
            rng = random.Random(seed)
            game = J.Game(J.random_layout("blue", rng), J.random_layout("red", rng))
            players = self._build_players(cfg)
            max_plies = int(cfg["max_plies"])
            with self.lock:
                self.game, self.players, self.status = game, players, "running"
                self.view = self.human_side() or "god"
                self.pieces, self.id_at = [], {}          # 回放用: 棋子唯一 id + 占位表
                for s in SIDES:
                    for (r, c), (cs, name) in sorted(game.cells.items()):
                        if cs == s:
                            self.id_at[(r, c)] = len(self.pieces)
                            self.pieces.append({"id": len(self.pieces), "side": s, "name": name,
                                                "r": r, "c": c})
            while game.winner is None and self.plies < max_plies:
                with self.lock:
                    if self.status == "stopped":
                        return
                    side = "red" if self.plies % 2 == 0 else "blue"
                    self.turn = side
                moves = game.legal_moves(side)
                if not moves:
                    with self.lock:
                        game.winner = other(side)
                        self.reason = f"{SIDE_CN[side]}方无子可走(困毙)"
                    break
                t = time.monotonic()
                if isinstance(players[side], HumanPlayer):
                    mv = self._wait_human()
                    if mv is None:
                        return
                    info = "人类落子"
                else:
                    mv, meta = players[side].choose(game, side, moves)
                    if isinstance(players[side], J.JevPlayer):
                        if "fallback" in meta:
                            info = f"JEV接口失败回退随机: {meta['fallback']}"
                        else:
                            info = "概率: " + ", ".join(f"{k} {p * 100:.1f}%" for k, p in meta["top"])
                    else:
                        info = f"深度{meta.get('depth', '?')}, {meta.get('nodes', 0)}节点"
                dt = time.monotonic() - t
                with self.lock:
                    pid_mv = self.id_at.get(mv[0])
                    def_pid = self.id_at.get(mv[1])
                    attacker_name = game.cells[mv[0]][1]
                    victim, desc = game.apply(mv, side)
                    self.plies += 1
                    # 结算后更新占位表, 顺便算出这一步"发生了什么"(给音效用)
                    after_cell = game.cells.get(mv[1])
                    gone = []
                    if after_cell is None:                # 双亡
                        gone = [pid_mv] + ([def_pid] if def_pid is not None else [])
                        self.id_at.pop(mv[0], None)
                        self.id_at.pop(mv[1], None)
                    elif after_cell[0] == side and after_cell[1] == attacker_name:
                        self.id_at.pop(mv[0], None)
                        if def_pid is not None:
                            gone.append(def_pid)
                        self.id_at[mv[1]] = pid_mv
                    else:                                # 攻方亡, 守方原位
                        gone = [pid_mv]
                        self.id_at.pop(mv[0], None)
                    rec = {"ply": self.plies, "side": side, "fr": list(mv[0]), "to": list(mv[1]),
                           "desc": desc, "dt": round(dt, 2), "info": info,
                           "pid": pid_mv, "gone": gone, "ev": classify_move(desc, victim, gone)}
                    self.moves.append(rec)
                    tm = self.timing[side]
                    tm["plies"] += 1
                    tm["total"] += dt
                    if dt > tm["max"]:
                        tm["max"], tm["max_desc"] = dt, desc
            with self.lock:
                if game.winner is None:
                    mr, mb = game.material("red"), game.material("blue")
                    if mr != mb:
                        game.winner = "red" if mr > mb else "blue"
                        self.reason = f"步数达上限, 子力判定 (红{mr} vs 蓝{mb})"
                    else:
                        self.reason = f"步数达上限, 子力相当判和 (红{mr} vs 蓝{mb})"
                self.winner = game.winner or ""
                self.reason = self.reason or "未知"
                self.replay = self._build_replay()
                self.turn = None
                self.seconds = round(time.monotonic() - t0, 1)
                if self.status != "stopped":
                    self.status = "finished"
        except Exception as exc:                       # 任何异常都落到页面上, 不静默
            import traceback
            with self.lock:
                self.status = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                self.tb = traceback.format_exc()

    def _build_replay(self):
        """把这一局打包成上帝视角回放数据(结构同 junqi.play_game 的 replay)。"""
        cfg = self.cfg
        title = f"{self.plies}手  红方={MODE_CN[cfg['red']]}  蓝方={MODE_CN[cfg['blue']]}"
        return {"title": title,
                "result": {"winner": self.winner or "draw", "winner_side": self.winner or "",
                           "reason": self.reason, "plies": self.plies, "seconds": round(self.seconds)},
                "geometry": geometry(),
                "pieces": self.pieces,
                "moves": [{"ply": m["ply"], "side": m["side"], "pid": m.get("pid"),
                           "fr": m["fr"], "to": m["to"], "gone": m.get("gone", []),
                           "desc": m["desc"], "info": m.get("info", "")} for m in self.moves]}

    # ---------------- 对外状态 ----------------

    def state(self):
        with self.lock:
            game, cfg = self.game, self.cfg
            view = self.view if self.view in VIEWS else "god"
            human = self.human_side()
            payload = {
                "status": self.status, "error": self.error, "config": cfg, "view": view,
                "plies": self.plies, "turn": self.turn, "max_plies": int(cfg["max_plies"]),
                "modes": {s: cfg.get(s) for s in SIDES}, "human": human,
                "board": board_view(game, view) if game is not None else None,
                "god_board": board_view(game, None) if (game is not None and view in SIDES) else None,
                "moves": self.moves[-400:], "timing": self.timing,
                "winner": self.winner, "reason": self.reason, "seconds": self.seconds,
                "has_replay": self.replay is not None, "legal_from": {},
            }
            if game is not None and view in SIDES:
                # 该视角下"确定性>阈值"的敌子: 名字+概率, 页面打金圈显示
                key = (len(game.history), view)
                if self._cert_key != key:
                    self._cert_key = key
                    post_ = J.posterior_certainty(game, view, mc=120)
                    enemy_cells = {c for c, (os_, _n) in game.cells.items() if os_ != view}
                    self._cert = {}
                    for cell, item in post_.items():
                        name, pr = item[0], item[1]
                        rest = item[2] if len(item) > 2 else []
                        if pr > self.cert_threshold and cell in enemy_cells:
                            self._cert[J.coord(cell)] = [name, round(pr, 3),
                                                        [[n, round(v, 3)] for n, v in rest]]
                payload["certain"] = self._cert
                payload["cert_threshold"] = self.cert_threshold
            if game is not None and self.status == "waiting" and human:
                side = self.turn or human
                frm = {}
                for mv in game.legal_moves(side):
                    frm.setdefault(f"{mv[0][0]},{mv[0][1]}", []).append([mv[1][0], mv[1][1]])
                payload["legal_from"] = frm
                payload["turn"] = side
            return payload


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>军棋暗棋 · 实时对局台</title>
<style>
  :root{
    --paper:#f4f2e5; --ink:#5a4a32; --ink-soft:#a4ae93; --rail:#65785d;
    --red:#c02b21; --red-deep:#8f1d13; --blue:#34549b; --blue-deep:#24396b;
    --gold:#c6913e; --gold-soft:#e8c987;
    --felt:#22261e; --panel:#2b3026; --panel-edge:#3d4436; --panel-edge2:#4a5242;
    --text:#e9e4d4; --text-dim:#a8a290; --text-faint:#7d7867;
  }
  *{box-sizing:border-box}
  body{margin:0;color:var(--text);background:radial-gradient(1200px 800px at 30% 0%,#2b3227 0%,var(--felt) 60%);
       font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  .kai{font-family:"KaiTi","STKaiti","Noto Serif SC",serif}
  header{display:flex;align-items:center;gap:14px;padding:14px 22px 8px;flex-wrap:wrap}
  .seal{width:38px;height:38px;border-radius:8px;flex:none;display:flex;align-items:center;justify-content:center;
        font-size:22px;color:#f6e7d0;background:linear-gradient(160deg,#a03328,#7c211a);
        box-shadow:0 2px 8px rgba(0,0,0,.45),inset 0 0 0 1.5px rgba(246,231,208,.35)}
  h1{font-size:18px;margin:0;letter-spacing:.06em;font-weight:650}
  header small{display:block;font-size:11px;color:var(--text-dim);letter-spacing:.28em;margin-top:2px}
  .badge{margin-left:auto;font-size:12px;padding:5px 12px;border-radius:99px;background:var(--panel);
         border:1px solid var(--panel-edge);color:var(--text-dim)}
  .badge b.ok{color:#9dc98a} .badge b.bad{color:#e08a7a}
  .wrap{display:flex;gap:16px;padding:6px 22px 18px;align-items:flex-start;flex-wrap:wrap;max-width:1720px;margin:0 auto}
  .panel{background:var(--panel);border:1px solid var(--panel-edge);border-radius:14px;padding:12px 14px;
         margin-bottom:12px;box-shadow:0 6px 20px rgba(0,0,0,.25)}
  .panel h2{margin:0 0 9px;font-size:12px;color:var(--text-dim);font-weight:600;letter-spacing:.2em;
            display:flex;align-items:center;gap:8px}
  .panel h2::after{content:"";flex:1;height:1px;background:var(--panel-edge)}
  .cfg{display:flex;flex-direction:column;gap:7px;font-size:12.5px}
  .cfg .row{display:flex;gap:8px;align-items:center}
  .cfg label{color:var(--text-dim);flex:0 0 92px}
  .cfg .unit{color:var(--text-faint);font-size:11.5px}
  select,input{background:#363d2f;color:var(--text);border:1px solid var(--panel-edge2);border-radius:8px;
               padding:5px 8px;font:inherit;font-size:12.5px}
  input[type=number]{width:74px}
  input[type=text]{width:96px}
  .btns{display:flex;gap:8px;margin-top:11px;flex-wrap:wrap}
  button{background:#363d2f;color:var(--text);border:1px solid var(--panel-edge2);border-radius:9px;
         padding:7px 14px;font:inherit;cursor:pointer;transition:.15s}
  button:hover{border-color:var(--gold);color:var(--gold-soft)}
  button.primary{background:linear-gradient(175deg,#d9a441,#b9852e);border-color:#d9a441;color:#33270f;font-weight:700}
  button.primary:hover{filter:brightness(1.08)}
  button:disabled{opacity:.45;cursor:not-allowed}
  .board{position:relative;width:776px;height:836px;background:var(--paper);border-radius:10px;
         border:1px solid #c7cbb7;box-shadow:0 14px 44px rgba(0,0,0,.5)}
  .board svg{position:absolute;left:0;top:0}
  .board svg.arrow{pointer-events:none;z-index:6;overflow:visible}
  .piece.last{box-shadow:0 0 0 3px var(--gold-soft),0 0 18px rgba(232,201,135,.55),0 6px 12px rgba(0,0,0,.4)}
  .piece.moving{box-shadow:0 0 0 3px var(--gold-soft),0 0 18px rgba(232,201,135,.55),0 4px 0 rgba(0,0,0,.25);z-index:5}
  .grave{position:absolute;width:22px;height:22px;border-radius:50%;z-index:4;display:flex;align-items:center;justify-content:center;background:rgba(120,32,24,.88);color:#f3ded0;font-size:13px;font-weight:700;box-shadow:0 2px 6px rgba(0,0,0,.4),inset 0 0 0 1.5px rgba(243,222,208,.4);animation:pop .25s ease;pointer-events:none}
  .grave.flag{background:var(--gold);color:#3a2c12;box-shadow:0 2px 8px rgba(198,145,62,.7)}
  @keyframes pop{from{transform:scale(.3);opacity:0}}
  @keyframes arrowin{from{opacity:0}}
  .board .hit{position:absolute;border-radius:8px;cursor:default}
  .board .hit.mine{cursor:pointer}
  .board .hit.mine:hover{background:rgba(198,145,62,.18)}
  .board .hit.sel{background:rgba(198,145,62,.28);outline:2px solid var(--gold)}
  .board .hit.tgt{cursor:pointer;background:rgba(198,145,62,.16);outline:1.5px dashed var(--gold)}
  .piece{position:absolute;width:104px;height:34px;border-radius:7px;z-index:3;display:flex;align-items:center;
         justify-content:center;font-family:"KaiTi","STKaiti","Noto Serif SC",serif;font-size:18px;font-weight:800;
         letter-spacing:.14em;text-indent:.14em;color:#fdf6e8;text-shadow:0 1px 1px rgba(0,0,0,.45);
         pointer-events:none;transition:left .25s cubic-bezier(.4,.9,.35,1),top .25s cubic-bezier(.4,.9,.35,1)}
  .piece::before{content:"";position:absolute;inset:0;border-radius:7px;
         box-shadow:inset 0 1px 0 rgba(255,255,255,.35),inset 0 -2px 3px rgba(0,0,0,.22)}
  .piece.red{background:linear-gradient(175deg,#dd4a39 0%,var(--red) 55%,#a3251c 100%);
             box-shadow:0 4px 0 var(--red-deep),0 6px 10px rgba(0,0,0,.35)}
  .piece.blue{background:linear-gradient(175deg,#5478bd 0%,var(--blue) 55%,#2a4682 100%);
              box-shadow:0 4px 0 var(--blue-deep),0 6px 10px rgba(0,0,0,.35)}
  .piece.unk{background:linear-gradient(175deg,#8d8a7c,#6d6a5d);color:#f0ece0;letter-spacing:0;text-indent:0}
  .piece.cert{background:linear-gradient(175deg,#d8a94a,#a97c2c);color:#fff8e6;font-size:16px;letter-spacing:.04em;text-indent:.04em}
  .piece.cert.red{box-shadow:0 0 0 3px var(--gold-soft),0 0 16px rgba(232,201,135,.5),0 4px 0 var(--red-deep)}
  .piece.cert.blue{box-shadow:0 0 0 3px var(--gold-soft),0 0 16px rgba(232,201,135,.5),0 4px 0 var(--blue-deep)}
  .piece.cert .pct{font-family:"Segoe UI",sans-serif;font-size:11px;font-weight:700;opacity:.92;margin-left:4px;letter-spacing:0}
  .legend{font-size:11.5px;color:var(--text-faint);margin-top:7px}
  .legend b{color:var(--gold-soft)}
  .piece.last{box-shadow:0 0 0 3px var(--gold-soft),0 6px 12px rgba(0,0,0,.4)}
  .side{flex:0 0 300px;min-width:280px}
  .side.listcol{flex:1 1 340px;min-width:300px}
  .st{font-size:13px;color:var(--text-dim);line-height:1.8}
  .st b{color:var(--text)}
  .st .turn-red{color:#ff9d8c;font-weight:700} .st .turn-blue{color:#9cc0ff;font-weight:700}
  .hint{font-size:12.5px;color:var(--gold-soft);margin-top:6px;min-height:18px}
  table.t{width:100%;border-collapse:collapse;font-size:12.5px}
  table.t th,table.t td{padding:4px 6px;text-align:right;font-variant-numeric:tabular-nums}
  table.t th:first-child,table.t td:first-child{text-align:left}
  table.t th{color:var(--text-dim);font-weight:600;font-size:11.5px;letter-spacing:.08em}
  #mvlist{height:430px;overflow-y:auto;font-size:12.5px;scrollbar-width:thin;scrollbar-color:var(--panel-edge2) transparent}
  #mvlist .mv{padding:3px 8px;border-radius:7px;display:flex;gap:8px;align-items:baseline;line-height:1.6}
  #mvlist .mv:nth-child(odd){background:rgba(255,255,255,.03)}
  #mvlist .no{color:var(--text-faint);min-width:26px;text-align:right;font-variant-numeric:tabular-nums}
  #mvlist .dt{margin-left:auto;color:var(--text-faint);font-variant-numeric:tabular-nums;white-space:nowrap}
  #mvlist .dt.slow{color:#e0b06a;font-weight:700}
  .chip{font-size:10.5px;padding:1px 6px;border-radius:5px;font-weight:700;flex:none}
  .chip.red{background:rgba(192,43,33,.22);color:#ff9d8c;border:1px solid rgba(192,43,33,.5)}
  .chip.blue{background:rgba(52,84,155,.25);color:#9cc0ff;border:1px solid rgba(52,84,155,.55)}
  .res{font-size:13.5px;padding:7px 14px;border-radius:10px;background:rgba(0,0,0,.22);
       border:1px solid var(--panel-edge2);margin-top:8px}
  .res b.red{color:#ff9d8c} .res b.blue{color:#9cc0ff} .res b.gold{color:var(--gold-soft)}
  footer{padding:0 22px 20px;color:var(--text-faint);font-size:11.5px;line-height:1.8}
</style>
</head>
<body>
<header>
  <div class="seal kai">軍</div>
  <div>
    <h1>军棋 · 暗棋实时对局台</h1>
    <small class="kai">J E V &nbsp;·&nbsp; R U L E - A I &nbsp;·&nbsp; 人 类</small>
  </div>
  <span class="badge" id="jevbadge">JEV 服务检测中…</span>
</header>

<div class="wrap">
  <div>
    <div class="panel" style="width:776px">
      <h2>对 局 设 置</h2>
      <div class="cfg">
        <div class="row"><label>红方</label><select id="red" style="flex:1">
          <option value="jev">JEV 决策模型</option>
          <option value="ai" selected>RULE-AI 树搜索</option>
          <option value="human">人类(点子走子)</option></select></div>
        <div class="row"><label>蓝方</label><select id="blue" style="flex:1">
          <option value="jev">JEV 决策模型</option>
          <option value="ai" selected>RULE-AI 树搜索</option>
          <option value="human">人类(点子走子)</option></select></div>
        <div class="row"><label>视角</label><select id="view" style="flex:1">
          <option value="god">上帝视角(观战)</option>
          <option value="red">红方视角</option>
          <option value="blue">蓝方视角</option></select></div>
        <div class="row"><label>AI 每步时限</label><input id="ai_time" type="number" value="1.0" min="0.05" step="0.05">
          <span class="unit">秒/步</span></div>
        <div class="row"><label>AI 深度 / 样本</label><input id="ai_depth" type="number" value="4" min="1" max="6" step="1">
          <input id="ai_samples" type="number" value="4" min="1" max="16" step="1" title="每步从知识库采样的敌子世界个数"></div>
        <div class="row"><label>信念复评 / 并行</label><input id="ai_refine" type="number" value="3" min="0" max="8" step="1"
          title="信念感知复评的候选手数: 走完更新知识库再重采样评估">
          <input id="ai_procs" type="number" value="0" min="0" max="16" step="1" title="并行子进程数(0=串行)"></div>
        <div class="row"><label>手数上限</label><input id="max_plies" type="number" value="200" min="20" max="600" step="10"></div>
        <div class="row"><label>布局种子</label><input id="seed" type="text" style="flex:1" placeholder="留空=随机"></div>
      </div>
      <div class="btns">
        <button class="primary" id="bStart">开始对局</button>
        <button id="bStop" disabled>停止</button>
        <button id="bReplay" disabled>上帝视角回放</button>
        <button id="bMute" title="音效开关">🔊 音效开</button>
      </div>
      <div class="st" style="margin-top:8px">状态: <b id="stxt">空闲</b></div>
      <div class="legend" id="legend"></div>
      <div class="res" id="res" style="display:none"></div>
    </div>
  </div>

  <div>
    <div class="board" id="board"><svg class="base" id="lines"></svg><svg class="arrow" id="arrowLayer" width="776" height="836" viewBox="0 0 776 836"></svg></div>
  </div>

  <div class="side">
    <div class="panel">
      <h2>对 局 信 息</h2>
      <div class="st" id="info">点"开始对局"后在此显示回合与用时。</div>
      <div class="hint" id="hint"></div>
    </div>
    <div class="panel">
      <h2>用 时 统 计</h2>
      <table class="t">
        <thead><tr><th>棋手</th><th>手数</th><th>总用时</th><th>平均/步</th><th>最慢一步</th></tr></thead>
        <tbody id="timing"><tr><td colspan="5" style="color:var(--text-faint)">—</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="side listcol">
    <div class="panel">
      <h2>着 法 记 录 (含每步用时)</h2>
      <div id="mvlist"></div>
    </div>
  </div>
</div>
<footer>
  暗棋: 页面按所选视角渲染, 敌子未被排除法确证时只显示 <b>?</b>(灰色棋子); 切到"上帝视角"可观战。
  人类着法由裁判 <span class="kai">Game.legal_moves()</span> 校验, 违规点不进去也传不上来。
  裁判引擎与规则自检同源于 <span class="kai">junqi.py</span>。
  音效: OpenGameArt "Various Sound Effects" (Spring Spring, CC0)。
</footer>
<script>
const CSX = 128, CSY = 64, PADX = 68, PADY = 34;   /* 横向严格 2 倍: 格 128x64, 左右留白也加倍 */
const SW = PADX * 2 + 5 * CSX, SH = PADY * 2 + 12 * CSY;
const PW = 104, PH = 34;                     /* 棋子尺寸 */
const FRONT_Y = PADY + 6 * CSY;
const COLS = "abcde";
const SIDE_CN = { red: "红", blue: "蓝" };
const GEO = __GEOM__;
const board = document.getElementById("board");
const $ = id => document.getElementById(id);
let ST = null, SEL = null, LASTKEY = null, jevOK = null;

const cx = c => PADX + c * CSX + CSX / 2;
const cy = r => PADY + r * CSY + CSY / 2;
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
const coord = p => COLS[p[1]] + (p[0] + 1);

/* ---------- 棋盘底图(纸板/公路/铁路/行营/大本营/山界) ---------- */
function drawBase() {
  const line = (a, b, cls) =>
    `<line class="${cls}" x1="${cx(a[1])}" y1="${cy(a[0])}" x2="${cx(b[1])}" y2="${cy(b[0])}"/>`;
  let s = `<style>
    .frame{fill:none;stroke:#5a4a32;stroke-opacity:.5;stroke-width:1.2}
    .frame2{fill:none;stroke:#5a4a32;stroke-opacity:.28;stroke-width:1}
    .rd{stroke:#a4ae93;stroke-width:1.7} .dg{stroke:#a4ae93;stroke-width:1.2;stroke-dasharray:4 3.5}
    .rl{stroke:#65785d;stroke-width:6;stroke-linecap:round}
    .tie{stroke:${"#f4f2e5"};stroke-width:2.6;stroke-dasharray:2.2 6.2}
    .mtn{fill:none;stroke:#9aa68c;stroke-width:1.5;stroke-linejoin:round}
    .mtnfill{fill:#edeadb;stroke:none}
    .sz{font:11px "KaiTi",serif;fill:#8d937b;letter-spacing:.45em}
    .camp{fill:#e3ebd5;stroke:#9dac83;stroke-width:1.4}
    .camptx{font:11px "KaiTi",serif;fill:#8a937b;text-anchor:middle;letter-spacing:.2em}
    .hq{fill:#e7e2cb;stroke:#9b9879;stroke-width:1.3}
    .hqtx{font:10px "KaiTi",serif;fill:#a49a76;text-anchor:middle;letter-spacing:.15em}
    .wm{font:44px "KaiTi",serif;fill:#65785d;fill-opacity:.08;text-anchor:middle;letter-spacing:.6em}
    .rowlab{font:11px "Segoe UI",sans-serif;fill:#8d8371;text-anchor:middle}
    .collab{font:11px "Segoe UI",sans-serif;fill:#8d8371;text-anchor:middle}</style>`;
  s += `<rect class="frame" x="8" y="8" width="${SW - 16}" height="${SH - 16}" rx="6"/>`;
  s += `<rect class="frame2" x="13" y="13" width="${SW - 26}" height="${SH - 26}" rx="4"/>`;
  s += `<text class="wm" x="${cx(2)}" y="${cy(2) + 16}">陆战棋</text>`;
  s += `<text class="wm" x="${cx(2)}" y="${cy(9) + 16}">陆战棋</text>`;
  for (const [a, b] of GEO.road) s += line(a, b, "rd");
  {
    const y = FRONT_Y, seg = 30;
    let d = `M8 ${y}`;
    for (let x = 8; x < SW - 8; x += seg) {
      const pk = x + seg / 2, h = 9 + 5 * Math.abs(Math.sin(x * 12.9898 + 78.233));
      d += ` L${pk.toFixed(1)} ${(y - h).toFixed(1)} L${Math.min(x + seg, SW - 8)} ${y}`;
    }
    s += `<path class="mtnfill" d="${d} L${SW - 8} ${y + 13} L8 ${y + 13} Z"/>`;
    s += `<path class="mtn" d="${d}"/>`;
    s += `<text class="sz" x="${cx(0)}" y="${y + 4}" text-anchor="middle">山界</text>`;
    s += `<text class="sz" x="${cx(4)}" y="${y + 4}" text-anchor="middle">山界</text>`;
  }
  for (const [a, b] of GEO.diag) s += line(a, b, "dg");
  for (const [a, b] of GEO.rail) { s += line(a, b, "rl"); s += line(a, b, "tie"); }
  for (const [r, c] of GEO.camps)
    s += `<circle class="camp" cx="${cx(c)}" cy="${cy(r)}" r="22"/>` +
         `<text class="camptx" x="${cx(c)}" y="${cy(r) + 4}">行营</text>`;
  for (const [r, c] of GEO.hq)
    s += `<rect class="hq" x="${cx(c) - 28}" y="${cy(r) - 18}" width="56" height="36" rx="8"/>` +
         `<rect x="${cx(c) - 24.5}" y="${cy(r) - 14.5}" width="49" height="29" rx="6" fill="none" ` +
         `stroke="#9b9879" stroke-width="1" stroke-opacity=".6"/>` +
         `<text class="hqtx" x="${cx(c)}" y="${cy(r) + 3.5}">大本营</text>`;
  for (let r = 0; r < 12; r++)
    s += `<text class="rowlab" x="22" y="${cy(r) + 4}">${r + 1}</text>` +
         `<text class="rowlab" x="${SW - 22}" y="${cy(r) + 4}">${r + 1}</text>`;
  for (let c = 0; c < 5; c++) s += `<text class="collab" x="${cx(c)}" y="24">${COLS[c]}</text>`;
  const svg = $("lines");
  svg.setAttribute("width", SW); svg.setAttribute("height", SH);
  svg.setAttribute("viewBox", `0 0 ${SW} ${SH}`);
  svg.innerHTML = s;
}

/* ---------- 交互层: 60 个可点格 ---------- */
function buildHits() {
  for (let r = 0; r < 12; r++) for (let c = 0; c < 5; c++) {
    const d = document.createElement("div");
    d.className = "hit";
    d.style.left = PADX + c * CSX + "px"; d.style.top = PADY + r * CSY + "px";
    d.style.width = CSX + "px"; d.style.height = CSY + "px";
    d.dataset.r = r; d.dataset.c = c;
    d.addEventListener("click", () => onClick(r, c));
    board.appendChild(d);
  }
}

function onClick(r, c) {
  if (!ST || ST.status !== "waiting" || !ST.human) return;
  const side = ST.turn, key = r + "," + c;
  const cell = ST.board && ST.board[r][c];
  const legal = ST.legal_from || {};
  if (SEL && SEL.side === side) {
    if (SEL.r === r && SEL.c === c) { SEL = null; render(); return; }
    if ((legal[SEL.r + "," + SEL.c] || []).some(t => t[0] === r && t[1] === c)) {
      const from = [SEL.r, SEL.c];
      SEL = null;
      post("/api/move", { fr: from, to: [r, c] });
      return;
    }
  }
  if (cell && cell.side === side && legal[key]) SEL = { r, c, side };
  else SEL = null;
  render();
}

async function post(path, body) {
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify(body) });
    const j = await r.json();
    if (j.error) { $("hint").textContent = "⚠ " + j.error; }
  } catch (e) { $("hint").textContent = "⚠ " + e; }
  await poll();
}

/* ---------- 渲染 ---------- */
function render() {
  if (!ST) return;
  const b = ST.board || [];
  for (let r = 0; r < 12; r++) for (let c = 0; c < 5; c++) {
    const p = b[r] ? b[r][c] : null, key = r + "," + c;
    let el = board.querySelector(`.piece[data-k="${CSS.escape(key)}"]`);
    if (!p) { if (el) el.remove(); continue; }
    if (!el) {
      el = document.createElement("div");
      el.dataset.k = key; el.className = "piece";
      board.appendChild(el);
    }
    const cert = (ST.certain || {})[key];
    const unk = p.name === "?" && !cert;          /* 有确定性标注的格子不再显示问号 */
    el.className = "piece " + p.side + (unk ? " unk" : "") + (cert ? " cert" : "") +
      (LASTKEY === key ? " last" : "");
    if (cert) {
      /* 标注的是"本视角下的概率分布"(与真实身份无关): 最高可能棋种 + 其概率 */
      el.innerHTML = "";
      const nm = document.createElement("span");
      nm.textContent = cert[0];
      const pct = document.createElement("span");
      pct.className = "pct";
      pct.textContent = Math.round(cert[1] * 100) + "%";
      el.appendChild(nm);
      el.appendChild(pct);
      const dist = [[cert[0], cert[1]]].concat(cert[2] || []);
      el.title = "本视角推断: " + dist.map(d => d[0] + " " + Math.round(d[1] * 100) + "%").join(" / ")
                + (p.name !== "?" ? "   (原始显示: " + p.name + ")" : "");
    } else {
      el.textContent = unk ? "?" : p.name;
    }
    el.style.left = (PADX + c * CSX + (CSX - PW) / 2) + "px";
    el.style.top = (PADY + r * CSY + (CSY - PH) / 2) + "px";
  }
  const humanTurn = ST.status === "waiting" && ST.human;
  for (let r = 0; r < 12; r++) for (let c = 0; c < 5; c++) {
    const hit = board.querySelector(`.hit[data-r="${r}"][data-c="${c}"]`);
    const cell = b[r] ? b[r][c] : null, key = r + "," + c;
    const legal = (ST.legal_from || {})[key] || [];
    const isTgt = SEL && legal.some(t => t[0] === r && t[1] === c);
    const isMine = humanTurn && cell && cell.side === ST.turn && legal.length > 0;
    hit.className = "hit" + (SEL && SEL.r === r && SEL.c === c ? " sel" : "") +
      (isTgt ? " tgt" : "") + (isMine ? " mine" : "");
  }
  // 状态 / 信息 / 提示
  const stMap = { idle: "空闲", running: "对局进行中", waiting: "等待人类着法",
                  finished: "已结束", stopped: "已停止", error: "出错" };
  $("stxt").textContent = (stMap[ST.status] || ST.status) + (ST.error ? " — " + ST.error : "");
  const redM = { jev: "JEV", ai: "RULE-AI", human: "人类" }[ST.modes.red];
  const blueM = { jev: "JEV", ai: "RULE-AI", human: "人类" }[ST.modes.blue];
  let turnTxt = "";
  if (ST.status === "running" && ST.turn)
    turnTxt = `轮到 <span class="turn-${ST.turn}">${SIDE_CN[ST.turn]}方</span>`;
  $("info").innerHTML = `第 <b>${ST.plies}</b> / ${ST.max_plies} 手 · ${turnTxt}<br>` +
    `红方 <b>${redM}</b> vs 蓝方 <b>${blueM}</b> · 视角 <b>${
      {god: "上帝", red: "红方", blue: "蓝方"}[ST.view]}</b> · 总用时 <b>${ST.seconds || 0}s</b>` +
    (ST.reason ? `<br>${esc(ST.reason)}` : "");
  $("hint").textContent = humanTurn
    ? (SEL ? "已选中己方棋子: 点金色虚线格落子(再点一次取消)" : "轮到你: 点自己的棋子, 再点目标格")
    : (ST.status === "waiting" ? "等待中…" : "");
  const res = $("res");
  if (ST.status === "finished") {
    res.style.display = "block";
    const w = ST.winner;
    res.innerHTML = w === "draw" || !w
      ? `⚖ 结束 · ${esc(ST.reason)} · ${ST.plies}手 ${ST.seconds}s`
      : `<b class="${w}">${w === "red" ? "红方" : "蓝方"}</b> 胜 · ${esc(ST.reason)} · ${ST.plies}手 ${ST.seconds}s`;
  } else res.style.display = "none";
  // 用时统计
  const tm = ST.timing || {};
  $("timing").innerHTML = ["red", "blue"].map(s => {
    const t = tm[s] || { plies: 0, total: 0, max: 0, max_desc: "" };
    const avg = t.plies ? (t.total / t.plies) : 0;
    return `<tr><td><span class="chip ${s}">${s === "red" ? "红" : "蓝"}</span> ${
      ({ jev: "JEV", ai: "RULE-AI", human: "人类" })[ST.modes[s]]}</td>` +
      `<td>${t.plies}</td><td>${t.total.toFixed(1)}s</td><td>${avg.toFixed(2)}s</td>` +
      `<td style="color:var(--text-dim)">${t.max.toFixed(1)}s</td></tr>`;
  }).join("");
  // 着法记录
  const ms = ST.moves || [];
  $("mvlist").innerHTML = ms.length ? ms.map(m =>
    `<div class="mv"><span class="no">${m.ply}</span>` +
    `<span class="chip ${m.side}">${SIDE_CN[m.side]}</span>` +
    `<span>${esc(m.desc)}</span>` +
    `<span class="dt${m.dt > 5 ? " slow" : ""}">${m.dt.toFixed(2)}s</span></div>`).join("")
    : `<div style="color:var(--text-faint)">尚无着法</div>`;
  const lv = $("mvlist"); lv.scrollTop = lv.scrollHeight;
  $("bStop").disabled = !(ST.status === "running" || ST.status === "waiting");
  $("bReplay").disabled = !ST.has_replay;
  const nCert = Object.keys(ST.certain || {}).length;
  const isSideView = ST.view === "red" || ST.view === "blue";
  const thr = Math.round((ST.cert_threshold || 0.7) * 100);
  $("legend").innerHTML = !isSideView ? "上帝视角: 双方真实身份"
    : (nCert ? "金圈 = 该视角下身份确定性 &gt; <b>" + thr + "%</b> 的敌子 (共 <b>" + nCert + "</b> 颗; 标注=本视角推断, 非真实身份)"
      : "该视角下暂无确定性 &gt; " + thr + "% 的敌子 (都是问号)");
  const ms2 = ST.moves || [];
  if (ms2.length && ms2[ms2.length - 1].ply !== played.ply) {
    const last2 = ms2[ms2.length - 1];
    played.ply = last2.ply;
    if (last2.ev && last2.ev !== "flag") playSnd(last2.ev);
    drawArrow(last2); showGraves(last2); flashMove(last2);
  } else if (!ms2.length) {
    clearArrow(); clearGraves();
  }
}

async function poll() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    ST = await r.json();
    const ms = ST.moves || [];
    if (ms.length) {
      const last = ms[ms.length - 1];
      LASTKEY = last.to[0] + "," + last.to[1];
    }
    render();
  } catch (e) { /* 服务未起: 静默重试 */ }
}

async function jevBadge() {
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    const j = await r.json();
    jevOK = j.jev;
    $("jevbadge").innerHTML = j.jev
      ? `JEV 服务: <b class="ok">已连接</b> (${esc(j.url)})` : `JEV 服务: <b class="bad">未连接</b> (${esc(j.url)})`;
  } catch (e) { $("jevbadge").textContent = "对局台服务未响应"; }
}

$("bStart").addEventListener("click", () => {
  const cfg = { red: $("red").value, blue: $("blue").value, view: $("view").value,
                ai_time: parseFloat($("ai_time").value) || 1.0,
                ai_depth: parseInt($("ai_depth").value) || 4,
                ai_samples: parseInt($("ai_samples").value) || 4,
                ai_refine: parseInt($("ai_refine").value) || 0,
                ai_procs: parseInt($("ai_procs").value) || 0,
                max_plies: parseInt($("max_plies").value) || 200,
                seed: $("seed").value.trim() };
  SEL = null; LASTKEY = null; played.ply = 0;
  clearArrow(); clearGraves();
  post("/api/start", cfg);
});
$("bStop").addEventListener("click", () => post("/api/stop", {}));
$("view").addEventListener("change", () => post("/api/view", { view: $("view").value }));
$("bReplay").addEventListener("click", () => window.open("/replay", "junqi_replay"));
$("bMute").addEventListener("click", () => {
  sndOn.on = !sndOn.on;
  $("bMute").textContent = sndOn.on ? "🔊 音效开" : "🔇 音效关";
  if (sndOn.on) playSnd("move");
});

/* ---------------- 着法箭头 / 移动光环 / 阵亡徽记 (与回放页同一套实现) ---------------- */
const ARROW_DEFS = '<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" '
  + 'markerHeight="6.5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#c6913e"/></marker></defs>';
const arrowLayer = document.getElementById("arrowLayer");
let arrowTimer = 0, graveEls = [];
const clearArrow = () => { arrowLayer.innerHTML = ARROW_DEFS; };
function drawArrow(m) {   /* 当前着法箭头: 铁路远距离移动画金色虚线箭头, 淡入后自动隐去 */
  if (!m) { clearArrow(); return; }
  const fr = m.fr, tr = m.to;
  const x1 = cx(fr[1]), y1 = cy(fr[0]), x2 = cx(tr[1]), y2 = cy(tr[0]);
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  if (len < CSY * 1.6) { clearArrow(); return; }
  const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
  const o = 30;                                   /* 头尾各缩一点 */
  const ax = x1 + ux * o, ay = y1 + uy * o, bx = x2 - ux * (o - 6), by = y2 - uy * (o - 6);
  arrowLayer.innerHTML = ARROW_DEFS + '<line x1="' + ax + '" y1="' + ay + '" x2="' + bx + '" y2="' + by
    + '" stroke="#c6913e" stroke-width="3.5" stroke-dasharray="7 5.5" stroke-linecap="round"'
    + ' marker-end="url(#ah)" opacity=".92" style="animation:arrowin .3s ease"/>';
  clearTimeout(arrowTimer);
  arrowTimer = setTimeout(clearArrow, 2600);       /* 停一会自动淡出, 不长期挡视线 */
}
function clearGraves() { graveEls.forEach(el => el.remove()); graveEls = []; }
function showGraves(m) {        /* 落点上的阵亡徽记: 阵亡 ✕(双亡两个), 夺旗为金色"旗" */
  clearGraves();
  const gone = m.gone || [];
  if (!gone.length) return;
  const isFlag = /夺旗/.test(m.desc || "");
  const [r, c] = m.to;
  gone.forEach((gpid, i) => {
    const el = document.createElement("div");
    el.className = "grave" + (isFlag ? " flag" : "");
    el.textContent = isFlag ? "旗" : "✕";
    const dx = gone.length > 1 ? (i === 0 ? -12 : 12) : 0;
    el.style.left = (cx(c) - 11 + dx) + "px";
    el.style.top = (cy(r) - 11) + "px";
    board.appendChild(el);
    graveEls.push(el);
  });
}
function flashMove(m) {        /* 刚走的子套一圈金光环 */
  board.querySelectorAll(".piece.moving").forEach(el => el.classList.remove("moving"));
  const key = m.to[0] + "," + m.to[1];
  const el = board.querySelector('.piece[data-k="' + CSS.escape(key) + '"]');
  if (!el) return;
  el.classList.add("moving");
  setTimeout(() => { el.classList.remove("moving"); clearGraves(); }, 2600);
}

/* ---------------- 音效(CC0, 内嵌 data URI, 离线可用) ---------------- */
const SND = __SND__;
const sndOn = { on: true };
const played = { ply: 0 };
function playSnd(ev) {
  if (!sndOn.on || !SND[ev]) return;
  try {
    const a = new Audio(SND[ev]);
    a.volume = ev === "move" ? 0.35 : 0.6;
    a.play().catch(() => {});
  } catch (e) {}
}

drawBase(); buildHits(); jevBadge(); poll();
setInterval(poll, 800);
setInterval(jevBadge, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    session = None
    server_version = "junqi-live/1.0"

    def _send(self, code, body, ctype):
        raw = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/replay":
            rep = self.session.replay
            if rep is None:
                return self._json({"error": "还没有已结束的对局"}, 404)
            return self._send(200, J.render_replay_html(rep), "text/html; charset=utf-8")
        if path in ("/", "/index.html"):
            html = PAGE.replace("__GEOM__", json.dumps(geometry())).replace("__SND__", sound_data())
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(self.session.state())
        elif path == "/api/health":
            url = self.session.cfg.get("url", "")
            self._json({"jev": jev_alive(url) if url else False, "url": url,
                        "status": self.session.status})
        else:
            self._json({"error": "未知路径"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json({"error": "请求体不是 JSON"}, 400)
        path = self.path.split("?")[0]
        s = self.session
        if path == "/api/start":
            return self._json(self._start(body))
        if path == "/api/move":
            try:
                err = s.submit(body["fr"], body["to"])
            except Exception as exc:
                return self._json({"error": f"参数错误: {exc}"}, 400)
            return self._json({"error": err} if err else {"ok": True})
        if path == "/api/stop":
            s.stop()
            return self._json({"ok": True})
        if path == "/api/view":
            v = body.get("view")
            if v in VIEWS:
                s.view = v
                return self._json({"ok": True, "view": v})
            return self._json({"error": "视角只能是 god/red/blue"}, 400)
        return self._json({"error": "未知路径"}, 404)

    def _start(self, body):
        s = self.session
        with s.lock:
            if s.status in ("running", "waiting"):
                return {"error": "已有对局在进行, 请先停止"}
        cfg = {
            "red": body.get("red", "ai"), "blue": body.get("blue", "ai"),
            "ai_time": max(0.02, float(body.get("ai_time") or 1.0)),
            "ai_depth": max(1, min(7, int(body.get("ai_depth") or 4))),
            "ai_samples": max(1, min(24, int(body.get("ai_samples") or 4))),
            "ai_refine": max(0, min(8, int(body.get("ai_refine", 3)))),
            "ai_procs": max(0, min(16, int(body.get("ai_procs", 0)))),
            "max_plies": max(20, min(1000, int(body.get("max_plies") or 200))),
            "seed": str(body.get("seed") or "").strip(),
            "url": (body.get("url") or s.cfg.get("url") or "http://127.0.0.1:8090").rstrip("/"),
        }
        for side in SIDES:
            if cfg[side] not in MODES:
                return {"error": f"{SIDE_CN[side]}方棋手只能是 {MODES}"}
        if "jev" in (cfg["red"], cfg["blue"]) and not jev_alive(cfg["url"]):
            return {"error": f"JEV 决策服务不可达 ({cfg['url']}): 请先起 start_jev.bat, "
                             f"或把该方改成 RULE-AI / 人类"}
        s.__init__(cfg)                       # 重置为新对局(锁已在下面再取)
        s.view = body.get("view") if body.get("view") in VIEWS else (s.human_side() or "god")
        s.start()
        return {"ok": True, "config": cfg, "view": s.view}

    def log_message(self, *args):               # 静音默认访问日志
        pass


_SND_CACHE = None
SOUND_FILES = {"move": "move.wav", "capture": "capture.wav", "blast": "blast.wav",
               "flag": "flag.wav", "loss": "loss.wav"}
SOUND_CREDITS = ("音效: OpenGameArt ‘Various Sound Effects’ (作者 Spring Spring, CC0) "
                 "https://opengameart.org/content/various-sound-effects-0")


def sound_data():
    """把 junqi/sounds/*.wav 内嵌成 data URI(自包含页面, 离线可用; 缺文件就静音降级)。"""
    global _SND_CACHE
    if _SND_CACHE is None:
        import base64
        out = {}
        root = Path(__file__).resolve().parent / "sounds"
        for ev, fn in SOUND_FILES.items():
            f = root / fn
            if f.exists():
                b64 = base64.b64encode(f.read_bytes()).decode()
                out[ev] = "data:audio/wav;base64," + b64
        _SND_CACHE = out
    return json.dumps(_SND_CACHE)


def geometry():
    return dict(J.geometry_edges(), camps=[list(p) for p in J.CAMPS],
                hq=[list(p) for p in J.HQ_ALL], cols=5, rows=12)


# ---------------------------------------------------------------- 自检(不碰 GPU)

def wait_status(session, want, timeout=120.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = session.status
        if st == want:
            return True
        if want == "finished" and st in ("error", "stopped"):
            return False
        time.sleep(0.05)
    return False


def selftest():
    # 1) 纯规则AI对局: 走完 + 每步用时 + 统计自洽
    s1 = Session({"red": "ai", "blue": "ai", "ai_time": 0.05, "ai_depth": 2, "ai_samples": 2,
                  "max_plies": 60, "seed": "5", "url": "http://127.0.0.1:1"}).start()
    assert wait_status(s1, "finished", 300), f"规则AI对局未在限时内结束: {s1.status} {s1.error}"
    st1 = s1.state()
    assert st1["plies"] == len(s1.moves) >= 1, "着法记录条数应与手数一致"
    assert all(m["dt"] >= 0 for m in s1.moves), "每步都应有用时"
    for side in SIDES:
        t = st1["timing"][side]
        assert t["plies"] == sum(1 for m in s1.moves if m["side"] == side), "统计手数应等于该方着法数"
        assert abs(t["total"] - sum(m["dt"] for m in s1.moves if m["side"] == side)) < 0.5, "总用时应等于各步之和"
    print(f"  [1] 规则AI vs 规则AI: {st1['plies']}手 终局={st1['winner'] or '和'} "
          f"({st1['reason']}) 用时 {st1['seconds']}s")
    # 2) 人类 vs 规则AI: 等人类回合 → 提交合法着法 → 生效; 非法着法被裁判拒绝
    s2 = Session({"red": "human", "blue": "ai", "ai_time": 0.05, "ai_depth": 2, "ai_samples": 2,
                  "max_plies": 40, "seed": "6", "url": "http://127.0.0.1:1"}).start()
    assert wait_status(s2, "waiting"), f"应很快轮到人类: {s2.status}"
    assert s2.view == "red", "有人类时默认视角应为该人类所在方"
    st2 = s2.state()
    assert st2["legal_from"], "等待人类时应下发合法着法"
    fr = sorted(st2["legal_from"])[0]
    fr = [int(x) for x in fr.split(",")]
    tos = st2["legal_from"][f"{fr[0]},{fr[1]}"]
    time.sleep(0.2)                                # 模拟"想了一会儿"再落子
    assert s2.submit(fr, tos[0]) == "", "合法着法应被接受"
    for _ in range(100):
        if s2.moves:
            break
        time.sleep(0.05)
    assert s2.moves and s2.moves[0]["side"] == "red", "人类着法应已落子"
    assert s2.moves[0]["fr"] == fr and s2.moves[0]["to"] == list(tos[0]), "落子应与提交的一致"
    assert s2.moves[0]["dt"] >= 0.15, f"人类思考时间也应计入用时: {s2.moves[0]['dt']}"
    assert wait_status(s2, "waiting", 60), "AI 走一步后应再次轮到人类"
    st2b = s2.state()
    fr2 = [int(x) for x in sorted(st2b["legal_from"])[0].split(",")]
    bogus_from = [(fr2[0] + 1) % 12, fr2[1]]          # 起点不是自己的子 → 非法
    assert s2.submit(bogus_from, [fr2[0], fr2[1]]) != "", "非法着法必须被裁判拒绝"
    assert s2.submit([11, 4], [10, 4]) != "", "走出界的着法必须被裁判拒绝"
    s2.stop()
    assert wait_status(s2, "stopped", 10) or s2.status == "stopped", "停止应生效"
    print("  [2] 人类 vs 规则AI: 合法着法生效 / 非法着法被拒 / 停止生效 / 人类用时计入")
    # 3) 暗棋遮罩: 人类视角下敌子多为 '?', 上帝视角下全是真名
    s3 = Session({"red": "human", "blue": "human", "max_plies": 40, "seed": "7",
                  "url": "http://127.0.0.1:1"})
    g3 = J.Game(*[dict(J.random_layout(x, random.Random(7))) for x in ("blue", "red")])
    s3.game = g3
    masked = sum(1 for row in board_view(g3, "red") for p in row
                 if p and p["side"] == "blue" and p["name"] == "?")
    truth = sum(1 for row in board_view(g3, None) for p in row if p and p["side"] == "blue")
    assert masked > 10 and masked <= truth, f"暗棋遮罩异常: {masked}/{truth}"
    print(f"  [3] 暗棋遮罩: 红方视角下 {masked}/{truth} 颗蓝子显示为 '?', 上帝视角全真名")
    # 4) HTTP 层: 起真实服务, 走 / → /api/start → /api/state → 人类 /api/move
    Handler.session = Session({"red": "ai", "blue": "ai", "ai_time": 0.05, "ai_depth": 2,
                               "ai_samples": 2, "max_plies": 20, "seed": "8",
                               "url": "http://127.0.0.1:1"})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        r = requests.get(base + "/", timeout=10)
        assert r.status_code == 200 and "__GEOM__" not in r.text and "开始对局" in r.text, "首页应可打开"
        r = requests.get(base + "/api/health", timeout=10).json()
        assert r["jev"] is False and r["url"], "/api/health 应报告 JEV 不可达(本自检不连真实服务)"
        r = requests.post(base + "/api/start", json={"red": "ai", "blue": "ai", "ai_time": 0.05,
                                                     "ai_depth": 2, "ai_samples": 2,
                                                     "max_plies": 12, "seed": "9"}, timeout=10).json()
        assert r.get("ok"), f"/api/start 应成功: {r}"
        done = False
        for _ in range(600):
            stt = requests.get(base + "/api/state", timeout=10).json()
            if stt["status"] in ("finished", "error"):
                done = stt["status"] == "finished"
                break
            time.sleep(0.1)
        assert done, "HTTP 起的对局应能跑完"
        r = requests.post(base + "/api/start", json={"red": "human", "blue": "ai", "ai_time": 0.05,
                                                     "ai_depth": 2, "ai_samples": 1,
                                                     "max_plies": 20, "seed": "10"}, timeout=10).json()
        assert r.get("ok"), f"人类局应能开: {r}"
        for _ in range(200):
            stt = requests.get(base + "/api/state", timeout=10).json()
            if stt["status"] == "waiting":
                break
            time.sleep(0.1)
        assert stt["status"] == "waiting", "应等到人类回合"
        key = sorted(stt["legal_from"])[0]
        fr4 = [int(x) for x in key.split(",")]
        to4 = stt["legal_from"][key][0]
        r = requests.post(base + "/api/move", json={"fr": fr4, "to": to4}, timeout=10).json()
        assert r.get("ok"), f"人类着法应被接受: {r}"
        r = requests.post(base + "/api/start", json={"red": "ai", "blue": "ai"}, timeout=10).json()
        assert "error" in r and "进行" in r["error"], f"重复开局应被拒: {r}"
        requests.post(base + "/api/stop", json={}, timeout=10)
        for _ in range(100):
            if requests.get(base + "/api/state", timeout=10).json()["status"] in ("stopped", "error"):
                break
            time.sleep(0.1)
        r = requests.post(base + "/api/start", json={"red": "jev", "blue": "ai"}, timeout=10).json()
        assert "error" in r and "JEV" in r["error"], f"JEV 不可达时应明确报错: {r}"
        r = requests.post(base + "/api/view", json={"view": "blue"}, timeout=10).json()
        assert r.get("ok") and r["view"] == "blue", f"切视角应生效: {r}"
        print("  [4] HTTP: 首页/健康检查/开局/状态轮询/人类着法/重复开局拒绝/JEV不可达提示 全部正常")
    finally:
        httpd.shutdown()
    # 5) 终局后能拿到上帝视角回放(棋子身份 + 每手位移/阵亡), 音效事件也已分类
    s5 = Session({"red": "ai", "blue": "ai", "ai_time": 0.05, "ai_depth": 2, "ai_samples": 2,
                  "max_plies": 24, "seed": "3", "url": "http://127.0.0.1:1"}).start()
    assert wait_status(s5, "finished", 300), f"规则AI对局未结束: {s5.status} {s5.error}"
    st5 = s5.state()
    assert st5["has_replay"], "结束后应提供上帝视角回放"
    rep = s5.replay
    assert len(rep["pieces"]) == 50 and rep["geometry"]["camps"], "回放应含 50 枚棋子与棋盘几何"
    assert len(rep["moves"]) == st5["plies"] and all("pid" in m and "gone" in m for m in rep["moves"]), \
        "回放每手应带棋子 id 与阵亡名单"
    html5 = J.render_replay_html(rep)
    assert "__REPLAY_DATA__" not in html5 and len(html5) > 20000, "回放页应自包含"
    assert "width:776px" in html5, "回放棋盘应为横向 2 倍宽"
    evs5 = {m["ev"] for m in st5["moves"]}
    assert evs5 and evs5 <= {"move", "capture", "trade", "flag", "loss"}, f"音效事件分类异常: {evs5}"
    assert classify_move("工兵 a1→a2 吃蓝排长", "排长", [1]) == "capture"
    assert classify_move("炸弹 c1→d1 炸弹同尽", None, [1, 2]) == "trade"
    assert classify_move("连长 b1→b12 夺旗!", "军旗", [3]) == "flag"
    assert classify_move("排长 a1→a2", None, []) == "move"
    print(f"  [5] 上帝视角回放: {len(rep['pieces'])}子/{len(rep['moves'])}手, 回放页 {len(html5)} 字节; "
          f"音效事件 {sorted(evs5)}")
    # 6) 音效资源内嵌(CC0 wav -> data URI), 缺失时降级为静音
    snd = json.loads(sound_data())
    assert set(snd) == set(SOUND_FILES), f"音效应全部内嵌: {sorted(snd)}"
    assert all(v.startswith("data:audio/wav;base64,") for v in snd.values()), "音效应为内嵌 data URI"
    assert "width:776px" in PAGE.replace("__SND__", "{}"), "对局台棋盘应为横向 2 倍宽"
    print(f"  [6] 音效: {len(snd)} 个事件已内嵌({len(sound_data()) // 1024} KB), 棋盘 388->776 宽")
    # 7) 确定性>70% 的敌子标记: 后验必须与真相一致, 且只在选了对局视角时下发
    rg7 = random.Random(17)
    g7 = J.Game(J.random_layout("blue", rg7), J.random_layout("red", rg7))
    ai7 = {s: J.RuleSearchAI(0.05, 2, samples=2) for s in SIDES}
    for ply7 in range(50):
        if g7.winner:
            break
        s7 = "red" if ply7 % 2 == 0 else "blue"
        ms7 = g7.legal_moves(s7)
        if not ms7:
            break
        mv7, _ = ai7[s7].choose(g7, s7, ms7)
        g7.apply(mv7, s7)
    s7 = Session({"red": "human", "blue": "ai", "ai_time": 0.05, "ai_depth": 2, "ai_samples": 2,
                  "max_plies": 40, "seed": "17", "url": "http://127.0.0.1:1"})
    s7.game = g7
    s7.view = "red"
    st7 = s7.state()
    assert "certain" in st7 and st7["cert_threshold"] == 0.7, "选了对局视角应下发确定性名单"
    post7 = J.posterior_certainty(g7, "red", mc=120)
    for coord7, cert7 in st7["certain"].items():
        nm7, pr7 = cert7[0], cert7[1]
        rest7 = cert7[2]
        cell7 = next(c for c in g7.cells if J.coord(c) == coord7)
        assert g7.cells[cell7][0] != "red", f"只能标注敌子: {coord7}"
        assert pr7 > 0.7, f"标注的确定性必须超过阈值: {coord7} {pr7}"
        assert post7[cell7][0] == nm7 and abs(post7[cell7][1] - pr7) <= 0.02, \
            f"标注必须是该视角后验的众数与概率: {coord7} {nm7}{pr7} vs {post7[cell7]}"
        assert nm7 in g7.knowledge["red"].at(cell7), f"标注的棋种必须仍在可能集合里: {coord7} {nm7}"
        assert sum(x[1] for x in rest7) + pr7 <= 1.001 and all(x[1] > 0 for x in rest7), \
            f"次可能分布必须合法: {coord7} {rest7}"
        assert all(x[1] <= pr7 for x in rest7), f"次可能概率不该超过众数: {coord7} {rest7}"
        assert rest7 == sorted(rest7, key=lambda x: -x[1])[:3], f"次可能应按概率降序取前几项: {coord7} {rest7}"
    s7.view = "god"
    assert "certain" not in s7.state(), "上帝视角不该下发确定性名单"
    singles7 = [c for c, pid in g7.knowledge["red"].pid_at.items()
                if len(g7.knowledge["red"].poss[pid]) == 1]
    if singles7:
        for c7 in singles7:
            nm7 = next(iter(g7.knowledge["red"].poss[g7.knowledge["red"].pid_at[c7]]))
            assert post7[c7][0] == nm7 and post7[c7][1] > 0.95, f"已确定的敌子后验应为 100%: {J.coord(c7)}"
    ex7 = sorted(st7["certain"].items())[:3]
    print(f"  [7] 确定性>70%: 该视角标出 {len(st7['certain'])} 颗敌子(标注=本视角后验众数+概率, 与真相无关), "
          f"例: {[(k, v[0], v[1]) for k, v in ex7]}; 上帝视角不下发")
    print("LIVE_PLAY SELFTEST PASS: 会话循环 / 每步用时统计 / 人类着法(裁判校验) / 暗棋遮罩 / "
          "JEV缺失提示 / HTTP 接口 / 上帝视角回放 / 内嵌音效 / 横向2倍棋盘 全部通过")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="代码自检(不碰 GPU)")
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--url", default="http://127.0.0.1:8090", help="JEV 决策服务地址")
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    Handler.session = Session({"red": "ai", "blue": "ai", "ai_time": 1.0, "ai_depth": 4,
                               "ai_samples": 4, "max_plies": 200, "seed": "", "url": a.url})
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"军棋暗棋实时对局台: http://127.0.0.1:{a.port}/   (JEV: {a.url} "
          f"{'已连接' if jev_alive(a.url) else '未连接 — 仍可玩 RULE-AI / 人类'})", flush=True)
    if a.open:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}/")).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
