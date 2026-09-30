#!/usr/bin/env python3
"""军棋(二人陆战棋·明棋)环境: JEV(决策模型) vs 规则评估+α-β树搜索AI。

规则合规: 所有走子/吃子/胜负由 Game(裁判) 按《陆战棋》规则统一校验与执行,
双方只能从 legal_moves() 中选择, 物理上无法违规。
- 棋盘 5列(a-e)×12行(1-12); 行1-6=蓝方半场, 行7-12=红方; 行6-7间为前线(山界), 仅 b/c/d 三桥相通。
- 每方 25 子: 司令1 军长1 师长2 旅长2 团长2 营长2 连长3 排长3 工兵3 地雷3 炸弹2 军旗1。
- 大本营(行1/12 的 b,d): 进入的棋子不能再动; 军旗必须布在大本营。
- 行营(每方5个, 与周围8点有斜线): 其中的棋子不可被攻击。
- 公路一步; 铁路(前线两行 + a/e两侧列)直线任意远不可转弯; 工兵铁路内可转弯(不可越子)。
- 吃子: 司令>军长>师长>旅长>团长>营长>连长>排长>工兵, 同级互亡;
  炸弹与任何子同归于尽; 地雷不能动, 只有工兵可挖雷(工兵存活), 炸弹碰雷同尽。
- 胜负: 夺取军旗, 或对方无子可走(困毙); 手数达上限按子力判定(结果中标注"判定")。

用法:
  python junqi.py --selftest                 # 规则自检
  python junqi.py --games 1 --ai-time 2.0    # 对局(第1局红=JEV 蓝=AI, 之后交替换边)
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------- 常量与棋盘几何

SIDES = ("blue", "red")
SIDE_CN = {"blue": "蓝", "red": "红"}
COLS = "abcde"
RANKS = {"司令": 9, "军长": 8, "师长": 7, "旅长": 6, "团长": 5, "营长": 4, "连长": 3, "排长": 2, "工兵": 1}
PIECE_COUNTS = {"司令": 1, "军长": 1, "师长": 2, "旅长": 2, "团长": 2, "营长": 2,
                "连长": 3, "排长": 3, "工兵": 3, "地雷": 3, "炸弹": 2, "军旗": 1}
VALUES = {"司令": 100, "军长": 85, "师长": 70, "旅长": 55, "团长": 45, "营长": 35,
          "连长": 28, "排长": 22, "工兵": 42, "炸弹": 62, "地雷": 32, "军旗": 0}
BIG = 10 ** 9

CAMP_LOCAL = {(1, 1), (1, 3), (2, 1), (2, 2), (2, 3)}   # 行营(本方局部坐标, 0=本方底线行)
HQ_LOCAL = {(0, 1), (0, 3)}                              # 大本营


def local_row(r, side):
    """全局行(0-11, 0=显示行1=蓝方底线) -> 该方局部行(0=本方底线, 5=前线)."""
    return r if side == "blue" else 11 - r


def cell_kind(r, c):
    for side in SIDES:
        lr = local_row(r, side)
        if (lr, c) in CAMP_LOCAL:
            return "camp"
        if (lr, c) in HQ_LOCAL:
            return "hq"
    return "post"


def is_rail(r, c):
    return r in (5, 6) or (c in (0, 4) and 1 <= r <= 10)


def build_geometry():
    walk, rail = {}, {}
    for r in range(12):
        for c in range(5):
            walk[(r, c)] = []
            if is_rail(r, c):
                rail[(r, c)] = []
    for r in range(12):
        for c in range(5):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < 12 and 0 <= nc < 5):
                    continue
                if r == 5 and nr == 6 and c not in (1, 2, 3):
                    continue  # 山界: a/e 两列前线不通
                walk[(r, c)].append((nr, nc))
            if cell_kind(r, c) == "camp":
                for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 12 and 0 <= nc < 5 and (nr, nc) not in walk[(r, c)]:
                        walk[(r, c)].append((nr, nc))
    for pos in rail:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = pos[0] + dr, pos[1] + dc
            if (nr, nc) in rail and (nr, nc) in walk[pos]:
                rail[pos].append((nr, nc))
    return walk, rail


WALK, RAIL = build_geometry()
CAMPS = sorted(p for p in WALK if cell_kind(*p) == "camp")
HQ_ALL = sorted(p for p in WALK if cell_kind(*p) == "hq")


def coord(pos):
    return f"{COLS[pos[1]]}{pos[0] + 1}"


def random_layout(side, rng):
    """随机合法布局: 军旗入大本营; 地雷限本方后两排; 炸弹不得入本方第一排(前线行)."""
    cells = [(lr, c) for lr in range(6) for c in range(5)]   # 局部坐标, 0=底线, 5=前线
    rng.shuffle(cells)
    put = {}
    put[rng.choice([(0, 1), (0, 3)])] = "军旗"
    back = [p for p in cells if p[0] <= 1 and p not in put]
    for p in rng.sample(back, 3):
        put[p] = "地雷"
    rest = [p for p in cells if p not in put and p[0] != 5]
    for p in rng.sample(rest, 2):
        put[p] = "炸弹"
    rest = [p for p in cells if p not in put]
    pool = [n for n, k in PIECE_COUNTS.items() for _ in range(k)
            if n not in ("军旗", "地雷", "炸弹")]
    rng.shuffle(pool)
    for p, name in zip(rest, pool):
        put[p] = name
    assert sorted(put.values()) == sorted(n for n, k in PIECE_COUNTS.items() for _ in range(k))
    return {((lr if side == "blue" else 11 - lr), c): name for (lr, c), name in put.items()}


# ---------------------------------------------------------------- 裁判

class Game:
    def __init__(self, blue_pos, red_pos, check=True):
        self.cells = {}
        for (r, c), name in blue_pos.items():
            self.cells[(r, c)] = ("blue", name)
        for (r, c), name in red_pos.items():
            self.cells[(r, c)] = ("red", name)
        if check:
            assert len(self.cells) == 50
        self.winner = None
        self.history = []      # (side, move, victim|None, desc)
        self.quiet_plies = 0

    def legal_moves(self, side):
        res = []
        for (r, c), (s, name) in list(self.cells.items()):
            if s != side or name in ("地雷", "军旗") or cell_kind(r, c) == "hq":
                continue
            dests = set()
            for n in WALK[(r, c)]:
                if self._landable(n, side):
                    dests.add(n)
            if is_rail(r, c):
                if name == "工兵":
                    dests |= self._rail_bfs((r, c), side)
                else:
                    dests |= self._rail_slide((r, c), side)
            for d in dests:
                res.append(((r, c), d))
        return res

    def _landable(self, pos, side):
        p = self.cells.get(pos)
        if p is None:
            return True
        if p[0] == side:
            return False
        return cell_kind(*pos) != "camp"   # 行营内的敌子不可攻击

    def _rail_slide(self, pos, side):
        out = set()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cur = pos
            while True:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt not in RAIL.get(cur, []):
                    break
                p = self.cells.get(nxt)
                if p is None:
                    out.add(nxt)
                    cur = nxt
                    continue
                if p[0] != side and cell_kind(*nxt) != "camp":
                    out.add(nxt)
                break
        return out

    def _rail_bfs(self, pos, side):
        """工兵: 铁路网内任意路径, 不可越子, 可停空点或吃沿途敌子."""
        seen, frontier, caps = {pos}, [pos], set()
        while frontier:
            cur = frontier.pop()
            for nxt in RAIL.get(cur, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                p = self.cells.get(nxt)
                if p is None:
                    frontier.append(nxt)
                else:
                    caps.add(nxt)
        out = set(seen)
        out.discard(pos)
        for p2 in caps:
            if self.cells[p2][0] != side and cell_kind(*p2) != "camp":
                out.add(p2)
        return out

    def make(self, move):
        """执行着法(不校验), 返回 undo; 军旗被夺即定胜负."""
        (fr, fc), (tr, tc) = move
        a = self.cells[(fr, fc)]
        d = self.cells.get((tr, tc))
        undo = (move, a, d, self.winner, self.quiet_plies)
        if d is None:
            del self.cells[(fr, fc)]
            self.cells[(tr, tc)] = a
            self.quiet_plies += 1
            return undo, None
        an, dn = a[1], d[1]
        del self.cells[(fr, fc)]
        self.quiet_plies = 0
        if dn == "军旗":
            self.winner = a[0]
            self.cells[(tr, tc)] = a
        elif dn == "地雷":
            if an == "工兵":
                self.cells[(tr, tc)] = a      # 挖雷, 工兵存活
            elif an == "炸弹":
                del self.cells[(tr, tc)]      # 同归于尽
            # else: 攻方已删, 雷仍在原位
        elif an == "炸弹" or dn == "炸弹":
            del self.cells[(tr, tc)]           # 同归于尽
        elif RANKS[an] > RANKS[dn]:
            self.cells[(tr, tc)] = a
        elif RANKS[an] < RANKS[dn]:
            pass                               # 攻方亡, 守方原位不动
        else:
            del self.cells[(tr, tc)]           # 同级互亡, 目标格落空
        return undo, dn

    def unmake(self, undo):
        move, a, d, winner, quiet = undo
        (fr, fc), (tr, tc) = move
        self.cells.pop((tr, tc), None)
        self.cells[(fr, fc)] = a
        if d is not None:
            self.cells[(tr, tc)] = d
        self.winner = winner
        self.quiet_plies = quiet

    def describe(self, move, combat=True):
        """着法描述。明棋下战斗结果确定, 选项描述如实预告: 吃X/挖雷/触雷阵亡/同尽/阵亡于X/夺旗。"""
        (fr, fc), (tr, tc) = move
        name = self.cells[(fr, fc)][1]
        d = self.cells.get((tr, tc))
        if d is not None and combat:
            an, dn = name, d[1]
            if dn == "军旗":
                tag = " 夺旗!"
            elif dn == "地雷":
                tag = " 挖雷" if an == "工兵" else " 触雷阵亡"
            elif an == "炸弹" or dn == "炸弹":
                tag = " 同尽"
            elif RANKS[an] > RANKS[dn]:
                tag = f" 吃{SIDE_CN[d[0]]}{dn}"
            elif RANKS[an] == RANKS[dn]:
                tag = f" 与{dn}同尽"
            else:
                tag = f" 阵亡于{dn}"
        elif is_rail(fr, fc) and (tr, tc) in RAIL.get((fr, fc), []) and abs(fr - tr) + abs(fc - tc) > 1:
            tag = " 铁路"
        else:
            tag = ""
        return f"{name} {coord((fr, fc))}→{coord((tr, tc))}{tag}"

    def apply(self, move, side):
        """裁判入口: 校验合法后执行。返回 (victim|None, 着法描述). 描述在结算后生成,
        如实反映战斗结果(吃子/触雷/同归于尽), 而非"有守子即吃"."""
        if move not in self.legal_moves(side):
            raise ValueError(f"非法着法 {move} ({SIDE_CN[side]})")
        base = self.describe(move, combat=False)
        attacker = base.split(" ")[0]           # 着法描述首段即攻方棋子名
        d = self.cells.get(move[1])
        undo, victim = self.make(move)
        tag = ""
        if d is not None:
            after = self.cells.get(move[1])
            if d[1] == "军旗":
                tag = " 夺旗!"
            elif after is None:                  # 目标格落空: 双亡
                tag = ("炸弹碰雷同尽" if d[1] == "地雷"
                       else "炸弹同尽" if "炸弹" in (attacker, d[1]) else "同级互亡")
            elif after == d:                     # 守方原位不动: 攻方亡
                tag = "触雷阵亡" if d[1] == "地雷" else f"攻方阵亡于{d[1]}"
            else:
                tag = f"吃{SIDE_CN[d[0]]}{d[1]}"
        desc = base + (" " + tag if tag else "")
        self.history.append((side, move, victim, desc))
        return victim, desc

    # ---------------- 局面记录 ----------------

    def render(self):
        rows = []
        for r in range(12):
            cells = []
            for c in range(5):
                p = self.cells.get((r, c))
                t = f"{SIDE_CN[p[0]]}{p[1]}" if p else "空"
                k = cell_kind(r, c)
                if k == "camp":
                    t += "◎"
                elif k == "hq":
                    t += "◆"
                cells.append(t)
            rows.append(f"行{r + 1:>2}: " + " ".join(cells))
        return "\n".join(rows)

    def captured(self):
        cnt = {s: {n: 0 for n in PIECE_COUNTS} for s in SIDES}
        for (s, name) in self.cells.values():
            cnt[s][name] += 1
        out = {}
        for s in SIDES:
            lost = []
            for n, k in PIECE_COUNTS.items():
                miss = k - cnt[s][n]
                if miss:
                    lost.append(f"{n}x{miss}" if miss > 1 else n)
            out[s] = "、".join(lost) or "无"
        return out

    def material(self, side):
        return sum(VALUES[name] for (s, name) in self.cells.values() if s == side)

    def snapshot(self):
        return (dict(self.cells), self.winner, self.quiet_plies)

    def restore(self, snap):
        self.cells, self.winner, self.quiet_plies = dict(snap[0]), snap[1], snap[2]


# ---------------------------------------------------------------- 规则评估 + α-β 树搜索 AI

def evaluate(g, side):
    score = 0
    for (r, c), (s, name) in g.cells.items():
        val = VALUES[name]
        if name not in ("地雷", "军旗"):
            adv = (11 - r) if s == "red" else r      # 向敌方底线推进量
            val += 0.25 * adv + (2 if cell_kind(r, c) == "camp" else 0)
        score += val if s == side else -val
    return score


class RuleSearchAI:
    """规则评估 + 迭代加深 α-β 树搜索(时限内尽量加深)."""

    def __init__(self, time_budget=2.0, max_depth=4, name="RULE-AI"):
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.name = name
        self.nodes = 0
        self.deadline = 0.0

    def choose(self, game, side, moves):
        self.deadline = time.monotonic() + self.time_budget
        self.nodes = 0
        opp = "blue" if side == "red" else "red"
        best, best_depth, best_score = moves[0], 0, 0
        for depth in range(2, self.max_depth + 1):
            snap = game.snapshot()
            alpha, cur_best = -BIG, None
            try:
                for m in self._ordered(game, side):
                    undo, _ = game.make(m)
                    v = -self._negamax(game, opp, depth - 1, -BIG, -alpha)
                    game.unmake(undo)
                    if v > alpha:
                        alpha, cur_best = v, m
            except TimeoutError:
                game.restore(snap)      # 超时路径上有未回退的 make, 整层回滚
                break
            best, best_depth, best_score = cur_best, depth, alpha
            if time.monotonic() > self.deadline or alpha >= BIG - 1000:
                break
        return best, {"depth": best_depth, "nodes": self.nodes, "score": best_score}

    def _ordered(self, game, side):
        caps, quiets = [], []
        for m in game.legal_moves(side):
            d = game.cells.get(m[1])
            if d is not None and d[0] != side:
                caps.append((VALUES[d[1]], m))
            else:
                quiets.append(m)
        caps.sort(key=lambda x: -x[0])
        return [m for _, m in caps] + quiets

    def _negamax(self, game, side, depth, alpha, beta):
        self.nodes += 1
        if self.nodes % 512 == 0 and time.monotonic() > self.deadline:
            raise TimeoutError
        if game.winner is not None:
            return BIG if game.winner == side else -BIG
        moves = game.legal_moves(side)
        if not moves:
            return -BIG          # 困毙(轮到该方无子可走即负)
        if depth == 0:
            return evaluate(game, side)
        opp = "blue" if side == "red" else "red"
        caps, quiets = [], []
        for m in moves:
            d = game.cells.get(m[1])
            (caps if d is not None and d[0] != side else quiets).append(m)
        caps.sort(key=lambda m: -VALUES[game.cells[m[1]][1]])
        best = -BIG
        for m in caps + quiets:
            undo, _ = game.make(m)
            v = -self._negamax(game, opp, depth - 1, -beta, -alpha)
            game.unmake(undo)
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        return best


# ---------------------------------------------------------------- JEV 棋手 (决策模型)

RULES_TEXT = (
    "军棋(陆战棋·明棋)规则: 棋盘5列(a-e)×12行(1-12), 行1-6是蓝方半场, 行7-12是红方半场; "
    "行6-7之间是前线山界, 仅b/c/d三座桥相通。公路走一步; 铁路(第6、7行整行和a/e两侧列)上棋子"
    "可直线任意远(不可转弯), 工兵在铁路上可任意转弯但不可越子。行营(◎)中的棋子不可被攻击; "
    "大本营(◆)中的棋子不能再移动。吃子大小: 司令>军长>师长>旅长>团长>营长>连长>排长>工兵, "
    "同级同归于尽; 炸弹与任何子同归于尽; 地雷不能动, 除工兵可挖雷(工兵存活)和炸弹碰雷同尽外, "
    "任何子碰雷阵亡。夺取敌方军旗或使对方无子可走即获胜。"
)


class JevPlayer:
    def __init__(self, url="http://127.0.0.1:8090", name="JEV", temperature=1.036):
        self.url = url.rstrip("/")
        self.name = name
        self.temperature = temperature
        self.http = requests.Session()

    def choose(self, game, side, moves):
        capped, trimmed = list(moves), False
        if len(moves) > 250:      # 接口上限255, 用吃子着法优先裁剪(实际对局极少触发)
            caps = [m for m in moves if game.cells.get(m[1])]
            rest = [m for m in moves if not game.cells.get(m[1])]
            capped, trimmed = (caps + rest)[:250], True
        criteria = {f"M{i:03d}": game.describe(m) for i, m in enumerate(capped)}
        payload = {"state": self.state_text(game, side),
                   "questions": {"move": {
                       "type": "choice",
                       "instructions": f"你是{SIDE_CN[side]}方。下面是当前局面的全部合法着法(选项), "
                                       f"选出对{SIDE_CN[side]}方最有利的一步。"
                                       + ("(着法过多, 已按吃子优先裁剪)" if trimmed else ""),
                       "criteria": criteria}}}
        last_err = None
        for _ in range(3):
            try:
                r = self.http.post(self.url + "/v1/systemone", json=payload, timeout=900)
                r.raise_for_status()
                ans = r.json()["answers"]["move"]
                key = ans["choice"]
                if key in criteria:
                    move = capped[int(key[1:])]
                    top = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])[:3]
                    return move, {"top": [(k, round(p, 3)) for k, p in top],
                                  "confidence": round(ans["confidence"], 3)}
                last_err = RuntimeError(f"无效选择 {key!r}")
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        return random.choice(capped), {"fallback": str(last_err)}

    def state_text(self, game, side):
        captured = game.captured()
        hist = "  ".join(f"{len(game.history) - len(game.history[-8:]) + i + 1}.{SIDE_CN[h[0]]} {h[3]}"
                         for i, h in enumerate(game.history[-8:]))
        return (f"你是{SIDE_CN[side]}方(JEV), 对手是蓝方AI。{RULES_TEXT}\n"
                f"行营位置: {' '.join(coord(p) for p in CAMPS)}。"
                f"大本营位置: {' '.join(coord(p) for p in HQ_ALL)}。\n"
                f"当前棋盘(第{len(game.history) + 1}手, 轮到{SIDE_CN[side]}方):\n{game.render()}\n"
                f"红方阵亡: {captured['red']} | 蓝方阵亡: {captured['blue']}\n"
                f"子力价值: 红{game.material('red')} vs 蓝{game.material('blue')}\n"
                f"最近着法: {hist or '开局'}")


# ---------------------------------------------------------------- 对局

def geometry_edges():
    """可视化用的连线: 公路(细线)/铁路(粗线)/行营斜线(虚线)。"""
    road, rail_e, diag = [], [], []
    seen = set()
    for pos, ns in WALK.items():
        for n in ns:
            key = (pos, n) if pos <= n else (n, pos)
            if key in seen:
                continue
            seen.add(key)
            a, b = key
            if abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1:
                diag.append([list(a), list(b)])
            elif b in RAIL.get(a, []):
                rail_e.append([list(a), list(b)])
            else:
                road.append([list(a), list(b)])
    return {"road": road, "rail": rail_e, "diag": diag}


def play_game(idx, args, log):
    rng = random.Random(args.seed + idx)
    g = Game(random_layout("blue", rng), random_layout("red", rng))
    jev = JevPlayer(args.url)
    ai = RuleSearchAI(args.ai_time, args.ai_depth)
    jev_side = "red" if idx % 2 == 0 else "blue"
    players = {"red": jev if jev_side == "red" else ai,
               "blue": ai if jev_side == "red" else jev}
    log(f"=== 第{idx + 1}局  红方={players['red'].name}  蓝方={players['blue'].name} ===")
    log(g.render())
    # 回放采集: 棋子唯一 id + 占位表, 每手记录位移与阵亡名单
    pieces, id_at = [], {}
    for side in SIDES:
        for (r, c), (s, name) in sorted(g.cells.items()):
            if s == side:
                id_at[(r, c)] = len(pieces)
                pieces.append({"id": len(pieces), "side": side, "name": name, "r": r, "c": c})
    replay_moves = []
    reason = None
    ply = 0
    t_start = time.monotonic()
    while g.winner is None and ply < args.max_plies:
        side = "red" if ply % 2 == 0 else "blue"
        moves = g.legal_moves(side)
        if not moves:
            g.winner = "blue" if side == "red" else "red"
            reason = f"{SIDE_CN[side]}方无子可走(困毙)"
            break
        player = players[side]
        t0 = time.monotonic()
        move, meta = player.choose(g, side, moves)
        dt = time.monotonic() - t0
        fr, to = move
        pid, defender_pid = id_at[fr], id_at.get(to)
        attacker_name = g.cells[fr][1]
        victim, desc = g.apply(move, side)
        ply += 1
        # 依据结算后的目标格状态更新占位表并生成阵亡名单
        after = g.cells.get(to)
        gone = []
        if after is None:                       # 双亡(炸弹/同级/炸弹碰雷)
            gone = [pid] + ([defender_pid] if defender_pid is not None else [])
            id_at.pop(fr, None)
            id_at.pop(to, None)
        elif after[0] == side and after[1] == attacker_name:   # 攻方进驻(移动/吃子/挖雷)
            id_at.pop(fr, None)
            if defender_pid is not None:
                gone.append(defender_pid)
            id_at[to] = pid
        else:                                   # 攻方亡, 守方原位(触雷/打不过)
            gone = [pid]
            id_at.pop(fr, None)
        extra = ""
        if isinstance(player, JevPlayer) and "top" in meta:
            extra = "概率: " + ", ".join(f"{k} {p * 100:.1f}%" for k, p in meta["top"])
        elif not isinstance(player, JevPlayer):
            extra = f"深度{meta.get('depth', '?')}, {meta.get('nodes', 0)}节点"
        if "fallback" in meta:
            extra += f" [JEV接口失败回退随机: {meta['fallback']}]"
        replay_moves.append({"ply": ply, "side": side, "pid": pid,
                             "fr": [fr[0], fr[1]], "to": [to[0], to[1]], "gone": gone,
                             "desc": desc, "info": extra})
        log(f"{ply:>3}. [{SIDE_CN[side]}/{player.name}] {desc}  {extra}  ({dt:.1f}s)")
        if g.winner is not None and victim == "军旗":
            reason = "夺取军旗"
    if g.winner is None:
        m_red, m_blue = g.material("red"), g.material("blue")
        if m_red != m_blue:
            g.winner = "red" if m_red > m_blue else "blue"
            reason = f"步数达上限, 子力判定 (红{m_red} vs 蓝{m_blue})"
        else:
            reason = f"步数达上限, 子力相当判和 (红{m_red} vs 蓝{m_blue})"
    log("--- 结果: " + (f"{SIDE_CN[g.winner]}方({players[g.winner].name})胜" if g.winner else "平局")
        + f" | 原因: {reason} | 共{ply}手, 用时{time.monotonic() - t_start:.0f}s ---")
    log(g.render())
    replay = {"title": f"第{idx + 1}局  红方={players['red'].name}  蓝方={players['blue'].name}",
              "result": {"winner": players[g.winner].name if g.winner else "draw",
                         "winner_side": g.winner or "",
                         "reason": reason or "", "plies": ply,
                         "seconds": round(time.monotonic() - t_start)},
              "geometry": dict(geometry_edges(), camps=[list(p) for p in CAMPS],
                               hq=[list(p) for p in HQ_ALL], cols=5, rows=12),
              "pieces": pieces,
              "moves": replay_moves}
    return {"game": idx + 1, "red": players["red"].name, "blue": players["blue"].name,
            "winner": players[g.winner].name if g.winner else "draw",
            "reason": reason, "plies": ply,
            "seconds": round(time.monotonic() - t_start), "replay": replay}


# ---------------------------------------------------------------- 可视化回放 (自包含 HTML)

REPLAY_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — 军棋回放</title>
<style>
  /* 视觉规格参考开源项目: phdsx/junqi (纸板棋盘/枕木铁路/行营圆/大本营双框, AGPL)
     + BitByBitG/siguo-junqi (MIT, SVG分层与回放箭头) + samuelyuan/online-junqi (MIT, 立体棋子) */
  :root {
    --paper:#f4f2e5; --paper-dim:#e7e2cb; --paper-green:#e3ebd5;
    --ink:#5a4a32; --ink-soft:#a4ae93; --rail:#65785d;
    --red:#c02b21; --red-deep:#8f1d13; --blue:#34549b; --blue-deep:#24396b;
    --gold:#c6913e; --gold-soft:#e8c987;
    --felt:#22261e; --panel:#2b3026; --panel-edge:#3d4436; --panel-edge2:#4a5242;
    --text:#e9e4d4; --text-dim:#a8a290; --text-faint:#7d7867;
    --cs:64px;
  }
  * { box-sizing:border-box; }
  html { background:var(--felt); }
  body { margin:0; color:var(--text);
         font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;
         background:radial-gradient(1200px 800px at 30% 0%, #2b3227 0%, var(--felt) 60%); }
  .kai { font-family:"KaiTi","STKaiti","Noto Serif SC",serif; }

  header { display:flex; align-items:center; gap:18px; padding:18px 26px 10px; flex-wrap:wrap; }
  .ttl { display:flex; align-items:center; gap:12px; }
  .ttl .seal { width:40px; height:40px; border-radius:8px; flex:none;
               background:linear-gradient(160deg,#a03328,#7c211a); color:#f6e7d0;
               display:flex; align-items:center; justify-content:center;
               font-size:24px; box-shadow:0 2px 8px rgba(0,0,0,.45), inset 0 0 0 1.5px rgba(246,231,208,.35); }
  .ttl h1 { font-size:19px; margin:0; letter-spacing:.06em; font-weight:650; }
  .ttl small { display:block; font-size:11.5px; color:var(--text-dim); letter-spacing:.3em; margin-top:3px; }
  .res { margin-left:auto; font-size:13.5px; padding:6px 16px; border-radius:99px;
         background:var(--panel); border:1px solid var(--panel-edge); color:var(--text-dim); }
  .res b.red { color:#ff9d8c; } .res b.blue { color:#9cc0ff; } .res b.gold { color:var(--gold-soft); }

  .wrap { display:flex; gap:22px; padding:6px 26px 10px; align-items:flex-start; flex-wrap:wrap; }

  /* ---------------- 棋盘: 纸板 + 印刷框线 ---------------- */
  .boardcol { flex:none; }
  .board { position:relative; width:388px; height:836px; background:var(--paper);
           border-radius:10px; border:1px solid #c7cbb7;
           box-shadow:0 14px 44px rgba(0,0,0,.5), 0 2px 0 rgba(255,255,255,.06) inset; }
  .board::after { content:""; position:absolute; inset:0; border-radius:10px; pointer-events:none;
           box-shadow:inset 0 0 60px rgba(90,74,50,.14); }   /* 纸面晕影 */
  .board > svg, .board > div { position:absolute; left:0; top:0; }
  svg.base { border-radius:10px; }
  svg.arrow { pointer-events:none; z-index:6; overflow:visible; }

  /* ---------------- 棋子: 立体瓷片 ---------------- */
  .piece { position:absolute; width:58px; height:32px; border-radius:7px; z-index:3;
           display:flex; align-items:center; justify-content:center;
           font-family:"KaiTi","STKaiti","Noto Serif SC",serif;
           font-size:18px; font-weight:800; letter-spacing:.14em; text-indent:.14em;
           color:#fdf6e8; text-shadow:0 1px 1px rgba(0,0,0,.45);
           transition:left .3s cubic-bezier(.4,.9,.35,1), top .3s cubic-bezier(.4,.9,.35,1),
                      opacity .3s, transform .3s; will-change:left,top; }
  .piece::before { content:""; position:absolute; inset:0; border-radius:7px;
           box-shadow:inset 0 1px 0 rgba(255,255,255,.35), inset 0 -2px 3px rgba(0,0,0,.22); }
  .piece.red  { background:linear-gradient(175deg,#dd4a39 0%, var(--red) 55%, #a3251c 100%);
                box-shadow:0 4px 0 var(--red-deep), 0 6px 10px rgba(0,0,0,.35); }
  .piece.blue { background:linear-gradient(175deg,#5478bd 0%, var(--blue) 55%, #2a4682 100%);
                box-shadow:0 4px 0 var(--blue-deep), 0 6px 10px rgba(0,0,0,.35); }
  .piece.dead { opacity:0; transform:scale(.55) rotate(6deg); pointer-events:none; }
  .piece.moving { box-shadow:0 0 0 3px var(--gold-soft), 0 0 18px rgba(232,201,135,.55),
                  0 4px 0 rgba(0,0,0,.25); z-index:5; }

  /* 阵亡/夺旗 标记 */
  .grave { position:absolute; width:22px; height:22px; border-radius:50%; z-index:4;
           display:flex; align-items:center; justify-content:center;
           background:rgba(120,32,24,.88); color:#f3ded0; font-size:13px; font-weight:700;
           box-shadow:0 2px 6px rgba(0,0,0,.4), inset 0 0 0 1.5px rgba(243,222,208,.4);
           animation:pop .25s ease; pointer-events:none; }
  .grave.flag { background:var(--gold); color:#3a2c12; box-shadow:0 2px 8px rgba(198,145,62,.7); }
  @keyframes pop { from { transform:scale(.3); opacity:0; } }

  /* ---------------- 右侧面板 ---------------- */
  .side { flex:1 1 320px; min-width:310px; max-width:460px; }
  .panel { background:var(--panel); border:1px solid var(--panel-edge); border-radius:14px;
           padding:13px 15px; margin-bottom:14px; box-shadow:0 6px 20px rgba(0,0,0,.25); }
  .panel h2 { margin:0 0 9px; font-size:12px; color:var(--text-dim); font-weight:600;
              letter-spacing:.22em; display:flex; align-items:center; gap:8px; }
  .panel h2::after { content:""; flex:1; height:1px; background:var(--panel-edge); }

  .ctrl { display:flex; gap:7px; align-items:center; flex-wrap:wrap; }
  .ctrl button { background:#363d2f; color:var(--text); border:1px solid var(--panel-edge2);
                 border-radius:9px; padding:7px 13px; font:inherit; cursor:pointer; transition:.15s; }
  .ctrl button:hover { border-color:var(--gold); color:var(--gold-soft); }
  .ctrl button.primary { background:linear-gradient(175deg,#d9a441,#b9852e); border-color:#d9a441;
                 color:#33270f; font-weight:700; }
  .ctrl button.primary:hover { filter:brightness(1.07); color:#33270f; }
  .ctrl select { background:#363d2f; color:var(--text); border:1px solid var(--panel-edge2);
                 border-radius:8px; padding:5px 8px; font:inherit; }
  .ctrl .spd { font-size:12px; color:var(--text-dim); }
  .seekrow { display:flex; gap:10px; align-items:center; margin-top:10px; }
  .seekrow input[type=range] { flex:1; accent-color:var(--gold); }
  .seekrow .ply { font-size:12.5px; color:var(--text-dim); min-width:64px; text-align:right; font-variant-numeric:tabular-nums; }
  #curdesc { margin-top:9px; font-size:13.5px; color:var(--gold-soft); min-height:22px;
             display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
  #curdesc .chip { flex:none; }

  .chip { font-size:11px; padding:1px 7px; border-radius:5px; flex:none; font-weight:700; }
  .chip.red  { background:rgba(192,43,33,.22); color:#ff9d8c; border:1px solid rgba(192,43,33,.5); }
  .chip.blue { background:rgba(52,84,155,.25); color:#9cc0ff; border:1px solid rgba(52,84,155,.55); }

  #mvlist { max-height:420px; overflow-y:auto; font-size:12.5px; scrollbar-width:thin;
            scrollbar-color:var(--panel-edge2) transparent; }
  #mvlist .mv { padding:4px 8px; border-radius:8px; cursor:pointer; display:flex; gap:8px;
                align-items:baseline; line-height:1.55; }
  #mvlist .mv:nth-child(odd) { background:rgba(255,255,255,.03); }
  #mvlist .mv:hover { background:rgba(232,201,135,.08); }
  #mvlist .mv.cur { background:rgba(198,145,62,.2); outline:1px solid rgba(198,145,62,.4); }
  #mvlist .no { color:var(--text-faint); min-width:30px; text-align:right;
                font-variant-numeric:tabular-nums; }
  #mvlist .nm { font-weight:700; margin-right:2px; }
  #mvlist .pth { font-variant-numeric:tabular-nums; letter-spacing:.03em; }
  #mvlist .tg { margin-left:5px; }
  .tg-eat { color:#f0a08e; } .tg-die { color:#9a927e; } .tg-mine { color:#9dc98a; }
  .tg-flag { color:var(--gold-soft); font-weight:800; } .tg-rail { color:var(--text-faint); }
  #mvlist .meta { margin-left:auto; color:var(--text-faint); font-size:11px; text-align:right; }

  .tray { display:flex; flex-wrap:wrap; gap:4px; min-height:26px; }
  .tray .dead-tile { font-size:12px; padding:2px 8px 2px 7px; border-radius:5px; color:#cfc7b2;
           font-family:"KaiTi","STKaiti",serif; font-weight:700; opacity:.8;
           text-decoration:line-through; border:1px solid; background:rgba(0,0,0,.18); }
  .tray.red  .dead-tile { border-color:#7a4038; color:#e8a89b; }
  .tray.blue .dead-tile { border-color:#3a5378; color:#a5bfe8; }
  .trays { display:flex; gap:14px; }
  .trays .half { flex:1; }

  footer { padding:2px 26px 20px; color:var(--text-faint); font-size:11.5px; line-height:1.8; }
  footer kbd { background:#363d2f; border:1px solid var(--panel-edge2); border-radius:4px;
               padding:0 5px; font-size:11px; color:var(--text-dim); }
</style>
</head>
<body>
<header>
  <div class="ttl">
    <div class="seal kai">軍</div>
    <div>
      <h1>__TITLE__</h1>
      <small class="kai">陆 战 棋 · 对 局 回 放</small>
    </div>
  </div>
  <span class="res" id="resbar"></span>
</header>
<div class="wrap">
  <div class="boardcol">
    <div class="board" id="board">
      <svg class="base" id="lines"></svg>
      <svg class="arrow" id="arrowLayer" width="388" height="836" viewBox="0 0 388 836"></svg>
    </div>
  </div>
  <div class="side">
    <div class="panel">
      <h2>对 局 控 制</h2>
      <div class="ctrl">
        <button id="bFirst" title="开局 (Home)">⏮</button>
        <button id="bPrev" title="上一手 (←)">◀</button>
        <button id="bPlay" class="primary" title="播放/暂停 (空格)">▶ 播放</button>
        <button id="bNext" title="下一手 (→)">▶</button>
        <button id="bLast" title="终局 (End)">⏭</button>
        <span class="spd">速度 <select id="speed"><option value="1400">慢</option><option value="800" selected>中</option><option value="400">快</option><option value="150">极快</option></select></span>
      </div>
      <div class="seekrow">
        <input type="range" id="seek" min="0" value="0">
        <span class="ply" id="plylab">0 / 0</span>
      </div>
      <div id="curdesc"></div>
    </div>
    <div class="panel">
      <h2>着 法 记 录</h2>
      <div id="mvlist"></div>
    </div>
    <div class="panel">
      <div class="trays">
        <div class="half"><h2>蓝 方 战 损</h2><div class="tray blue" id="trayB"></div></div>
        <div class="half"><h2>红 方 战 损</h2><div class="tray red" id="trayR"></div></div>
      </div>
    </div>
  </div>
</div>
<footer>
  <kbd>←</kbd> <kbd>→</kbd> 逐手 · <kbd>空格</kbd> 播放/暂停 · <kbd>Home</kbd>/<kbd>End</kbd> 开局/终局 · 点击着法跳转<br>
  回放由 junqi.py 裁判引擎自动生成 (行营庇护 / 铁路滑行 / 工兵转弯 / 地雷炸弹 / 夺旗与困毙均经规则自检) ·
  棋盘视觉参考开源项目 phdsx/junqi · BitByBitG/siguo-junqi · samuelyuan/online-junqi
</footer>
<script>
const DATA = __REPLAY_DATA__;
const CS = 64, PAD = 34;
const SW = 388, SH = 836;                 /* 纸面尺寸 = 5*64+2*34, 12*64+2*34 */
const FRONT_Y = PAD + 6 * CS;             /* 前线: 第6/7行之间 */
const COLS = "abcde";
const board = document.getElementById("board");
const state = new Map();   /* id -> {el, r, c, alive, name, side} */

const cx = c => PAD + c * CS + CS / 2;
const cy = r => PAD + r * CS + CS / 2;
const P  = "#f4f2e5";                     /* 纸色, 用于铁路枕木叠画 */

function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;"); }

function build() {
  let s = `<defs>
    <marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5"
            orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#c6913e"/></marker>
  </defs>
  <style>
    .frame{fill:none;stroke:#5a4a32;stroke-opacity:.5;stroke-width:1.2}
    .frame2{fill:none;stroke:#5a4a32;stroke-opacity:.28;stroke-width:1}
    .rd{stroke:#a4ae93;stroke-width:1.7}
    .dg{stroke:#a4ae93;stroke-width:1.2;stroke-dasharray:4 3.5}
    .rl{stroke:#65785d;stroke-width:6;stroke-linecap:round}
    .tie{stroke:${P};stroke-width:2.6;stroke-dasharray:2.2 6.2;stroke-linecap:butt}
    .mtn{fill:none;stroke:#9aa68c;stroke-width:1.5;stroke-linejoin:round}
    .mtnfill{fill:#edeadb;stroke:none}
    .sz{font:11px "KaiTi","STKaiti",serif;fill:#8d937b;letter-spacing:.45em}
    .camp{fill:#e3ebd5;stroke:#9dac83;stroke-width:1.4}
    .camptx{font:11px "KaiTi","STKaiti",serif;fill:#8a937b;text-anchor:middle;letter-spacing:.2em}
    .hq{fill:#e7e2cb;stroke:#9b9879;stroke-width:1.3}
    .hqtx{font:10px "KaiTi","STKaiti",serif;fill:#a49a76;text-anchor:middle;letter-spacing:.15em}
    .wm{font:44px "KaiTi","STKaiti",serif;fill:#65785d;fill-opacity:.08;text-anchor:middle;letter-spacing:.6em}
    .rowlab{font:11px "Segoe UI",sans-serif;fill:#8d8371;text-anchor:middle}
    .collab{font:11px "Segoe UI",sans-serif;fill:#8d8371;text-anchor:middle}
  </style>`;
  const line = (a, b, cls) =>
    `<line class="${cls}" x1="${cx(a[1])}" y1="${cy(a[0])}" x2="${cx(b[1])}" y2="${cy(b[0])}"/>`;

  /* 印刷内框 (双线) */
  s += `<rect class="frame"  x="8"  y="8"  width="${SW - 16}" height="${SH - 16}" rx="6"/>`;
  s += `<rect class="frame2" x="13" y="13" width="${SW - 26}" height="${SH - 26}" rx="4"/>`;

  /* 半场中央楷体水印 */
  s += `<text class="wm" x="${cx(2)}" y="${cy(2) + 16}">陆战棋</text>`;
  s += `<text class="wm" x="${cx(2)}" y="${cy(9) + 16}">陆战棋</text>`;

  /* 公路在下层 */
  for (const [a, b] of DATA.geometry.road) s += line(a, b, "rd");

  /* 山界群山 (横贯前线, 铁路桥后续叠画其上) */
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

  /* 行营斜线 */
  for (const [a, b] of DATA.geometry.diag) s += line(a, b, "dg");

  /* 铁路: 粗底 + 纸色虚线枕木 → 双轨 */
  for (const [a, b] of DATA.geometry.rail) { s += line(a, b, "rl"); s += line(a, b, "tie"); }

  /* 行营圆 + 大本营双框 */
  for (const [r, c] of DATA.geometry.camps)
    s += `<circle class="camp" cx="${cx(c)}" cy="${cy(r)}" r="22"/>` +
         `<text class="camptx" x="${cx(c)}" y="${cy(r) + 4}">行营</text>`;
  for (const [r, c] of DATA.geometry.hq)
    s += `<rect class="hq" x="${cx(c) - 28}" y="${cy(r) - 18}" width="56" height="36" rx="8"/>` +
         `<rect x="${cx(c) - 24.5}" y="${cy(r) - 14.5}" width="49" height="29" rx="6" ` +
         `fill="none" stroke="#9b9879" stroke-width="1" stroke-opacity=".6"/>` +
         `<text class="hqtx" x="${cx(c)}" y="${cy(r) + 3.5}">大本营</text>`;

  /* 行列坐标 */
  for (let r = 0; r < 12; r++)
    s += `<text class="rowlab" x="22" y="${cy(r) + 4}">${r + 1}</text>` +
         `<text class="rowlab" x="${SW - 22}" y="${cy(r) + 4}">${r + 1}</text>`;
  for (let c = 0; c < 5; c++)
    s += `<text class="collab" x="${cx(c)}" y="24">${COLS[c]}</text>`;

  document.getElementById("lines").setAttribute("width", SW);
  document.getElementById("lines").setAttribute("height", SH);
  document.getElementById("lines").setAttribute("viewBox", `0 0 ${SW} ${SH}`);
  document.getElementById("lines").innerHTML = s;

  for (const p of DATA.pieces) {
    const el = document.createElement("div");
    el.className = "piece " + p.side;
    el.textContent = p.name;
    board.appendChild(el);
    const st = { el, r: p.r, c: p.c, alive: true, name: p.name, side: p.side };
    state.set(p.id, st);
    place(st);
  }
  const res = DATA.result;
  const bar = document.getElementById("resbar");
  if (res.winner === "draw") bar.innerHTML = "⚖ 平局 · " + esc(res.reason) + ` · ${res.plies}手 ${res.seconds}s`;
  else bar.innerHTML = `<b class="${res.winner === "red" ? "red" : "blue"}">${res.winner === "red" ? "红方" : "蓝方"}(${esc(res.winner)})胜</b> · ${esc(res.reason)} · ${res.plies}手 ${res.seconds}s`;
}

function place(st) {
  st.el.style.left = (cx(st.c) - 29) + "px";
  st.el.style.top = (cy(st.r) - 19) + "px";
  st.el.classList.toggle("dead", !st.alive);
}

function drawArrow(m) {   /* 当前手走子箭头 (仅铁路等远距离移动; 单步由金色高亮环标注) */
  const g = document.getElementById("arrowLayer");
  if (!m) { g.innerHTML = ""; return; }
  const [fr, fc] = m.fr, [tr, tc] = m.to;
  const x1 = cx(fc), y1 = cy(fr), x2 = cx(tc), y2 = cy(tr);
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  if (len < CS * 1.6) { g.innerHTML = ""; return; }
  const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
  const o = 30;                                   /* 端点让开棋子 */
  const ax = x1 + ux * o, ay = y1 + uy * o, bx = x2 - ux * (o - 6), by = y2 - uy * (o - 6);
  g.innerHTML = `<line x1="${ax}" y1="${ay}" x2="${bx}" y2="${by}" stroke="#c6913e"
      stroke-width="3.5" stroke-dasharray="7 5.5" stroke-linecap="round"
      marker-end="url(#ah)" opacity=".92" style="animation:arrowin .3s ease"/>`;
}

function drawGraves(m) {  /* 当前手阵亡标记; 夺旗用金徽记 */
  document.querySelectorAll(".grave").forEach(e => e.remove());
  if (!m) return;
  for (const gid of m.gone) {
    const st = state.get(gid);
    const d = document.createElement("div");
    d.className = "grave" + (m.desc.includes("夺旗") ? " flag" : "");
    d.textContent = m.desc.includes("夺旗") ? "旗" : "✕";
    d.style.left = (cx(st.c) - 11) + "px";
    d.style.top = (cy(st.r) - 26) + "px";
    board.appendChild(d);
  }
}

const TAG_RULES = [
  ["夺旗", "tg-flag"], ["吃", "tg-eat"], ["挖雷", "tg-mine"],
  ["阵亡", "tg-die"], ["触雷", "tg-die"], ["同尽", "tg-die"], ["互亡", "tg-die"],
];
function tagSpan(tag) {
  if (!tag) return "";
  const hit = TAG_RULES.find(([k]) => tag.includes(k));
  return `<span class="tg ${hit ? hit[1] : "tg-die"}">${esc(tag)}</span>`;
}
function parseDesc(d) {
  const m = d.match(/^(\S+)\s+(\S+→\S+)(?:\s+(.*))?$/) || [null, "", d, ""];
  return { name: m[1], path: m[2], tag: m[3] || "" };
}

function renderTo(k) {   /* 重演到第 k 手 (0 = 初始布局) */
  for (const [id, st] of state) {
    const p = DATA.pieces[id];
    st.r = p.r; st.c = p.c; st.alive = true;
  }
  for (let i = 0; i < k; i++) {
    const m = DATA.moves[i];
    const st = state.get(m.pid);
    st.r = m.to[0]; st.c = m.to[1];
    for (const gid of m.gone) state.get(gid).alive = false;
  }
  for (const [, st] of state) place(st);
  document.querySelectorAll(".piece.moving").forEach(e => e.classList.remove("moving"));
  const cur = k > 0 ? DATA.moves[k - 1] : null;
  if (cur) state.get(cur.pid).el.classList.add("moving");
  drawArrow(cur);
  drawGraves(cur);

  /* 面板 */
  const list = document.getElementById("mvlist");
  [...list.children].forEach((d, i) => d.classList.toggle("cur", i === k - 1));
  if (k > 0) list.children[k - 1].scrollIntoView({ block: "nearest" });
  document.getElementById("plylab").textContent = k + " / " + DATA.moves.length;
  document.getElementById("seek").value = k;
  const cd = document.getElementById("curdesc");
  if (!cur) cd.innerHTML = `<span style="color:#a8a290">初始布局 — 双方 25 子随机布阵完毕</span>`;
  else {
    const p = parseDesc(cur.desc);
    cd.innerHTML = `<span class="chip ${cur.side}">${cur.side === "red" ? "红" : "蓝"}</span>` +
      `<b>${esc(p.name)}</b><span>${esc(p.path)}</span>` + tagSpan(p.tag);
  }
  /* 战损 */
  const goneB = [], goneR = [];
  for (let i = 0; i < k; i++) for (const gid of DATA.moves[i].gone) {
    const st = state.get(gid);
    (st.side === "red" ? goneR : goneB).push(st.name);
  }
  const cnt = arr => { const m = {}; for (const n of arr) m[n] = (m[n] || 0) + 1;
    return Object.entries(m).map(([n, c]) => c > 1 ? `${n}×${c}` : n); };
  const tile = n => `<span class="dead-tile">${n}</span>`;
  document.getElementById("trayB").innerHTML = cnt(goneB).map(tile).join("") || "<i style='color:#6f685a;font-size:12px'>无</i>";
  document.getElementById("trayR").innerHTML = cnt(goneR).map(tile).join("") || "<i style='color:#6f685a;font-size:12px'>无</i>";
}

let cur = 0, timer = null;
function step(d) { const k = Math.min(DATA.moves.length, Math.max(0, cur + d)); if (k !== cur) { cur = k; renderTo(cur); } }
function goto(k) { cur = Math.min(DATA.moves.length, Math.max(0, k)); renderTo(cur); }
function play() {
  if (timer) { stop(); return; }
  if (cur >= DATA.moves.length) goto(0);
  document.getElementById("bPlay").textContent = "⏸ 暂停";
  timer = setInterval(() => {
    if (cur >= DATA.moves.length) { stop(); return; }
    step(1);
  }, parseInt(document.getElementById("speed").value, 10));
}
function stop() { clearInterval(timer); timer = null; document.getElementById("bPlay").textContent = "▶ 播放"; }

document.getElementById("bFirst").onclick = () => { stop(); goto(0); };
document.getElementById("bLast").onclick = () => { stop(); goto(DATA.moves.length); };
document.getElementById("bPrev").onclick = () => { stop(); step(-1); };
document.getElementById("bNext").onclick = () => { stop(); step(1); };
document.getElementById("bPlay").onclick = play;
document.getElementById("speed").onchange = () => { if (timer) { stop(); play(); } };
document.getElementById("seek").oninput = e => { stop(); goto(parseInt(e.target.value, 10)); };
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") { stop(); step(-1); e.preventDefault(); }
  else if (e.key === "ArrowRight") { stop(); step(1); e.preventDefault(); }
  else if (e.key === " ") { play(); e.preventDefault(); }
  else if (e.key === "Home") { stop(); goto(0); e.preventDefault(); }
  else if (e.key === "End") { stop(); goto(DATA.moves.length); e.preventDefault(); }
});
const style = document.createElement("style");
style.textContent = "@keyframes arrowin { from { opacity:0; } }";
document.head.appendChild(style);

build();
const list = document.getElementById("mvlist");
DATA.moves.forEach((m, i) => {
  const p = parseDesc(m.desc);
  const d = document.createElement("div");
  d.className = "mv";
  d.innerHTML = `<span class="no">${i + 1}.</span>` +
    `<span class="chip ${m.side}">${m.side === "red" ? "红" : "蓝"}</span>` +
    `<span><span class="nm">${esc(p.name)}</span><span class="pth">${esc(p.path)}</span>${tagSpan(p.tag)}</span>` +
    (m.info ? `<span class="meta">${esc(m.info)}</span>` : "");
  d.title = m.desc + (m.info ? " · " + m.info : "");
  d.onclick = () => { stop(); goto(i + 1); };
  list.appendChild(d);
});
renderTo(0);
</script>
</body>
</html>
"""


def render_replay_html(replay):
    return (REPLAY_TEMPLATE
            .replace("__TITLE__", replay["title"])
            .replace("__REPLAY_DATA__", json.dumps(replay, ensure_ascii=False)))


# ---------------------------------------------------------------- 规则自检

def selftest():
    # 1) 工兵铁路转弯可达; 不可越子
    g = Game({}, {}, check=False)
    g.cells.clear()
    g.cells[(5, 0)] = ("red", "工兵")
    g.cells[(5, 4)] = ("blue", "排长")
    dst = {m[1] for m in g.legal_moves("red")}
    assert (6, 1) in dst and (5, 3) in dst, "工兵应能沿铁路转弯(过桥/沿前线)"
    assert (4, 4) not in dst and (7, 1) not in dst, "工兵不可越子, 也不可到达非铁路点"
    # 2) 普通子铁路直线滑行, 不可转弯, 一步过桥
    g2 = Game({}, {}, check=False)
    g2.cells.clear()
    g2.cells[(5, 0)] = ("red", "排长")
    dst2 = {m[1] for m in g2.legal_moves("red")}
    assert (5, 4) in dst2 and (4, 0) in dst2 and (1, 0) in dst2, "普通子应能沿铁路直线滑行"
    assert (6, 1) not in dst2 and (4, 1) not in dst2, "普通子不能在铁路上转弯"
    assert (6, 0) not in dst2, "山界处a列不通"
    # 3) 行营保护
    g3 = Game({}, {}, check=False)
    g3.cells.clear()
    g3.cells[(10, 1)] = ("blue", "司令")     # 红方行营(显示行11,b)
    g3.cells[(11, 1)] = ("red", "司令")
    assert (10, 1) not in {m[1] for m in g3.legal_moves("red")}, "行营内的敌子不可被攻击"
    # 4) 地雷: 司令亡/工兵挖/炸弹同尽
    g4 = Game({}, {}, check=False)
    g4.cells.clear()
    g4.cells[(5, 2)] = ("blue", "地雷")
    g4.cells[(4, 2)] = ("red", "司令")
    undo, victim = g4.make(((4, 2), (5, 2)))
    assert g4.cells.get((5, 2)) == ("blue", "地雷"), "司令碰雷阵亡, 雷仍在"
    g4.unmake(undo)
    g4.cells[(4, 2)] = ("red", "工兵")
    undo, victim = g4.make(((4, 2), (5, 2)))
    assert g4.cells.get((5, 2)) == ("red", "工兵"), "工兵挖雷后占据"
    g4.unmake(undo)
    g4.cells[(4, 2)] = ("red", "炸弹")
    undo, victim = g4.make(((4, 2), (5, 2)))
    assert g4.cells.get((5, 2)) is None, "炸弹碰雷同归于尽"
    g4.unmake(undo)
    # 5) 同级互亡 / 低打高
    g5 = Game({}, {}, check=False)
    g5.cells.clear()
    g5.cells[(7, 2)] = ("blue", "团长")
    g5.cells[(8, 2)] = ("red", "团长")
    undo, _ = g5.make(((8, 2), (7, 2)))
    assert g5.cells.get((7, 2)) is None, "同级互亡"
    g5.unmake(undo)
    g5.cells[(8, 2)] = ("red", "营长")
    undo, _ = g5.make(((8, 2), (7, 2)))
    assert g5.cells.get((7, 2)) == ("blue", "团长"), "低打高, 攻方亡守方留"
    g5.unmake(undo)
    g5.cells[(8, 2)] = ("red", "炸弹")
    _, desc5 = g5.apply(((8, 2), (7, 2)), "red")
    assert "炸弹同尽" in desc5, "炸弹攻击普通子应记为炸弹同尽 (而非同级互亡)"
    # 6) 夺旗即胜
    g6 = Game({}, {}, check=False)
    g6.cells.clear()
    g6.cells[(0, 1)] = ("blue", "军旗")
    g6.cells[(1, 1)] = ("red", "连长")
    g6.apply(((1, 1), (0, 1)), "red")
    assert g6.winner == "red", "夺旗即胜"
    # 7) 大本营内的棋子不能动
    g7 = Game({}, {}, check=False)
    g7.cells.clear()
    g7.cells[(11, 1)] = ("red", "司令")      # 红方大本营(显示行12,b)
    assert g7.legal_moves("red") == [], "大本营内的棋子不能再动"
    print("SELFTEST PASS: 铁路滑行/工兵转弯/行营保护/地雷与炸弹/同级互亡/夺旗胜/大本营冻结/山界 全部符合规则")


# ---------------------------------------------------------------- 主程序

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--url", default="http://127.0.0.1:8090", help="JEV 决策服务")
    ap.add_argument("--ai-time", type=float, default=2.0, help="AI 每步搜索时限(秒)")
    ap.add_argument("--ai-depth", type=int, default=4, help="AI 最大搜索深度")
    ap.add_argument("--max-plies", type=int, default=200, help="每局手数上限(达上限按子力判定)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).parent / "games"))
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    results = [play_game(i, a, log) for i in range(a.games)]
    log("\n=== 总战绩 ===")
    tally = {}
    for r in results:
        tally[r["winner"]] = tally.get(r["winner"], 0) + 1
        log(f"第{r['game']}局: {r['winner']}胜({r['reason']}, {r['plies']}手, {r['seconds']}s)")
    log("比分: " + "  ".join(f"{k} {v}" for k, v in tally.items()))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tr = outdir / f"game_{stamp}.txt"
    tr.write_text("\n".join(lines), encoding="utf-8")
    (outdir / f"game_{stamp}.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "replay"} for r in results],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    for r in results:
        hp = outdir / f"game_{stamp}_g{r['game']}.html"
        hp.write_text(render_replay_html(r["replay"]), encoding="utf-8")
        log(f"可视化回放: {hp}")
    print(f"棋谱已保存: {tr}")


if __name__ == "__main__":
    main()
