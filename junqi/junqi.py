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
                       else "炸弹同尽" if d[1] == "炸弹" else "同级互亡")
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
  :root { --cs:64px; --red:#c23b2e; --blue:#2456a4; --board:#f3e9d4; --line:#8a7a5c; --ink:#2b2116; }
  * { box-sizing:border-box; }
  body { margin:0; background:#232019; color:#e8e2d5; font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif; }
  header { display:flex; align-items:baseline; gap:16px; padding:14px 22px 8px; flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; letter-spacing:.04em; }
  header .res { font-size:14px; padding:3px 12px; border-radius:99px; background:#3a3428; }
  header .res b.red { color:#ff8f7e; } header .res b.blue { color:#8fb7ff; }
  .wrap { display:flex; gap:20px; padding:8px 22px 26px; align-items:flex-start; flex-wrap:wrap; }
  .boardcol { position:relative; }
  .board { position:relative; background:var(--board); border-radius:14px; padding:26px;
           box-shadow:0 12px 40px rgba(0,0,0,.5), inset 0 0 0 2px #b49f77, inset 0 0 0 6px var(--board), inset 0 0 0 7px #cdb88f; }
  svg.lines { position:absolute; left:26px; top:26px; pointer-events:none; }
  .cell { position:absolute; width:var(--cs); height:var(--cs); display:flex; align-items:center; justify-content:center; }
  .piece { position:absolute; width:54px; height:32px; border-radius:6px; display:flex; align-items:center; justify-content:center;
           font-size:15px; font-weight:700; letter-spacing:1px; cursor:default; transition:left .28s ease, top .28s ease, opacity .28s;
           border:1.6px solid; box-shadow:0 2px 5px rgba(43,33,22,.35); background:#faf4e4; z-index:3; }
  .piece.red { color:var(--red); border-color:var(--red); }
  .piece.blue { color:var(--blue); border-color:var(--blue); }
  .piece.dead { opacity:0; pointer-events:none; }
  .piece.moving { box-shadow:0 0 0 3px #d9a441, 0 4px 10px rgba(0,0,0,.35); z-index:5; }
  .camp-mark { position:absolute; width:46px; height:46px; border:2px dashed #b3a179; border-radius:50%; z-index:1; }
  .hq-mark { position:absolute; font-size:10px; color:#a4906a; text-align:center; line-height:1.25; z-index:1; }
  .rowlab, .collab { position:absolute; font-size:11px; color:#8d8371; z-index:0; }
  .side { flex:1 1 300px; min-width:300px; max-width:430px; }
  .panel { background:#2e2a21; border:1px solid #45402f; border-radius:12px; padding:12px 14px; margin-bottom:14px; }
  .panel h2 { margin:0 0 8px; font-size:13px; color:#b3aa93; font-weight:600; letter-spacing:.08em; }
  .ctrl { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .ctrl button { background:#3d382a; color:#e8e2d5; border:1px solid #5a5340; border-radius:8px; padding:6px 13px; font:inherit; cursor:pointer; }
  .ctrl button:hover { border-color:#d9a441; color:#ffd98a; }
  .ctrl button.primary { background:#d9a441; border-color:#d9a441; color:#2b2116; font-weight:700; }
  .ctrl input[type=range] { flex:1; accent-color:#d9a441; }
  .ctrl .spd { font-size:12px; color:#b3aa93; }
  #mvlist { max-height:380px; overflow-y:auto; font-size:12.5px; }
  #mvlist div { padding:4px 8px; border-radius:6px; cursor:pointer; display:flex; gap:8px; line-height:1.5; }
  #mvlist div:nth-child(odd) { background:#332e23; }
  #mvlist div.cur { background:#5a4a22; color:#ffe1a1; }
  #mvlist .no { color:#8d8371; min-width:30px; text-align:right; }
  #mvlist .who { min-width:16px; } 
  .tag-red { color:#ff9d8c; } .tag-blue { color:#9cc0ff; }
  .tray { display:flex; flex-wrap:wrap; gap:4px; min-height:26px; }
  .tray span { font-size:12px; padding:2px 8px; border-radius:5px; border:1px solid #565040; color:#cfc7b2; text-decoration:line-through; opacity:.75; }
  .tray.red span { border-color:#7a4038; } .tray.blue span { border-color:#3a5378; }
  footer { padding:0 22px 20px; color:#7d745f; font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>军棋回放 · __TITLE__</h1>
  <span class="res" id="resbar"></span>
</header>
<div class="wrap">
  <div class="boardcol">
    <div class="board" id="board">
      <svg class="lines" id="lines"></svg>
    </div>
  </div>
  <div class="side">
    <div class="panel">
      <h2>控制</h2>
      <div class="ctrl">
        <button id="bFirst" title="开局">⏮</button>
        <button id="bPrev" title="上一手 (←)">◀</button>
        <button id="bPlay" class="primary" title="播放/暂停 (空格)">▶ 播放</button>
        <button id="bNext" title="下一手 (→)">▶</button>
        <button id="bLast" title="终局">⏭</button>
        <span class="spd">速度 <select id="speed"><option value="1400">慢</option><option value="800" selected>中</option><option value="400">快</option><option value="150">极快</option></select></span>
      </div>
      <div class="ctrl" style="margin-top:8px">
        <input type="range" id="seek" min="0" value="0" style="width:100%">
        <span id="plylab" style="font-size:13px;min-width:70px;text-align:right">0 / 0</span>
      </div>
      <div id="curdesc" style="margin-top:6px;font-size:13px;color:#ffe1a1;min-height:20px"></div>
    </div>
    <div class="panel">
      <h2>着法记录</h2>
      <div id="mvlist"></div>
    </div>
    <div class="panel">
      <h2>蓝方战损</h2><div class="tray blue" id="trayB"></div>
      <h2 style="margin-top:10px">红方战损</h2><div class="tray red" id="trayR"></div>
    </div>
  </div>
</div>
<footer>由 junqi.py 自动生成 · 裁判规则引擎: 行营庇护/铁路滑行/工兵转弯/地雷炸弹/夺旗与困毙判定均经规则自检 · ←/→ 翻着法, 空格播放</footer>
<script>
const DATA = __REPLAY_DATA__;
const CS = 64, PAD = 26;
const board = document.getElementById("board");
const COLS = "abcde";
const state = new Map();   // id -> {piece element, r, c, alive, name, side}

function xy(r, c) { return [PAD + c * CS + CS / 2, PAD + r * CS + CS / 2]; }

function build() {
  board.style.width  = PAD * 2 + 5 * CS + "px";
  board.style.height = PAD * 2 + 12 * CS + "px";
  const svg = document.getElementById("lines");
  const W = 5 * CS, H = 12 * CS;
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  let s = "";
  const line = (a, b, cls) => {
    const [x1, y1] = xy(a[0], a[1]), [x2, y2] = xy(b[0], b[1]);
    return `<line class="${cls}" x1="${x1 - PAD}" y1="${y1 - PAD}" x2="${x2 - PAD}" y2="${y2 - PAD}"/>`;
  };
  s += "<style>.rd{stroke:#a5946f;stroke-width:2}.rl{stroke:#6f5f43;stroke-width:6;stroke-linecap:round}.rli{stroke:#f3e9d4;stroke-width:3;stroke-linecap:round}.dg{stroke:#b3a179;stroke-width:1.6;stroke-dasharray:5 4}</style>";
  for (const [a, b] of DATA.geometry.road) s += line(a, b, "rd");
  for (const [a, b] of DATA.geometry.rail) s += line(a, b, "rl") + line(a, b, "rli");
  for (const [a, b] of DATA.geometry.diag) s += line(a, b, "dg");
  svg.innerHTML = s;
  for (const [r, c] of DATA.geometry.camps) {
    const d = document.createElement("div");
    d.className = "camp-mark";
    const [x, y] = xy(r, c);
    d.style.left = (x - PAD - 23) + "px"; d.style.top = (y - PAD - 23) + "px";
    board.appendChild(d);
  }
  for (const [r, c] of DATA.geometry.hq) {
    const d = document.createElement("div");
    d.className = "hq-mark";
    const [x, y] = xy(r, c);
    d.style.left = (x - PAD - 24) + "px"; d.style.top = (y - PAD + 14) + "px";
    d.style.width = "48px";
    d.textContent = "大本营";
    board.appendChild(d);
  }
  for (let r = 0; r < 12; r++) {
    const d = document.createElement("div");
    d.className = "rowlab";
    d.style.left = "4px"; d.style.top = (PAD + r * CS + CS / 2 - 7) + "px";
    d.textContent = String(r + 1);
    board.appendChild(d);
  }
  for (let c = 0; c < 5; c++) {
    const d = document.createElement("div");
    d.className = "collab";
    d.style.left = (PAD + c * CS + CS / 2 - 4) + "px"; d.style.top = "6px";
    d.textContent = COLS[c];
    board.appendChild(d);
  }
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
  if (res.winner === "draw") bar.innerHTML = "平局 · " + res.reason;
  else bar.innerHTML = `<b class="${res.winner === "red" ? "red" : "blue"}">${res.winner === "red" ? "红方" : "蓝方"}(${res.winner})胜</b> · ${res.reason} · ${res.plies}手 ${res.seconds}s`;
}

function place(st) {
  const [x, y] = xy(st.r, st.c);
  st.el.style.left = (x - PAD - 27) + "px";
  st.el.style.top = (y - PAD - 16) + "px";
  st.el.classList.toggle("dead", !st.alive);
}

function renderTo(k) {   // 重演到第 k 手 (0 = 初始布局)
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
  if (k > 0) {
    const m = DATA.moves[k - 1];
    state.get(m.pid).el.classList.add("moving");
  }
  // 面板
  const list = document.getElementById("mvlist");
  [...list.children].forEach((d, i) => d.classList.toggle("cur", i === k - 1));
  if (k > 0) list.children[k - 1].scrollIntoView({ block: "nearest" });
  document.getElementById("plylab").textContent = k + " / " + DATA.moves.length;
  document.getElementById("seek").value = k;
  document.getElementById("curdesc").textContent =
    k === 0 ? "初始布局" :
    (DATA.moves[k - 1].side === "red" ? "红 " : "蓝 ") + DATA.moves[k - 1].desc +
    (DATA.moves[k - 1].info ? " · " + DATA.moves[k - 1].info : "");
  // 战损
  const goneB = [], goneR = [];
  for (let i = 0; i < k; i++) for (const gid of DATA.moves[i].gone) {
    const st = state.get(gid);
    (st.side === "red" ? goneR : goneB).push(st.name);
  }
  const cnt = arr => { const m = {}; for (const n of arr) m[n] = (m[n] || 0) + 1; return Object.entries(m).map(([n, c]) => c > 1 ? `${n}×${c}` : n); };
  document.getElementById("trayB").innerHTML = cnt(goneB).map(n => `<span>${n}</span>`).join("") || "<i style='color:#6f685a;font-size:12px'>无</i>";
  document.getElementById("trayR").innerHTML = cnt(goneR).map(n => `<span>${n}</span>`).join("") || "<i style='color:#6f685a;font-size:12px'>无</i>";
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
});

build();
const list = document.getElementById("mvlist");
DATA.moves.forEach((m, i) => {
  const d = document.createElement("div");
  d.innerHTML = `<span class="no">${i + 1}.</span><span class="who ${m.side === "red" ? "tag-red" : "tag-blue"}">${m.side === "red" ? "红" : "蓝"}</span><span>${m.desc}</span>`;
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
