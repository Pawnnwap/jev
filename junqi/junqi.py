#!/usr/bin/env python3
"""军棋(二人陆战棋·全暗棋)环境: JEV(决策模型) vs 规则评估+α-β树搜索AI。

规则合规: 所有走子/吃子/胜负由 Game(裁判) 按《陆战棋》规则统一校验与执行,
双方只能从 legal_moves() 中选择, 物理上无法违规。
- 棋盘 5列(a-e)×12行(1-12); 行1-6=蓝方半场, 行7-12=红方; 行6-7间为前线(山界), 仅 b/c/d 三桥相通。
- 每方 25 子: 司令1 军长1 师长2 旅长2 团长2 营长2 连长3 排长3 工兵3 地雷3 炸弹2 军旗1。
- 大本营(行1/12 的 b,d): 出于公平只布【军旗+排长】两子; 入营即冻结(视同占位符, 未扛旗不能动),
  空营封格 — 其他棋子不得再次进入。
- 行营(每方5个, 与周围8点有斜线): 其中的棋子不可被攻击。
- 公路一步; 铁路(前线两行 + a/e两侧列)直线任意远不可转弯; 工兵铁路内可转弯(不可越子)。
- 吃子: 司令>军长>师长>旅长>团长>营长>连长>排长>工兵, 同级互亡;
  炸弹与任何子同归于尽; 地雷不能动, 只有工兵可挖雷(工兵存活), 炸弹碰雷同尽。
- 胜负: 夺取军旗, 或对方无子可走(困毙); 手数达上限按子力判定(结果中标注"判定")。

全暗棋信息规则(对双方一致生效):
- 只能看见己方棋子身份; 敌子只见位置(棋盘记作 ?), 身份靠战况+排除法推断。
- 吃子结果不泄露敌子身份: 攻击前只知道可能结果集合; 双亡时敌子只可能是
  【同级子 或 炸弹】两种可能(对掉与炸弹等效, 不可分辨)。
- 司令阵亡 → 其方军旗位置公开; 双司令对掉 → 双方军旗同时公开。
- 每方知识库 Belief 只含可观察事实推出的 poss(可能类型集合), JEV state 与
  RULE-AI 的世界采样都只从 belief 出发, 不读敌方真值。

用法:
  python junqi.py --selftest                 # 规则+暗棋自检
  python junqi.py --games 1 --ai-time 2.0    # 对局(第1局红=JEV 蓝=AI, 之后交替换边)
"""
import argparse
import json
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
                # 山界(前线): 第6/7行之间只有 b/c/d 三桥相通, a/e 两列双向都不通
                # (必须双向判定: 只判 r==5 会让 e7→e6 这种"反向越界"漏过去)
                if {r, nr} == {5, 6} and c not in (1, 2, 3):
                    continue
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
    """随机合法布局: 大本营只布【军旗+排长】(公平性约束); 地雷限本方后两排; 炸弹不得入本方第一排(前线行)."""
    cells = [(lr, c) for lr in range(6) for c in range(5)]   # 局部坐标, 0=底线, 5=前线
    rng.shuffle(cells)
    flag_cell = rng.choice([(0, 1), (0, 3)])
    put = {flag_cell: "军旗", (0, 4 - flag_cell[1]): "排长"}   # 另一个大本营坐排长
    back = [p for p in cells if p[0] <= 1 and p not in put]
    for p in rng.sample(back, 3):
        put[p] = "地雷"
    rest = [p for p in cells if p not in put and p[0] != 5]
    for p in rng.sample(rest, 2):
        put[p] = "炸弹"
    rest = [p for p in cells if p not in put]
    pool = [n for n, k in PIECE_COUNTS.items() for _ in range(k)
            if n not in ("军旗", "地雷", "炸弹")]
    pool.remove("排长")                       # 一颗排长已入大本营
    rng.shuffle(pool)
    for p, name in zip(rest, pool):
        put[p] = name
    assert sorted(put.values()) == sorted(n for n, k in PIECE_COUNTS.items() for _ in range(k))
    return {((lr if side == "blue" else 11 - lr), c): name for (lr, c), name in put.items()}


# ---------------------------------------------------------------- 暗棋知识库 (belief)

def name_order(name):
    """棋种排序键(同价值时按字典序, 保证输出与采样确定)。"""
    return (-VALUES.get(name, 0), name)


def poss_text(poss, limit=4):
    poss = sorted(poss, key=name_order)
    return "/".join(poss[:limit]) + ("/…" if len(poss) > limit else "")


def seed_poss(opp, cell):
    """开局时"从对手视角"该敌格内棋子可能是什么棋种 —— 只用布局约束, 不含任何具体布局信息。
    大本营只布军旗+排长; 地雷仅本方后两排; 炸弹不进本方第一排(前线行)。"""
    if cell_kind(*cell) == "hq":
        return {"军旗", "排长"}
    names = {n for n in PIECE_COUNTS if n != "军旗"}
    lr = local_row(cell[0], opp)
    if lr >= 2:
        names.discard("地雷")
    if lr == 5:
        names.discard("炸弹")
    return names


def combat_poss(pre, mine, role, result, deep=False, open_target=False):
    """一次交战观察 → 敌子仍可能的棋种集合。
    role='atk': mine 是己方(已知)攻子, 推敌守子; role='def': mine 是己方(已知)守子, 推敌攻子。
    result: 'win'=守方亡/攻方存活, 'trade'=双亡, 'lose'=攻方亡/守方存活。
    deep=True 目标落在敌方自己的后两排(二线以内, 雷区); open_target=True 目标在二线以外。
    只做与 pre 的收缩; 若收缩为空(观察与知识矛盾, 正常不会发生)则原样退回 pre, 绝不泄露。"""
    if role == "atk":
        if result == "win":            # 攻方存活 → 敌子比我弱(工兵还能挖雷), 或就是被夺的军旗
            s = {n for n in pre if n in RANKS and RANKS[n] < RANKS[mine]}
            if mine == "工兵":
                s.add("地雷")
            elif mine == "炸弹":
                s = set()             # 炸弹除夺旗外不可能单独存活
            if "军旗" in pre:
                s.add("军旗")
        elif result == "trade":       # 双亡 → 同级, 或敌子是炸弹(对掉与炸弹不可分辨)
            if mine == "炸弹":
                s = set(pre) - {"军旗"}
            else:
                s = ({n for n in pre if n in RANKS and RANKS[n] == RANKS[mine]}
                     | ({"炸弹"} & set(pre)))
        else:                         # 'lose' 攻方亡守方在
            if mine == "司令" and deep:
                # 司令(二线以内唯一能压过一切非雷子的子)阵亡而守方完好 → 只可能是雷或炸弹
                s = {"地雷"} | ({"炸弹"} & set(pre))
            else:
                s = {n for n in pre if n in RANKS and RANKS[n] > RANKS[mine]}
                if mine == "炸弹":
                    s = set()         # 炸弹不可能单独阵亡
                elif mine != "工兵":
                    s.add("地雷")     # 非工兵碰雷: 雷保留, 攻方单亡
    else:
        if result == "win":            # 我的守子亡、敌攻子活 → 敌子更强; 我的雷被挖则只有工兵
            if mine == "地雷":
                s = {"工兵"}
            elif mine == "军旗":
                s = set(pre)           # 夺旗: 任何能动的子
            elif mine == "炸弹":
                s = set()             # 炸弹作守方必同尽
            else:
                s = {n for n in pre if n in RANKS and RANKS[n] > RANKS[mine]}
        elif result == "trade":
            if mine == "地雷":
                s = {"炸弹"}
            elif mine == "军旗":
                s = set()
            elif mine == "炸弹":
                s = set(pre) - {"军旗"}
            else:
                s = {n for n in pre if n in RANKS and RANKS[n] == RANKS[mine]} | ({"炸弹"} & set(pre))
        else:                         # 'lose' 敌攻子亡、我的子还在 → 敌子比我弱
            if mine == "地雷":
                s = set(pre) - {"工兵", "炸弹"}
            elif mine in RANKS:
                s = {n for n in pre if n in RANKS and RANKS[n] < RANKS[mine]}
            else:
                s = set()
            # 注: "吃掉二线以外的子 → 不可能是工兵" 不成立 —— 工兵可以从铁路格向相邻公路格走一步
            # (自检里的 truth∈poss 体检抓到过多个反例), 故这里不做排除, 只靠"铁路拐弯=工兵"正向明牌。
    s = set(s) & set(pre)
    return s if s else set(pre)


class Belief:
    """一方(view)对敌方(opp)棋子的暗棋知识 —— 只记录可观察事实:
        pid   : 敌子稳定身份(随子移动, 不因换格而改变)
        poss  : 每颗活敌子当前仍可能的棋种集合
        moved : 观察到移动过的敌子 pid (地雷/军旗永不动, 移动即可排除)
        dead  : 已阵亡敌子 pid -> 其可能棋种的最终集合
    只由 Game.apply() 更新: make/unmake(搜索)绝不触碰, 因此搜索里要么用世界真值、
    要么用 belief 采样出的世界, 不会把"上帝视角"混进玩家决策。"""

    def __init__(self, view, opp):
        self.view, self.opp = view, opp
        self.poss, self.pid_at, self.moved, self.dead = {}, {}, set(), {}
        self._next = 0

    def _new_pid(self):
        self._next += 1
        return self._next

    def bind(self, cell):
        pid = self.pid_at.get(cell)
        if pid is None:
            pid = self._new_pid()
            self.pid_at[cell] = pid
            self.poss[pid] = seed_poss(self.opp, cell)
        return pid

    def at(self, cell):
        """该敌格内棋子的可能棋种(未登记或已被吃光时退回布局约束)。"""
        pid = self.pid_at.get(cell)
        return set(self.poss[pid]) if pid is not None else seed_poss(self.opp, cell)

    def move(self, fr, to):
        """观察到敌子 fr→to: 身份随子迁移, 并按"能动的子才可能动"排除军旗与地雷。"""
        pid = self.pid_at.pop(fr, None)
        if pid is None:
            return
        self.poss[pid] -= {"军旗", "地雷"}
        self.pid_at[to] = pid
        self.moved.add(pid)

    def kill(self, cell, poss=None):
        """该敌格上的敌子阵亡, 记录其最终可能棋种(供排除法计数)。"""
        pid = self.pid_at.pop(cell, None)
        if pid is None:
            return None
        self.poss[pid] = set(poss) if poss is not None else set(self.poss[pid])
        self.dead[pid] = self.poss[pid]
        return pid

    def prune(self):
        """排除法剪枝(迭代到不动点):
        ①某棋种已全部阵亡(有阵亡子被确证为该种) → 从所有敌子 poss 中剔除;
        ②某棋种"存活名额"已被"poss 收窄到该种"的活子占满 → 其余敌子不可能是该种。
        收缩会置空的集合一律跳过(宁可不剪, 不可凭空造信息)。"""
        for _ in range(4):
            live = list(self.pid_at.values())
            dead_ids = list(self.dead)

            def shrink(s, n):
                if n in s and len(s) > 1:
                    s.discard(n)
                    return True
                return False

            changed = False
            for n, k in PIECE_COUNTS.items():
                dead_n = sum(1 for p in dead_ids if self.dead[p] == {n})
                if dead_n >= k:                                  # ①
                    for p in live:
                        changed |= shrink(self.poss[p], n)
                    for p in dead_ids:
                        changed |= shrink(self.dead[p], n)
                    continue
                claim = [p for p in live if self.poss[p] == {n}]
                if k - dead_n and len(claim) >= k - dead_n:        # ②
                    for p in live:
                        if p not in claim:
                            changed |= shrink(self.poss[p], n)
                # ③ 稀缺棋种归位: 能是该棋种的活子数量 ≤ 剩余个数的下界(已确证死的 + 疑似死的都扣掉)
                #    → 它们只能全是该棋种。用下界而不是上界, 否则会把"疑似死"的棋子算漏, 误判成确定。
                # ③ 稀缺棋种归位: 剩余个数的下界 == 能是该棋种的活子数量 → 这些活子只能全是该棋种
                #    (只有相等才被逼死: 下界 < 候选数时中间还有余地, 不能定)
                lo_left = k - dead_n - sum(1 for p in dead_ids if n in self.dead[p])
                can_be = [p for p in live if n in self.poss[p]]
                if can_be and lo_left == len(can_be):
                    for p in can_be:
                        if self.poss[p] != {n}:
                            self.poss[p] = {n}
                            changed = True
            if not changed:
                break


# ---------------------------------------------------------------- 裁判

class Game:
    def __init__(self, blue_pos, red_pos, check=True):
        cells = {}
        for (r, c), name in blue_pos.items():
            cells[(r, c)] = ("blue", name)
        for (r, c), name in red_pos.items():
            cells[(r, c)] = ("red", name)
        self._setup(cells, check)

    @classmethod
    def from_cells(cls, cells):
        """由完整占位表 {(格): (side, 棋种)} 直接建局 —— 供 belief 采样出的世界使用
        (该世界里"敌子棋种"就是采样落实的值, 世界内部按完整信息搜索)。"""
        g = cls({}, {}, check=False)
        g._setup(dict(cells), check=False)
        return g

    def clone_belief(self):
        """深拷贝一个带知识库的副本: 可以在副本上 apply(推进信念)而不污染真实局面。"""
        g = Game.from_cells(dict(self.cells))
        g.flags_public = dict(self.flags_public)
        g.winner, g.quiet_plies = self.winner, self.quiet_plies
        for s in SIDES:
            src, dst = self.knowledge[s], g.knowledge[s]
            dst.poss = {p: set(v) for p, v in src.poss.items()}
            dst.pid_at = dict(src.pid_at)
            dst.moved = set(src.moved)
            dst.dead = {p: set(v) for p, v in src.dead.items()}
            dst._next = src._next
        return g

    def _setup(self, cells, check=True):
        self.cells = cells
        if check:
            assert len(self.cells) == 50
        self.winner = None
        self.history = []      # (side, move, victim|None, desc, dark{side: 该方暗棋视角描述})
        self.quiet_plies = 0
        # ---- 暗棋状态 ----
        self.flags_public = {"blue": False, "red": False}   # 司令阵亡 → 该方军旗位置公开
        self.hq_origin = {s: {} for s in SIDES}              # 开局时各方大本营里的两子(军旗/排长)
        self.knowledge = {s: Belief(s, "blue" if s == "red" else "red") for s in SIDES}
        for (r, c), (s, name) in self.cells.items():
            self.knowledge["blue" if s == "red" else "red"].bind((r, c))   # 每方只为敌子建身份
            if cell_kind(r, c) == "hq":
                self.hq_origin[s][(r, c)] = name

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
            return cell_kind(*pos) != "hq"   # 大本营封格: 腾空后任何棋子都不得再进入
        if p[0] == side:
            return False
        k = cell_kind(*pos)
        if k == "camp":
            return False                    # 行营内的敌子不可攻击
        if k == "hq" and p[1] != "军旗" and self.flags_public[p[0]]:
            return False                    # 敌旗位置已公开: 只许进军旗营夺旗, 占位营(排长/空)不再进人
        return True

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
        """工兵: 铁路网内任意路径, 不可越子, 可停空点或吃沿途敌子。
        落点只允许"空格"或"可攻击的敌子"(行营内的敌子不可攻击), 不能落到己方子上。"""
        seen, frontier = {pos}, [pos]
        while frontier:
            cur = frontier.pop()
            for nxt in RAIL.get(cur, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if self.cells.get(nxt) is None:
                    frontier.append(nxt)      # 只有空格能继续往前探, 有子即算路障
        out = set()
        for cell in seen:
            if cell == pos:
                continue
            p = self.cells.get(cell)
            if p is None:
                out.add(cell)                # 空点(含空行营)可停
            elif p[0] != side and cell_kind(*cell) != "camp":
                out.add(cell)                # 敌子可吃
        return out

    def make(self, move):
        """执行着法(不校验), 返回 undo; 军旗被夺即定胜负。
        司令阵亡会置位 flags_public(该方军旗位置公开) —— 放在这里而不是 apply,
        是为了让搜索内部模拟的每一手也照样触发"军旗公开"(unmake 会还原)。"""
        (fr, fc), (tr, tc) = move
        a = self.cells[(fr, fc)]
        d = self.cells.get((tr, tc))
        fp = (self.flags_public["blue"], self.flags_public["red"])
        undo = (move, a, d, self.winner, self.quiet_plies, fp)
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
        after = self.cells.get((tr, tc))
        att_alive = after == a
        def_alive = d is not None and after == d
        if an == "司令" and not att_alive:     # 攻方司令阵亡 → 该方军旗公开
            self.flags_public[a[0]] = True
        if dn == "司令" and not def_alive:     # 守方司令阵亡 → 该方军旗公开
            self.flags_public[d[0]] = True
        return undo, dn

    def unmake(self, undo):
        move, a, d, winner, quiet, fp = undo
        (fr, fc), (tr, tc) = move
        self.cells.pop((tr, tc), None)
        self.cells[(fr, fc)] = a
        if d is not None:
            self.cells[(tr, tc)] = d
        self.winner = winner
        self.quiet_plies = quiet
        self.flags_public["blue"], self.flags_public["red"] = fp

    @staticmethod
    def _combat_class(name, dn):
        """攻方 name 攻守方 dn 的结果类别: 夺旗! / 挖雷 / 触雷阵亡 / 吃 / 同尽 / 阵亡。"""
        if dn == "军旗":
            return "夺旗!"
        if dn == "地雷":
            return "挖雷" if name == "工兵" else "触雷阵亡"
        if name == "炸弹" or dn == "炸弹":
            return "同尽"
        if RANKS[name] > RANKS[dn]:
            return "吃"
        if RANKS[name] == RANKS[dn]:
            return "同尽"
        return "阵亡"

    @staticmethod
    def _truth_tag(name, dn, dside):
        """上帝视角的战斗结果标签(棋谱/回放用, 如实写明敌子真名)。"""
        cls = Game._combat_class(name, dn)
        if cls == "夺旗!":
            return " 夺旗!"
        if cls == "挖雷":
            return " 挖雷"
        if cls == "触雷阵亡":
            return " 触雷阵亡"
        if cls == "同尽":
            return " 同尽" if (name == "炸弹" or dn == "炸弹") else f" 与{dn}同尽"
        if cls == "吃":
            return f" 吃{SIDE_CN[dside]}{dn}"
        return f" 阵亡于{dn}"

    def describe(self, move, combat=True, view=None):
        """着法描述。view=None → 裁判/上帝视角(棋谱与回放, 如实预告战斗结果);
        view=side → 该方视角: 敌子身份按 belief 的可能集合预告, 如
        '吃蓝师长'(已收窄到唯一) 或 '敌子未明: 师长(吃)/炸弹(同尽)'。"""
        (fr, fc), (tr, tc) = move
        name = self.cells[(fr, fc)][1]
        d = self.cells.get((tr, tc))
        if d is not None and combat:
            if view is None:
                tag = self._truth_tag(name, d[1], d[0])
            else:
                poss = self.knowledge[view].at((tr, tc))
                if len(poss) == 1:
                    tag = self._truth_tag(name, next(iter(poss)), d[0])
                elif len(poss) <= 4:
                    tag = " 敌子未明: " + "/".join(
                        f"{n}({self._combat_class(name, n)})" for n in sorted(poss, key=name_order))
                else:
                    classes = []
                    for n in sorted(poss, key=name_order):
                        if self._combat_class(name, n) not in classes:
                            classes.append(self._combat_class(name, n))
                    tag = " 敌子未明: " + "/".join(classes)
        elif is_rail(fr, fc) and (tr, tc) in RAIL.get((fr, fc), []) and abs(fr - tr) + abs(fc - tc) > 1:
            tag = " 铁路"
        else:
            tag = ""
        return f"{name} {coord((fr, fc))}→{coord((tr, tc))}{tag}"

    def _dark_tags(self, r, poss, defender, dside, as_defender):
        """已结算战斗的暗棋标签: as_defender=False → 攻方视角(敌子是守子);
        True → 守方视角(我的守子棋种已知, 敌攻子未知)。"""
        if len(poss) == 1:
            n = next(iter(poss))
            if not as_defender:
                if r == "win":
                    return " 夺旗!" if n == "军旗" else f" 吃{SIDE_CN[dside]}{n}"
                return " 同尽" if (r == "trade" and n == "炸弹") else (
                    f" 与{n}同尽" if r == "trade" else f" 阵亡于{n}")
            if r == "win":
                return f" {defender}被{SIDE_CN[dside]}{n}吃掉"
            return f" {defender}与{SIDE_CN[dside]}{n}同尽" if r == "trade" else f" {SIDE_CN[dside]}{n}攻{defender}阵亡"
        txt = poss_text(poss)
        if not as_defender:
            if r == "win":
                return f" 吃{SIDE_CN[dside]}子(敌未明: {txt})"
            return f" 与敌子同尽(敌未明: {txt})" if r == "trade" else f" 阵亡于敌子(敌未明: {txt})"
        if r == "win":
            return f" {defender}被敌子吃掉(敌未明: {txt})"
        return f" {defender}与敌子同尽(敌未明: {txt})" if r == "trade" else f" 敌子攻{defender}阵亡(敌未明: {txt})"

    def apply(self, move, side):
        """裁判入口: 校验合法后执行。返回 (victim|None, 着法描述)。
        描述两版并存: 裁判真相版(棋谱/回放)与按方暗棋版(存进 history, 供 JEV 只看可知信息);
        同时按"看得见的战况"更新双方 belief 与军旗公开状态。"""
        if move not in self.legal_moves(side):
            raise ValueError(f"非法着法 {move} ({SIDE_CN[side]})")
        opp = "blue" if side == "red" else "red"
        (fr, fc), (tr, tc) = move
        a_pre, d_pre = self.cells[(fr, fc)], self.cells.get((tr, tc))
        name, defender = a_pre[1], (d_pre[1] if d_pre else None)
        base = self.describe(move, combat=False)      # 己方视角的基础描述(只有自己的棋名)
        k_me, k_opp = self.knowledge[side], self.knowledge[opp]
        d_pid = k_me.pid_at.get((tr, tc)) if d_pre else None
        a_pid = k_opp.pid_at.get((fr, fc))            # 敌攻子(对守方而言的 pid)
        undo, victim = self.make(move)
        after = self.cells.get((tr, tc))
        att_alive, def_alive = after == a_pre, d_pre is not None and after == d_pre
        # ---- 上帝视角结果标签 ----
        tag = ""
        if d_pre is not None:
            if defender == "军旗":
                tag = " 夺旗!"
            elif after is None:                        # 目标格落空: 双亡
                tag = ("炸弹碰雷同尽" if defender == "地雷"
                       else "炸弹同尽" if "炸弹" in (name, defender) else "同级互亡")
            elif after == d_pre:                       # 守方原位不动: 攻方亡
                tag = "触雷阵亡" if defender == "地雷" else f"攻方阵亡于{defender}"
            else:
                tag = f"吃{SIDE_CN[opp]}{defender}"
        # ---- 司令阵亡 → 该方军旗位置公开(公开信息) ----
        notes = []
        if name == "司令" and not att_alive:
            notes.append(f"{SIDE_CN[side]}司令阵亡, {SIDE_CN[side]}军旗位置公开")
            self.flags_public[side] = True
        if defender == "司令" and not def_alive:
            notes.append(f"{SIDE_CN[opp]}司令阵亡, {SIDE_CN[opp]}军旗位置公开")
            self.flags_public[opp] = True
        note = ("  (" + "; ".join(notes) + ")") if notes else ""
        desc = base + ((" " + tag) if tag else "") + note
        # ---- belief 更新(只看双方都看得见的事实) ----
        r = None
        deep = open_target = False        # 目标是否落在敌方自己的后两排(二线/雷区) / 二线以外
        if d_pre is None:
            k_opp.move((fr, fc), (tr, tc))            # 敌子移动 → 排除军旗/地雷
        else:
            deep = local_row(tr, opp) <= 1
            open_target = local_row(tr, opp) > 1
            r = ("win" if (att_alive and not def_alive)
                 else "trade" if (not att_alive and not def_alive) else "lose")
            if d_pid is not None:                     # 攻方视角: 我的攻子已知, 推敌守子
                if defender == "军旗":
                    k_me.poss[d_pid] = {"军旗"}       # 夺旗是全局可见的结果
                    k_me.kill((tr, tc), {"军旗"})
                else:
                    k_me.poss[d_pid] = combat_poss(k_me.poss[d_pid], name, "atk", r, deep=deep)
                    if not def_alive:
                        k_me.kill((tr, tc), k_me.poss[d_pid])
            if a_pid is not None:                     # 守方视角: 敌攻子刚动过, 由我方守子与结果推断
                k_opp.move((fr, fc), (tr, tc))
                k_opp.poss[a_pid] = combat_poss(k_opp.poss[a_pid], defender, "def", r,
                                                open_target=open_target)
                if not att_alive:
                    k_opp.kill((tr, tc), k_opp.poss[a_pid])
        for s in (side, opp):
            if self.flags_public[s]:        # 军旗公开后, 原先占营的两子身份随之确定
                kb = self.knowledge["blue" if s == "red" else "red"]
                for cell, origin in self.hq_origin[s].items():
                    pid = kb.pid_at.get(cell)
                    if pid is not None and self.cells.get(cell, (None, None))[1] == origin:
                        kb.poss[pid] = {origin}      # 原封的军旗/排长还在原营 → 身份可定
        # 观察到敌子"在铁路上拐弯"(起终点都是铁路格, 且行与列都变了、又不是相邻一步的公路移动)
        # → 这种着法只有工兵能走, 该子身份自动明牌为工兵(双方都看得见这一手, 是公开信息)
        if (is_rail(fr, fc) and is_rail(tr, tc) and abs(fr - tr) + abs(fc - tc) > 1
                and fr != tr and fc != tc):
            pid_turn = a_pid if a_pid is not None else k_opp.pid_at.get((tr, tc))
            if pid_turn is not None:
                k_opp.poss[pid_turn] = {"工兵"}
        for kb in (k_me, k_opp):
            kb.prune()
        # ---- 暗棋版历史描述(每方只看到自己该知道的) ----
        dark = {}
        for v in (side, opp):
            if v == side:
                if d_pre is None:
                    dark[v] = base
                else:
                    # 守子已亡时用 refine 后的最终 poss(夺旗会被钉死成 {军旗}), 否则用活子 poss
                    poss = (k_me.dead[d_pid] if d_pid is not None and not def_alive
                            else k_me.at((tr, tc)))
                    dark[v] = base + self._dark_tags(r, poss, defender, opp, False) + note
            else:
                tail = base.split(" ", 1)[1]          # 坐标+铁路标记(与棋种无关, 公开)
                # 敌子身份: 收窄到唯一可能时照实写名, 否则写 '?'
                if d_pre is None:
                    pe = k_opp.at((tr, tc)) if a_pid is not None else set()
                else:
                    pe = (k_opp.dead[a_pid] if a_pid is not None and not att_alive
                          else k_opp.at((tr, tc)) if a_pid is not None
                          else seed_poss(side, (fr, fc)))
                head = (f"{SIDE_CN[side]}{next(iter(pe))}" if len(pe) == 1
                        else f"{SIDE_CN[side]}?")
                if d_pre is None:
                    dark[v] = f"{head} {tail}"
                else:
                    dark[v] = f"{head} {tail}" + self._dark_tags(r, pe, defender, side, True) + note
        self.history.append((side, move, victim, desc, dark))
        return victim, desc

    # ---------------- 局面记录 ----------------

    def render(self, annotate=False, view=None):
        """annotate=True 时有级别的棋子后附 [n] 吃子级别 (给 JEV 的文本棋盘用),
        规则化标记: 级别数字大者吃数字小者, 同级同归于尽; 炸弹/地雷/军旗无级别。
        view=None → 上帝视角(棋谱/回放); view=side → 暗棋: 己方棋子全名, 敌方棋子只在
        belief 已收窄到唯一可能时显示棋名, 否则记作 '蓝?'。"""
        rows = []
        for r in range(12):
            cells = []
            for c in range(5):
                p = self.cells.get((r, c))
                if p is None:
                    t = "空"
                else:
                    name = p[1]
                    if view is not None and p[0] != view:
                        poss = self.knowledge[view].at((r, c))
                        name = next(iter(poss)) if len(poss) == 1 else "?"
                    t = f"{SIDE_CN[p[0]]}{name}"
                    if annotate and name in RANKS:
                        t += f"[{RANKS[name]}]"
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
        return (dict(self.cells), self.winner, self.quiet_plies, dict(self.flags_public))

    def restore(self, snap):
        self.cells, self.winner, self.quiet_plies = dict(snap[0]), snap[1], snap[2]
        self.flags_public = dict(snap[3])


# ---------------------------------------------------------------- 规则评估 + α-β 树搜索 AI

def evaluate(g, side):
    score = 0
    eng = {side: 0, "blue" if side == "red" else "red": 0}
    mines_opp = 0
    opp = "blue" if side == "red" else "red"
    # 敌旗位置公开后: 把"向底线推进"换成"直接指向旗格" —— 目标唯一, 不必再猜是哪座营
    flag_cell = None
    if g.flags_public[opp]:
        flag_cell = next((pos for pos, (s2, n2) in g.cells.items()
                          if s2 == opp and n2 == "军旗"), None)
    for (r, c), (s, name) in g.cells.items():
        val = VALUES[name]
        if name not in ("地雷", "军旗"):
            adv = (11 - r) if s == "red" else r      # 向敌方底线推进量
            val += 0.25 * adv + (2 if cell_kind(r, c) == "camp" else 0)
        if flag_cell is not None and name not in ("地雷", "军旗"):
            val += 1.2 * (12 - (abs(r - flag_cell[0]) + abs(c - flag_cell[1])))  # 贴旗/守旗
        if name == "工兵":
            eng[s] += 1
        elif name == "地雷" and s != side:
            mines_opp += 1
        score += val if s == side else -val
    # 公平增补(与 JEV 秘笈同源, 皆由规则推出): 敌雷未尽时工兵是唯一挖雷资本, 保值;
    # 己方大本营滞留非旗子被冻结, 是战力浪费。
    if mines_opp:
        score += 10 * eng[side] * mines_opp / 3
        score -= 10 * eng["blue" if side == "red" else "red"] * sum(
            1 for (s2, n2) in g.cells.values() if n2 == "地雷" and s2 == side) / 3
    for (r, c), (s, name) in g.cells.items():
        if s == side and name not in ("军旗",) and cell_kind(r, c) == "hq":
            score -= 8
    return score


def _root_worker(payload):
    """子进程: 在给定世界上跑一层根节点搜索, 返回 [(move, value)]。
    payload 只含可 pickle 的纯数据(占位表/公开信息/着法/时限), 不含任何对象引用。"""
    cells, fp, side, depth, real_legal, mine_pen, last_quiet, budget = payload
    g = Game.from_cells(cells)
    g.flags_public = dict(fp)
    ai = RuleSearchAI(time_budget=max(0.05, budget), max_depth=depth, samples=1, refine=0)
    ai.last_quiet = dict(last_quiet)
    ai.deadline = time.monotonic() + max(0.05, budget)
    return ai._root(g, side, depth, real_legal, mine_pen), ai.nodes


class RuleSearchAI:
    """规则评估 + 迭代加深 α-β 树搜索(时限内尽量加深)。
    暗棋公平性: 绝不读敌方真实棋种 —— 每手先从 belief 采样 samples 个"具体世界"
    (敌子棋种按可能集合随机落实, 受各棋种数量与军旗必须在大本营的约束),
    在各世界里分别搜索, 再按"每个世界给自己 argmax 投一票"聚合; 平票时取
    "各世界根节点精确分均值"最大者。采样只依赖 belief 与着法史(双方都看得见的信息),
    因此相同观测必得相同决策 —— 敌子真名怎么换都影响不了它。"""

    def __init__(self, time_budget=2.0, max_depth=4, name="RULE-AI", samples=4, refine=3, procs=0):
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.name = name
        self.samples = max(1, samples)
        self.refine = max(0, refine)         # 信念感知复评的候选手数(0 = 关闭)
        self.procs = max(0, int(procs))      # 并行子进程数(0 = 串行; CPU 密集, 线程无效)
        self._executor = None
        self.nodes = 0
        self.deadline = 0.0
        self.last_quiet = {}         # from -> to: 每颗子自己的上一手闲着, 根节点防拉锯 ε (与 JEV 剔除对应)

    def choose(self, game, side, moves):
        self.deadline = time.monotonic() + self.time_budget
        self.nodes = 0
        real_legal = set(moves)            # 裁判的合法着法集合是硬约束: 世界里多出来的着法一律不算
        search_legal = real_legal
        sapper = {m for m in real_legal
                  if game.cells[m[0]][1] == "工兵" and game.cells.get(m[1])
                  and game.cells[m[1]][0] != side
                  and not suspect_tags(game, side, m[1])}
        if sapper:                      # 工兵军纪: 不主动打"非疑似雷/弹"的格子
            search_legal = real_legal - sapper
        # 雷区警惕: 非工兵去撞"疑似地雷"一律排在搜索顺序最后(大子损失更大)
        risk_last = {m for m in search_legal
                     if game.cells[m[0]][1] != "工兵" and game.cells.get(m[1])
                     and game.cells[m[1]][0] != side
                     and "地雷" in suspect_tags(game, side, m[1])}
        # 雷区惩罚与本子价值成正比(司令撞雷损失≈100, 连长≈28), 直接压进根节点打分
        mine_pen = {}
        for m in risk_last:
            mine_pen[m] = -1.3 * VALUES[game.cells[m[0]][1]]
        worlds = self._sample_worlds(game, side)
        pick, pick_depth, pick_score = self._expect_best(game, side, sorted(search_legal)), 0, 0.0
        last_vals = []
        for depth in range(2, self.max_depth + 1):
            depth_vals, timed_out = [], False
            pool = self._pool()
            if pool is not None and len(worlds) > 1:
                left = self.deadline - time.monotonic()
                if left <= 0.05:
                    timed_out = True
                    depth_vals = []
                else:
                    payloads = [(dict(w.cells), dict(w.flags_public), side, depth, search_legal,
                                 mine_pen, dict(self.last_quiet), left) for w in worlds]
                    try:
                        for vals, nodes in pool.map(_root_worker, payloads):
                            depth_vals.append(vals)
                            self.nodes += nodes       # 子进程搜的节点也算进来, 便于看强度
                    except TimeoutError:
                        timed_out = True
                    except Exception:            # 并行不可用就退回串行
                        self.close()
                        depth_vals = []
            if not depth_vals and not timed_out:
                for w in worlds:
                    try:
                        depth_vals.append(self._root(w, side, depth, search_legal, mine_pen))
                    except TimeoutError:
                        timed_out = True
                        break
            if depth_vals:
                m, sc = self._vote(depth_vals)
                if m is not None:
                    pick, pick_depth, pick_score = m, depth, sc
                    last_vals = depth_vals
            if timed_out or time.monotonic() > self.deadline or pick_score >= BIG - 1000:
                break
        # 阶段 B: 信念感知复评 —— 对阶段 A 最好的几手, 走完更新信念后按新信念重采样再评一次。
        # 这一步把"我这手让对手/我自己学到什么"计入价值(探雷换信息、司令暴露后对方改打旗营等)。
        if last_vals and self.refine > 0 and time.monotonic() < self.deadline:
            cands = self._top_candidates(last_vals, self.refine)
            if len(cands) > 1:
                try:
                    bv = self.belief_values(game, side, [m for m in cands if m in real_legal],
                                            max(1, pick_depth - 1), mine_pen)
                    if bv:
                        pick = max(bv, key=lambda m: bv[m])
                        pick_score = bv[pick]
                        pick_depth = max(1, pick_depth - 1)
                except Exception:            # 复评只是增强项, 失败就用阶段 A 的结果
                    pass
        if pick not in real_legal:          # 兜底: 任何情况下都只交裁判认可着法
            pick = self._expect_best(game, side, moves)
        if game.cells.get(pick[1]) is None:
            self.last_quiet[pick[0]] = pick[1]
        else:
            self.last_quiet.pop(pick[0], None)
        return pick, {"depth": pick_depth, "nodes": self.nodes, "score": round(pick_score, 2),
                      "worlds": len(worlds)}

    def _pool(self):
        """惰性建进程池并复用(Windows spawn 建池要 ~0.5s, 每手新建会把收益吃掉)。"""
        if self.procs <= 1:
            return None
        if self._executor is None:
            try:
                from concurrent.futures import ProcessPoolExecutor
                self._executor = ProcessPoolExecutor(max_workers=self.procs)
            except Exception:
                self.procs = 1
                return None
        return self._executor

    def close(self):
        """关掉进程池(改动棋手/退出时调用)。"""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
            self._executor = None

    def _move_value(self, world, side, move, depth, mine_pen=None):
        """在某个世界里给指定着法打分(用于信念感知复评)。"""
        opp = "blue" if side == "red" else "red"
        undo, _ = world.make(move)
        try:
            v = -self._negamax(world, opp, depth - 1, -BIG, BIG)
        finally:
            world.unmake(undo)
        if self.last_quiet.get(move[1]) == move[0]:
            v -= 0.5
        return v + (mine_pen or {}).get(move, 0.0)

    def belief_values(self, game, side, moves, depth, mine_pen=None):
        """信念感知复评: 每个候选着法都先在"我方信念副本"上真走一步(apply 会更新信念),
        再按更新后的信念重采样世界, 在世界集合上取该着法的均值。
        这样"这一手让我学到什么"会直接体现在后续走法价值上 —— 探雷/送子换信息能被计价。"""
        out = {}
        opp = "blue" if side == "red" else "red"
        for m in moves:
            child = game.clone_belief()
            try:
                child.apply(m, side)             # 结算 + 更新我方信念(我看到的结果)
            except ValueError:
                continue
            if child.winner is not None:          # 直接赢了, 无需再评估
                out[m] = float(BIG - 1000)
                continue
            legal_child = set(child.legal_moves(side))
            if not legal_child:
                continue
            pen_child = {c: v for c, v in (mine_pen or {}).items() if c != m}
            worlds = self._sample_worlds(child, side)
            vals = [self._move_value(w, side, m, depth, pen_child) for w in worlds]
            if vals:
                out[m] = sum(vals) / len(vals)
        return out

    @staticmethod
    def _top_candidates(depth_vals, n=4):
        """阶段 A 的投票 + 均分排序, 取前 n 个候选进入信念感知复评。"""
        votes, sums, counts = {}, {}, {}
        for vals in depth_vals:
            if not vals:
                continue
            m = max(vals, key=lambda x: x[1])[0]
            votes[m] = votes.get(m, 0) + 1
        for vals in depth_vals:
            for m, v in vals:
                sums[m] = sums.get(m, 0.0) + v
                counts[m] = counts.get(m, 0) + 1
        rank = sorted(votes, key=lambda m: (-votes[m], -(sums.get(m, 0.0) / max(1, counts.get(m, 1)))))
        return rank[:n]

    def _root(self, world, side, depth, real_legal, mine_pen=None):
        """在一个具体世界里跑根节点(全窗口, 各着法分值可跨世界比较)。
        只对"真局也合法"的着法打分 —— 世界是假设, 裁判的合法集合才是硬约束;
        mine_pen 是"疑似撞雷"的额外扣分(与本子价值成正比), 这些着法也排到最后。"""
        opp = "blue" if side == "red" else "red"
        mine_pen = mine_pen or {}
        snap = world.snapshot()
        vals = []
        order = self._ordered(world, side)
        if mine_pen:
            order.sort(key=lambda m: 1 if m in mine_pen else 0)
        try:
            for m in order:
                if m not in real_legal:
                    continue
                undo, _ = world.make(m)
                v = -self._negamax(world, opp, depth - 1, -BIG, BIG)
                world.unmake(undo)
                if self.last_quiet.get(m[1]) == m[0]:
                    v -= 0.5     # 防拉锯: 该子原路返回闲着, 根节点微惩 (不影响深层搜索)
                v += mine_pen.get(m, 0.0)
                vals.append((m, v))
        except TimeoutError:
            world.restore(snap)      # 超时路径上有未回退的 make, 整层回滚
            raise
        return vals

    @staticmethod
    def _vote(depth_vals):
        """每个世界投自己最优着法一票; 平票时取"各世界根节点精确分均值"最大者。"""
        votes = {}
        for vals in depth_vals:
            if not vals:
                continue
            m = max(vals, key=lambda x: x[1])[0]
            votes[m] = votes.get(m, 0) + 1
        if not votes:
            return None, 0.0

        def mean(m):
            vs = [v for vals in depth_vals for (mm, v) in vals if mm == m]
            return sum(vs) / len(vs) if vs else -BIG

        top = max(votes.values())
        tied = [m for m, c in votes.items() if c == top]
        best = max(tied, key=mean) if len(tied) > 1 else tied[0]
        return best, mean(best)

    # ---------------- 暗棋: 从 belief 采样具体世界 ----------------

    def _sample_worlds(self, game, side):
        """采样 samples 个具体世界: 己方棋种为已知真值, 敌方棋种按 poss 随机落实。"""
        k = game.knowledge[side]
        seed = self._obs_seed(game, side)
        worlds = []
        for t in range(self.samples):
            rng = random.Random(seed + 7919 * t)
            assign = self._sample_assignment(k, rng)
            cells = {(r, c): (s, name if s == side else assign[(r, c)])
                     for (r, c), (s, name) in game.cells.items()}
            w = Game.from_cells(cells)
            w.flags_public = dict(game.flags_public)   # 军旗公开是公开信息, 各世界必须与真局一致
            worlds.append(w)
        return worlds

    @staticmethod
    def _sample_assignment(k, rng):
        """给每颗活敌子落实一个具体棋种: 逐格按 poss 随机取(受各棋种剩余数量约束),
        大本营两格一军旗一排长(已公开时按公开信息)。排序+种子化 rng → 结果确定。"""
        used = {n: sum(1 for p in k.dead if k.dead[p] == {n}) for n in PIECE_COUNTS}
        assign = {}
        hqs = [c for c in sorted(k.pid_at) if cell_kind(*c) == "hq"]
        for c in hqs:                                  # 军旗已公开时 poss 已被收窄成唯一
            if len(k.poss[k.pid_at[c]]) == 1:
                n = next(iter(k.poss[k.pid_at[c]]))
                assign[c], used[n] = n, used.get(n, 0) + 1
        rest_hq = [c for c in hqs if c not in assign]
        if len(rest_hq) == 2:
            flag_cell = rng.choice(rest_hq)
            for c in rest_hq:
                n = "军旗" if c == flag_cell else "排长"
                assign[c], used[n] = n, used.get(n, 0) + 1
        elif rest_hq:                                   # 只剩一个大本营有敌子 → 军旗必在其中
            c = rest_hq[0]
            assign[c], used["军旗"] = "军旗", used.get("军旗", 0) + 1
        for c in sorted(k.pid_at):
            if c in assign:
                continue
            poss = k.poss[k.pid_at[c]]
            names = [n for n in sorted(poss, key=name_order) if used.get(n, 0) < PIECE_COUNTS[n]]
            n = names[0] if len(names) == 1 else rng.choice(names or sorted(poss, key=name_order))
            assign[c], used[n] = n, used.get(n, 0) + 1
        return assign

    @staticmethod
    def _obs_seed(game, side):
        """由公开信息(着法史的坐标与行棋方)导出的确定性种子 —— 不含任何敌子真名。"""
        side_id = {"blue": 1, "red": 2}
        h = 7
        for rec in game.history:
            for pos in (rec[1][0], rec[1][1]):
                h = (h * 131 + pos[0] * 7 + pos[1]) & 0xFFFFFFFF
            h = (h * 131 + side_id[rec[0]]) & 0xFFFFFFFF
        return (h * 2654435761 + side_id[side]) & 0xFFFFFFFF

    def _expect_best(self, game, side, moves):
        """搜索全部超时时(极窄时限)的兜底: 按 belief 的期望战果选最优着法, 仍不读敌子真名。"""
        k = game.knowledge[side]
        best, best_ev = moves[0], -BIG
        for m in moves:
            d = game.cells.get(m[1])
            if d is None or d[0] == side:
                continue
            mine = game.cells[m[0]][1]
            poss = [n for n in k.at(m[1]) if n not in ("地雷", "军旗")] or list(k.at(m[1]))
            ev = 0.0
            for n in poss:
                if n == "军旗":
                    ev += 10 * BIG / max(1, len(poss))
                elif n == "地雷":
                    # 撞雷损失与本子价值成正比: 大子忌碰雷, 工兵才是探雷资本
                    ev += (60 if mine == "工兵" else -1.1 * VALUES[mine]) / max(1, len(poss))
                elif mine == "炸弹" or n == "炸弹":
                    ev += (0.3 * VALUES[n] - 0.7 * VALUES[mine]) / max(1, len(poss))
                elif RANKS[mine] > RANKS[n]:
                    ev += VALUES[n] / max(1, len(poss))
                elif RANKS[mine] == RANKS[n]:
                    ev += 0.2 * VALUES[n] / max(1, len(poss))
                else:
                    ev -= 0.6 * VALUES[mine] / max(1, len(poss))
            if ev > best_ev:
                best, best_ev = m, ev
        return best

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
    "军棋(陆战棋·暗棋)规则: 棋盘5列(a-e)×12行(1-12), 行1-6蓝方半场, 行7-12红方半场; "
    "行6/7之间为山界, 仅b/c/d三桥相通。公路行一格; 铁路(第6、7行整行与a/e两侧列)上可直线任意远"
    "(不可转弯), 工兵在铁路网内可任意转弯(不可越子)。行营(◎)内的棋子不可被攻击; "
    "大本营(◆)只布军旗与排长, 入营即冻结不能再动, 腾空后任何棋子都不得再进入; "
    "敌方军旗位置公开后, 只许进军旗营夺旗, 另一个大本营(占位排长或空营)禁止再进去送子。"
    "吃子级别: 己方棋子后的[n]即级别, 司令[9]>军长[8]>师长[7]>旅长[6]>团长[5]>营长[4]>连长[3]>排长[2]>工兵[1], "
    "级别大者歼灭级别小者, 同级相碰同归于尽(兑子); 炸弹与任何子相碰同归于尽(可炸掉司令); "
    "地雷不能移动, 仅工兵可挖雷(工兵存活并占其格), 炸弹碰雷同尽, 其余子触雷阵亡(雷保留)。"
    "夺取敌方军旗, 或令对方无子可动(困毙), 即获胜; 手数打满按剩余子力判定。"
    "【暗棋】你只能看见己方棋子的身份; 敌方棋子一律显示为'蓝?'或'红?', 只有被排除法确证时才显示棋名。"
    "攻击选项里的'敌子未明: 师长(吃)/炸弹(同尽)'给出该敌子的全部可能棋种与各自结果 —— "
    "同级与炸弹在结果上完全等价(都是同尽), 无法分辨, 不要把双亡当作'一定是同级'。"
    "敌子每动一次就可能被排除军旗与地雷; 敌司令阵亡后, 敌方军旗位置即公开(注意: 大本营只布军旗与排长)。"
)

TIPS_TEXT = (
    "高手秘笈: "
    "①司令[9]/军长[8]是核心战力, 吃高价值子收益巨大, 但司令唯一的克星是炸弹 —— "
    "所以只打'敌子未明'里不含炸弹(或已确定不是炸弹)的目标; "
    "②炸弹换司令/军长永远血赚, 换团长[5]以下亏; 炸弹在没把握时留作前线威慑; "
    "③工兵[1]是唯一能挖雷的子, 敌雷未清时每多一颗敌雷工兵越值钱, 别拿工兵去兑排长; "
    "③b雷区纪律: 雷只可能埋在敌方后两排(二线以下), 越大的子撞雷损失越狠 —— "
    "司令[9]/军长[8]/师长[7]千万别去试雷, 要探雷只派工兵[1], 或用连长[3]/排长[2]去逼对方动; "
    "③c工兵只主动打'高度怀疑是雷或弹'的格子, 其余目标一律不碰 —— 工兵活着才挖得到雷; "
    "④被攻击时退进行营(◎)即安全, 行营也是不挨打的前沿支点; "
    "⑤铁路是高速通道: 工兵沿铁路转弯可直插敌空虚的后两排(雷区), 普通子只能沿铁路直线突进; "
    "⑥兑子先算级别差: 用[2]换[7]血赚, 用[8]换[5]血亏; 子力领先时主动兑子简化局势; "
    "⑦同一子在两格间来回拉锯零进展, 原路返回的着法已从选项中剔除, 请向前走; "
    "⑧敌军旗必在其两个大本营之一, 通路打开后直插夺旗; 守势同理, 盯死己方大本营前的通道。 "
    "暗棋诀窍: 敌子每动一次, 它的可能棋种就少几个(军旗/地雷立刻出局); 反复用小子试探雷区与可疑子, "
    "把敌子的可能集合逼到一两种, 再决定用司令还是炸弹出手 —— 宁可多花一两手把身份问清, 也不要蒙着眼打司令。"
)


def posterior_certainty(game, view, mc=120, seed=12345):
    """暗棋身份确定性: 按知识库采样 mc 个"具体世界", 统计每颗敌子各棋种出现的频率。
    返回 {cell: (最可能棋种, 概率, 次可能[(棋种, 概率)]) } —— 概率就是这一格的"确定程度"(1.0 = 完全确定)。
    采样用的是 RULE-AI 那套约束(数量、大本营、已公开的军旗), 不读敌子真名。"""
    k = game.knowledge[view]
    cells = sorted(k.pid_at)
    if not cells:
        return {}
    hits = {c: {} for c in cells}
    for i in range(mc):
        assign = RuleSearchAI._sample_assignment(k, random.Random(seed + 7919 * i))
        for c in cells:
            n = assign.get(c)
            if n:
                hits[c][n] = hits[c].get(n, 0) + 1
    out = {}
    for c in cells:
        if not hits[c]:
            continue
        ranked = sorted(hits[c].items(), key=lambda kv: (-kv[1], name_order(kv[0])))
        name, cnt = ranked[0]
        out[c] = (name, cnt / mc,
                  [(n, k / mc) for n, k in ranked[1:4]])      # 次可能的几种, 供界面展示分布
    return out


def suspect_tags(game, side, cell):
    """该敌格"高度疑似"的棋种(只由 belief 推出, 不读真名)。
    地雷: 可能集合含雷, 且(该格已基本锁定 或 剩下的候选雷位≤2 —— 全场只有3颗雷)。
    炸弹: 可能集合已收窄到≤2 且含炸弹(同级与炸弹结果等价, 只能靠排除法缩小)。"""
    k = game.knowledge[side]
    poss = k.at(cell)
    tags = set()
    if "地雷" in poss and (len(poss) <= 3 or
                           sum(1 for p in k.pid_at.values() if "地雷" in k.poss[p]) <= 2):
        tags.add("地雷")
    if "炸弹" in poss and len(poss) <= 2:
        tags.add("炸弹")
    return tags


def suspect_text(game, side, cell):
    return "/".join(sorted(suspect_tags(game, side, cell), key=name_order))


def mine_intel(game, side):
    """地雷情报 (暗棋版): 地雷的布局约束是"本方后两排 + 不能移动", 因此敌子一旦被观察到
    移动过就不可能是雷, 棋盘上敌方的雷数恰为3。全部结论只来自 belief(可能棋种集合)与
    行动史, 不读敌方真名。返回
    (候选雷位[coord], 敌后两排未动的非雷滞留子[(coord, poss)], 己方雷位[coord])。"""
    opp = "blue" if side == "red" else "red"
    k = game.knowledge[side]
    cand, sitters, mines_side = [], [], []
    for (r, c), (s, name) in sorted(game.cells.items()):
        if s == side:
            if name == "地雷":
                mines_side.append(coord((r, c)))
            continue
        pid = k.pid_at.get((r, c))
        if pid is None:
            continue
        poss = k.poss[pid]
        if "地雷" in poss:
            cand.append(coord((r, c)))
        elif pid not in k.moved and "军旗" not in poss and local_row(r, opp) <= 1:
            sitters.append((coord((r, c)), poss))
    return cand, sitters, mines_side


class JevPlayer:
    HIST_PLIES = 24          # 状态里回放的最近手数

    def __init__(self, url="http://127.0.0.1:8090", name="JEV", temperature=1.036, max_plies=200):
        self.url = url.rstrip("/")
        self.name = name
        self.temperature = temperature
        self.max_plies = max_plies
        self.last_quiet = {}         # from -> to: 每颗子自己的上一手闲着, 用于防来回拉锯
        self.http = requests.Session()

    def _no_shuffle(self, game, capped):
        """防拉锯 (逐子): 若某子自己的上一手是闲着 f→t, 则剔除该子现在的闲着 t→f (原路返回)。
        只在还有其他着法时剔除; 返回格被占(变成吃子)或该子上手是吃子时不适用。"""
        kept = [m for m in capped
                if not (self.last_quiet.get(m[1]) == m[0] and game.cells.get(m[1]) is None)]
        return kept if kept else capped

    def choose(self, game, side, moves):
        capped, trimmed = list(moves), False
        if len(moves) > 250:      # 接口上限255, 用吃子着法优先裁剪(实际对局极少触发)
            caps = [m for m in moves if game.cells.get(m[1])]
            rest = [m for m in moves if not game.cells.get(m[1])]
            capped, trimmed = (caps + rest)[:250], True
        capped = self._no_shuffle(game, capped)
        capped = self._sapper_doctrine(game, side, capped, moves)
        criteria = {}
        for i, m in enumerate(capped):
            note = self.safety_note(game, side, m)
            criteria[f"M{i:03d}"] = game.describe(m, view=side) + (f" [{note}]" if note else "")
        instructions = (
            f"你是{SIDE_CN[side]}方, 暗棋: 敌子身份不可见, 每条选项只如实预告你这一手可能得到的结果。 "
            f"方括号是走子后的风险评估(只依据敌子的可能棋种): 会被吃=危险, 可互换=兑子, 入行营=落点不可被攻击; "
            f"无方括号=落点暂时安全。'敌子未明: A(吃)/B(同尽)'表示该格敌子可能是A也可能B, "
            f"要按最坏情况(同级=同尽)算风险, 也要看最好情况(A=吃)的收益值不值这个风险。"
            f"请选出对{SIDE_CN[side]}方最有利的一步: 优先安全夺旗或吃掉高价值子(敌子已确证时最果断), "
            f"避免在'敌子未明'里用司令/军长去赌可能藏着炸弹的目标; 雷区(敌方后两排)一律派工兵去探, "
            f"大子撞雷损失与自身价值成正比, 千万别用司令/军长/师长去试雷; "
            f"工兵的选项已按军纪律掉(只保留疑似雷/弹的目标)。我方子力领先时可主动兑子, "
            f"落后时避免兑子并优先保住工兵。同一子来回拉锯零进展, 原路返回的着法已被剔除, 请推进有意义的着法。"
            + ("(着法过多, 已按吃子优先裁剪)" if trimmed else ""))
        payload = {"state": self.state_text(game, side),
                   "questions": {"move": {"type": "choice", "instructions": instructions,
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
                    if game.cells.get(move[1]) is None:
                        self.last_quiet[move[0]] = move[1]     # 记该子的闲着轨迹
                    else:
                        self.last_quiet.pop(move[0], None)     # 吃子后拉锯记忆清零
                    top = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])[:3]
                    return move, {"top": [(k, round(p, 3)) for k, p in top],
                                  "confidence": round(ans["confidence"], 3)}
                last_err = RuntimeError(f"无效选择 {key!r}")
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        move = random.choice(capped)
        if game.cells.get(move[1]) is None:
            self.last_quiet[move[0]] = move[1]
        else:
            self.last_quiet.pop(move[0], None)
        return move, {"fallback": str(last_err)}

    @staticmethod
    def _sapper_doctrine(game, side, capped, all_moves):
        """工兵军纪: 工兵是唯一挖雷资本, 只允许主动攻击"高度怀疑是地雷或炸弹"的格子,
        其余目标一律不碰(保住它去探雷)。若过滤后空无一着, 就退回全部着法(不制造无棋可走)。"""
        kept = [m for m in capped
                if not (game.cells[m[0]][1] == "工兵"
                        and game.cells.get(m[1]) and game.cells[m[1]][0] != side
                        and not suspect_tags(game, side, m[1]))]
        return kept if kept else list(all_moves)

    @staticmethod
    def safety_note(game, side, move):
        """走完这步后, 我方落在目标格的棋子是否会被对方攻击, 以及这一步本身有没有撞雷风险。
        暗棋版: 威胁判定只用"棋盘位置与行营"(双方都看得见) + belief 里敌子的可能棋种,
        不读敌方真名; 敌子"可能是工兵"时, 铁路转弯的威胁也必须计入 —— 否则用排除法
        就能反推出它不是工兵。撞雷损失按本子价值折算(越大的子越忌讳雷区)。"""
        opp = "blue" if side == "red" else "red"
        mine_warn = ""
        tgt = game.cells.get(move[1])
        if tgt is not None and tgt[0] != side and "地雷" in suspect_tags(game, side, move[1]):
            src = game.cells[move[0]][1]
            mine_warn = (f"疑似撞雷({src}损失≈{VALUES[src]})"
                         + ("，大子忌撞雷宜先试探" if VALUES[src] >= 55 else "，宜派工兵来探"))
        note = JevPlayer._threat_note(game, side, move) + mine_warn
        return note.strip("， ")

    @staticmethod
    def _threat_note(game, side, move):
        """只算"落点会不会被打"这一半, 与雷区提示分开, 便于组合。"""
        opp = "blue" if side == "red" else "red"
        k = game.knowledge[side]
        undo, _ = game.make(move)
        try:
            if game.winner is not None:
                return ""
            mine = game.cells.get(move[1])
            if mine is None or mine[0] != side:
                return ""                    # 攻方亡: 选项描述已预告可能结果
            if cell_kind(*move[1]) == "camp":
                return "入行营，不可被攻击"
            mv = mine[1]
            cand = {}
            for m2 in game.legal_moves(opp):              # 真实可达(已含敌工兵的转弯)
                if m2[1] == move[1] and k.pid_at.get(m2[0]) is not None:
                    cand[k.pid_at[m2[0]]] = k.poss[k.pid_at[m2[0]]]
            for (r, c), (s, _) in game.cells.items():      # belief 里的"可能是工兵"也要算威胁
                if s != opp or not is_rail(r, c):
                    continue
                pid = k.pid_at.get((r, c))
                if pid is None or pid in cand or "工兵" not in k.poss[pid]:
                    continue
                if move[1] in game._rail_bfs((r, c), opp):
                    cand[pid] = k.poss[pid]

            def cat_of(n):
                """敌子 n 撞我方 mv 的结果类别: kill=我亡, trade=同尽, safe=我吃掉它。"""
                if n == "炸弹":
                    return "trade" if mv == "炸弹" else "kill"
                if mv == "炸弹":
                    return "trade"
                if RANKS[n] > RANKS[mv]:
                    return "kill"
                if RANKS[n] == RANKS[mv]:
                    return "trade"
                return "safe"

            kill, trade = [], []
            for pid, poss in cand.items():
                moveable = {n for n in poss if n not in ("地雷", "军旗")}
                if not moveable:
                    continue
                cats = {cat_of(n) for n in moveable}
                if "kill" in cats:
                    kill.append(moveable)
                if "trade" in cats:
                    trade.append(moveable)

            def report(bucket, cat, prefix, suffix):
                p = next((q for q in bucket if all(cat_of(n) == cat for n in q)), bucket[0])
                hard = all(cat_of(n) == cat for n in p)
                name = next(iter(p)) if len(p) == 1 else "子"
                tail = "" if len(p) == 1 else f"(敌子可能: {poss_text(p)})"
                return f"移后{'会' if hard else '可能'}{prefix}{SIDE_CN[opp]}{name}{suffix}{tail}"

            if kill:
                return report(kill, "kill", "被", "吃掉")
            if trade:
                return report(trade, "trade", "与", "同尽")
            return ""
        finally:
            game.unmake(undo)
    def state_text(self, game, side):
        opp = "blue" if side == "red" else "red"
        k = game.knowledge[side]
        alive = {}
        for (r, c), (s, name) in game.cells.items():
            if s == side and name != "军旗":
                alive[name] = alive.get(name, 0) + 1
        my_flag = next((coord((r, c)) for (r, c), (s, n) in sorted(game.cells.items())
                        if s == side and n == "军旗"), "?")
        # 敌方: 只知存活总数与"已被排除法确证"的那几颗, 阵亡总数可知, 类型与子力不可知
        live_pids = list(k.pid_at.values())
        known = {}
        for p in live_pids:
            if len(k.poss[p]) == 1:
                n = next(iter(k.poss[p]))
                known[n] = known.get(n, 0) + 1
        dead_total = len(k.dead)

        def roster():
            def tag(n):
                return f"{n}[{RANKS[n]}]" if n in RANKS else n
            body = "、".join(f"{tag(n)}×{k2}" if k2 > 1 else tag(n)
                             for n, k2 in PIECE_COUNTS.items() if alive.get(n))
            return body or "无"

        own_lost = game.captured()[side]
        known_txt = "、".join(f"{n}×{v}" for n, v in sorted(known.items(), key=lambda x: name_order(x[0]))) or "无"
        hist_rows = game.history[-self.HIST_PLIES:]
        base = len(game.history) - len(hist_rows)
        hist = "  ".join(f"{base + i + 1}.{SIDE_CN[h[0]]}{h[4].get(side, h[3])}"
                         for i, h in enumerate(hist_rows))
        toward = "向行1方向进攻" if side == "red" else "向行12方向进攻"
        cand, sitters, mines_side = mine_intel(game, side)
        sit_txt = ", ".join(f"{p}(可能: {poss_text(ps, 3)})" for p, ps in sitters) or "无"
        opp_flag = (f"{next((coord((r, c)) for (r, c), (s, n) in sorted(game.cells.items()) if s == opp and n == '军旗'), '?')}"
                    f"(已公开)" if game.flags_public[opp] else "未知(在两个大本营之一)")
        flag_rule = ("敌旗已公开: 现在只许进军旗营夺旗, 另一个大本营(占位排长/空营)已禁止进入。"
                     if game.flags_public[opp] else "敌旗未公开时两个大本营都能进, 但进去就出不来。")
        return (f"你是{SIDE_CN[side]}方(JEV), 对手是{SIDE_CN[opp]}方AI。{RULES_TEXT}\n{TIPS_TEXT}\n"
                f"行营(◎)坐标: {' '.join(coord(p) for p in CAMPS)}; "
                f"大本营(◆)坐标: {' '.join(coord(p) for p in HQ_ALL)}; "
                f"{SIDE_CN[side]}方从本方半场{toward}。\n"
                f"【局势】第{len(game.history) + 1}手/至多{self.max_plies}手(打满按剩余子力判定), 轮到{SIDE_CN[side]}方。"
                f"己方子力{game.material(side)}; 敌方子力不可知(暗棋), 只能数出存活{len(live_pids)}子/已亡{dead_total}子。\n"
                f"【军旗】己方军旗在{my_flag}(大本营内, 你自己当然知道); 敌方军旗在{opp_flag}; "
                f"夺取敌方军旗立即获胜。大本营只布军旗与排长, 入营即冻结不能动, 腾空后任何子都进不去。{flag_rule}\n"
                f"【地雷情报】敌方雷只可能在其本方后两排且全程未动, 恰3颗 —— 候选雷位: {' '.join(cand) or '无'}"
                f"(这些格子里的敌子可能是雷, 它们的其它可能棋种见下面state与选项); "
                f"敌方后两排未动且可排除雷/旗的滞留子(可放心吃): {sit_txt}; "
                f"己方地雷: {' '.join(mines_side) or '无'}(勿忘它们仍在护旗)。\n"
                f"【存活棋子】己方({SIDE_CN[side]}): {roster()} | 敌方({SIDE_CN[opp]}): 存活{len(live_pids)}子, "
                f"其中已确证棋名: {known_txt}\n"
                f"【阵亡】己方: {own_lost} | 敌方: 共亡{dead_total}子(类型不可知)\n"
                f"【最近着法(近{len(hist_rows)}手, 敌子一律不写真名)】{hist or '开局(尚无着法)'}\n"
                f"【当前棋盘】\n{game.render(annotate=True, view=side)}")


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


def play_game(idx, args, log, live=None):
    rng = random.Random(args.seed + idx)
    g = Game(random_layout("blue", rng), random_layout("red", rng))
    jev = JevPlayer(args.url, max_plies=args.max_plies)
    ai = RuleSearchAI(args.ai_time, args.ai_depth, samples=getattr(args, "ai_samples", 4),
                      refine=getattr(args, "ai_refine", 3), procs=getattr(args, "ai_procs", 0))
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
    if live:
        live.start_game(f"第{idx + 1}局  红方={players['red'].name}  蓝方={players['blue'].name}",
                        dict(geometry_edges(), camps=[list(p) for p in CAMPS],
                             hq=[list(p) for p in HQ_ALL], cols=5, rows=12),
                        pieces)
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
        if live:
            live.append(replay_moves[-1])
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
    if live:
        live.finish(replay["result"])
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

  .wrap { display:flex; gap:20px; padding:6px 26px 10px; align-items:flex-start; flex-wrap:wrap;
          max-width:1720px; margin:0 auto; }

  /* ---------------- 棋盘: 纸板 + 印刷框线 ---------------- */
  .boardcol { flex:none; }
  .board { position:relative; width:776px; height:836px; background:var(--paper);
           border-radius:10px; border:1px solid #c7cbb7;
           box-shadow:0 14px 44px rgba(0,0,0,.5), 0 2px 0 rgba(255,255,255,.06) inset; }
  .board::after { content:""; position:absolute; inset:0; border-radius:10px; pointer-events:none;
           box-shadow:inset 0 0 60px rgba(90,74,50,.14); }   /* 纸面晕影 */
  .board > svg, .board > div { position:absolute; left:0; top:0; }
  svg.base { border-radius:10px; }
  svg.arrow { pointer-events:none; z-index:6; overflow:visible; }

  /* ---------------- 棋子: 立体瓷片 ---------------- */
  .piece { position:absolute; width:104px; height:34px; border-radius:7px; z-index:3;
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
  .side { flex:0 0 300px; min-width:270px; }
  .side.listcol { flex:1 1 360px; min-width:330px; }
  .listcol .panel { height:836px; display:flex; flex-direction:column; }
  @media (max-width:1360px) { .listcol .panel { height:auto; } }
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

  #mvlist { flex:1; min-height:220px; overflow-y:auto; font-size:12.5px; scrollbar-width:thin;
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
      <small class="kai">陆 战 棋 · 对 局 回 放 （上 帝 视 角）</small>
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
      <div class="trays">
        <div class="half"><h2>蓝 方 战 损</h2><div class="tray blue" id="trayB"></div></div>
        <div class="half"><h2>红 方 战 损</h2><div class="tray red" id="trayR"></div></div>
      </div>
    </div>
  </div>
  <div class="side listcol">
    <div class="panel">
      <h2>着 法 记 录</h2>
      <div id="mvlist"></div>
    </div>
  </div>
</div>
<footer>
  <kbd>←</kbd> <kbd>→</kbd> 逐手 · <kbd>空格</kbd> 播放/暂停 · <kbd>Home</kbd>/<kbd>End</kbd> 开局/终局 · 点击着法跳转<br>
  回放由 junqi.py 裁判引擎自动生成 (行营庇护 / 铁路滑行 / 工兵转弯 / 地雷炸弹 / 夺旗与困毙均经规则自检) ·
  本页是<b>上帝视角</b>: 双方棋子真实身份与真实战斗结果, 便于复盘; 对局中双方按暗棋规则只知己方棋子 ·
  棋盘视觉参考开源项目 phdsx/junqi · BitByBitG/siguo-junqi · samuelyuan/online-junqi
</footer>
<script>
const DATA = __REPLAY_DATA__;
const LIVE = __LIVE__;                     /* 实时观战模式: 轮询 /state 增量推演 */
const CSX = 128, CSY = 64, PADX = 68, PADY = 34;   /* 横向严格 2 倍: 格 128x64, 左右留白也加倍 */
const PW = 104, PH = 34;                     /* 棋子尺寸 */
const SW = PADX * 2 + 5 * CSX, SH = PADY * 2 + 12 * CSY;   /* 纸面 776x836 */
const FRONT_Y = PADY + 6 * CSY;             /* 前线: 第6/7行之间 */
const COLS = "abcde";
const board = document.getElementById("board");
const state = new Map();   /* id -> {el, r, c, alive, name, side} */

const cx = c => PADX + c * CSX + CSX / 2;
const cy = r => PADY + r * CSY + CSY / 2;
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
  setResult(DATA.result);
}

function setResult(res) {
  const bar = document.getElementById("resbar");
  if (!res || !res.winner) { bar.innerHTML = LIVE ? "⏳ 对局进行中…" : ""; return; }
  if (res.winner === "draw") bar.innerHTML = "⚖ 平局 · " + esc(res.reason) + ` · ${res.plies}手 ${res.seconds}s`;
  else bar.innerHTML = `<b class="${res.winner === "red" ? "red" : "blue"}">${res.winner === "red" ? "红方" : "蓝方"}(${esc(res.winner)})胜</b> · ${esc(res.reason)} · ${res.plies}手 ${res.seconds}s`;
}

function place(st) {
  st.el.style.left = (cx(st.c) - PW / 2) + "px";
  st.el.style.top = (cy(st.r) - PH / 2) + "px";
  st.el.classList.toggle("dead", !st.alive);
}

function drawArrow(m) {   /* 当前手走子箭头 (仅铁路等远距离移动; 单步由金色高亮环标注) */
  const g = document.getElementById("arrowLayer");
  if (!m) { g.innerHTML = ""; return; }
  const [fr, fc] = m.fr, [tr, tc] = m.to;
  const x1 = cx(fc), y1 = cy(fr), x2 = cx(tc), y2 = cy(tr);
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  if (len < CSY * 1.6) { g.innerHTML = ""; return; }
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
function addMoveRow(m, i) {
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
}
DATA.moves.forEach((m, i) => addMoveRow(m, i));
document.getElementById("seek").max = DATA.moves.length;
renderTo(0);

if (LIVE) {
  /* 实时观战: 轮询 /state, 增量推演新手; 用户回看历史时暂停自动推进, 回到最新处继续 */
  let liveSeen = 0, liveDone = false;
  const poll = async () => {
    try {
      const s = await (await fetch("/state", {cache: "no-store"})).json();
      if (s.title !== DATA.title) { location.reload(); return; }   // 下一局开新盘
      /* 先同步新手进 DATA.moves (step 的上限是它), 再自动推演 */
      for (let i = DATA.moves.length; i < s.moves.length; i++) { DATA.moves.push(s.moves[i]); addMoveRow(s.moves[i], i); }
      document.getElementById("seek").max = DATA.moves.length;
      if (cur === liveSeen && cur < DATA.moves.length) while (cur < DATA.moves.length) step(1);
      liveSeen = DATA.moves.length;
      if (!liveDone && s.result && s.result.done) {
        liveDone = true;
        setResult(s.result);
        document.getElementById("curdesc").innerHTML =
          "<span style='color:#ffd98a'>终局 — 完整回放 HTML 已写入 junqi/games/</span>";
      }
    } catch (e) { /* 局间间隙, 忽略 */ }
  };
  setInterval(poll, 1200);
  poll();
}
</script>
</body>
</html>
"""


def render_replay_html(replay):
    return (REPLAY_TEMPLATE
            .replace("__LIVE__", "false")
            .replace("__TITLE__", replay["title"])
            .replace("__REPLAY_DATA__", json.dumps(replay, ensure_ascii=False)))


def live_page_html(base):
    """实时观战页 (上帝视角): 同一模板, moves 置空由轮询增量推演。"""
    stub = {"title": base["title"],
            "result": {"winner": "", "winner_side": "", "reason": "", "plies": 0, "seconds": 0},
            "geometry": base["geometry"], "pieces": base["pieces"], "moves": []}
    return (REPLAY_TEMPLATE
            .replace("__LIVE__", "true")
            .replace("__TITLE__", base["title"] + " · 实时观战")
            .replace("__REPLAY_DATA__", json.dumps(stub, ensure_ascii=False)))


class LiveBroadcaster:
    """对局实况服务: GET / 返回实时观战页, GET /state 返回当前对局全量 JSON。
    对局线程只做 start_game/append/finish 三个动作, 锁保护; 轮询端增量推演动画。"""

    def __init__(self, port):
        self.port = port
        self.url = None
        self.lock = threading.Lock()
        self.state = {"game_id": 0, "title": "", "result": None,
                      "geometry": None, "pieces": [], "moves": []}

    def start(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self2):
                path = self2.path.split("?", 1)[0]
                if path == "/state":
                    with outer.lock:
                        body = json.dumps(outer.state, ensure_ascii=False).encode()
                    self2._send(200, body, "application/json")
                elif path in ("/", "/live"):
                    with outer.lock:
                        base = {"title": outer.state["title"],
                                "geometry": outer.state["geometry"],
                                "pieces": outer.state["pieces"]}
                    if base["geometry"] is None:
                        self2._send(200, "<meta charset='utf-8'>等待对局开始…".encode(), "text/html; charset=utf-8")
                        return
                    self2._send(200, live_page_html(base).encode(), "text/html; charset=utf-8")
                else:
                    self2._send(404, b"{}", "application/json")

            def log_message(self2, *a):
                pass

        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        except OSError:
            return False
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.port}/"
        return True

    def start_game(self, title, geometry, pieces):
        with self.lock:
            self.state["game_id"] += 1
            self.state["title"] = title
            self.state["geometry"] = geometry
            self.state["pieces"] = pieces
            self.state["moves"] = []
            self.state["result"] = None

    def append(self, move_rec):
        with self.lock:
            self.state["moves"].append(move_rec)

    def finish(self, result):
        with self.lock:
            self.state["result"] = {**result, "done": True}


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
    # 1b) 工兵铁路转弯: 不能落到自己子上(曾误判为可吃, 会把己方子一起删掉), 可吃敌子
    g1b = Game({}, {}, check=False)
    g1b.cells[(5, 0)] = ("red", "工兵")
    g1b.cells[(4, 0)] = ("red", "排长")            # 己方子: 不可落(且不能把它吃掉)
    g1b.cells[(6, 2)] = ("blue", "司令")           # 前线铁路上的敌子: 可吃
    g1b.cells[(5, 3)] = ("blue", "连长")
    dst1b = {m[1] for m in g1b.legal_moves("red")}
    assert (4, 0) not in dst1b, "工兵不能落到自己的子上"
    assert (6, 2) in dst1b and (5, 3) in dst1b, "工兵应能吃前线敌子"
    assert g1b.cells[(4, 0)] == ("red", "排长") and g1b.cells[(5, 0)] == ("red", "工兵"), "棋盘未被改动"
    # 1c) 铁路细则: 任何子都不可越子; 普通子不可转弯; 工兵可转弯但同样不可越子/越敌子
    g1c = Game({}, {}, check=False)
    g1c.cells[(5, 0)] = ("red", "工兵")
    g1c.cells[(2, 0)] = ("blue", "排长")           # a3 敌子: 可吃, 其身后 a2/a1 到不了
    d1c = {m[1] for m in g1c.legal_moves("red") if m[0] == (5, 0)}
    assert (3, 0) in d1c and (2, 0) in d1c, "工兵可沿铁路走到 a4 并吃 a3 的敌子"
    assert (1, 0) not in d1c and (0, 0) not in d1c, "工兵不可越子到 a2/a1"
    g1d = Game({}, {}, check=False)
    g1d.cells[(5, 0)] = ("red", "排长")
    g1d.cells[(2, 0)] = ("blue", "团长")
    d1d = {m[1] for m in g1d.legal_moves("red") if m[0] == (5, 0)}
    assert (4, 0) in d1d and (3, 0) in d1d and (2, 0) in d1d, "普通子可沿 a 列直线推进并吃第一个敌子"
    assert (1, 0) not in d1d, "普通子不可越子"
    assert (6, 1) not in d1d and (4, 1) not in d1d, "普通子不可在铁路上转弯"
    g1e = Game({}, {}, check=False)
    g1e.cells[(5, 0)] = ("red", "工兵")
    g1e.cells[(5, 1)] = ("blue", "连长")           # 前线正前方有敌子
    d1e = {m[1] for m in g1e.legal_moves("red") if m[0] == (5, 0)}
    assert (5, 1) in d1e, "工兵可吃相邻敌子"
    assert (5, 2) not in d1e and (5, 3) not in d1e, "工兵不可穿过敌子继续往前"
    g1f = Game({}, {}, check=False)
    g1f.cells[(5, 0)] = ("red", "工兵")
    g1f.cells[(5, 1)] = ("red", "排长")           # 己方子挡路: 既不能吃也不能越过
    d1f = {m[1] for m in g1f.legal_moves("red") if m[0] == (5, 0)}
    assert (5, 1) not in d1f and (5, 2) not in d1f, "工兵不可吃/越过己方子"
    # 2) 普通子铁路直线滑行, 不可转弯, 一步过桥
    g2 = Game({}, {}, check=False)
    g2.cells.clear()
    g2.cells[(5, 0)] = ("red", "排长")
    dst2 = {m[1] for m in g2.legal_moves("red")}
    assert (5, 4) in dst2 and (4, 0) in dst2 and (1, 0) in dst2, "普通子应能沿铁路直线滑行"
    assert (6, 1) not in dst2 and (4, 1) not in dst2, "普通子不能在铁路上转弯"
    assert (6, 0) not in dst2, "山界处a列不通"
    # 1d) 山界必须双向都封死(曾只判 r==5, 导致 e7→e6 反向越界, 红子沿e列滑到e3)
    for c_edge in (0, 4):
        assert (6, c_edge) not in WALK[(5, c_edge)] and (5, c_edge) not in WALK[(6, c_edge)], \
            f"山界: 第{c_edge}列前线双向都不通"
        assert (6, c_edge) not in RAIL.get((5, c_edge), ()) and (5, c_edge) not in RAIL.get((6, c_edge), ()), \
            f"山界: 第{c_edge}列铁路双向都不通"
    for c_bridge in (1, 2, 3):
        assert (6, c_bridge) in WALK[(5, c_bridge)] and (5, c_bridge) in WALK[(6, c_bridge)], \
            f"山界: 第{c_bridge}列是桥, 双向都得通"
    # 实际着法层面: 红营长在 e10 只能滑到 e7 为止, 不得越过前线到 e6/e3
    g1g = Game({}, {}, check=False)
    g1g.cells[(9, 4)] = ("red", "营长")
    d1g = {m[1] for m in g1g.legal_moves("red") if m[0] == (9, 4)}
    assert (6, 4) in d1g, "红营长应能滑到本方前线格 e7"
    assert (5, 4) not in d1g and (2, 4) not in d1g, f"红营长不得越过山界滑进敌方半场: {sorted(d1g)}"
    g1h = Game({}, {}, check=False)
    g1h.cells[(2, 4)] = ("blue", "团长")
    d1h = {m[1] for m in g1h.legal_moves("blue") if m[0] == (2, 4)}
    assert (5, 4) in d1h, "蓝团长应能滑到本方前线格 e6"
    assert (6, 4) not in d1h and (9, 4) not in d1h, f"蓝团长不得越过山界滑进敌方半场: {sorted(d1h)}"
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
    # 8) 地雷情报: 暗棋版只用 belief 排除法 (候选雷位必须覆盖真实雷, 己方雷自己看得见)
    rng8 = random.Random(3)
    g8 = Game(random_layout("blue", rng8), random_layout("red", rng8))
    cand, sitters, mines_side = mine_intel(g8, "red")
    truth_mines = {coord(p) for p, (s, n) in g8.cells.items() if s == "blue" and n == "地雷"}
    assert len(mines_side) == 3, "己方应有3颗地雷(自己看得见)"
    assert truth_mines <= set(cand), f"候选雷位必须覆盖真实雷: {truth_mines} vs {cand}"
    assert len(cand) <= 8, f"后两排非大本营格最多8个: {cand}"
    assert all(p not in cand for p, _ in sitters), "候选雷位与可安全吃的滞留子不应重叠"
    # 9) 布局公平性与合法性: 大本营只布军旗+排长; 雷只在本方后两排; 炸弹不进前线排
    for sd in range(8):
        rg9 = random.Random(sd)
        for s9 in SIDES:
            lay9 = random_layout(s9, rg9)
            assert {lay9[p] for p in lay9 if cell_kind(*p) == "hq"} == {"军旗", "排长"}, \
                "大本营只可布军旗+排长"
            assert sum(1 for n in lay9.values() if n == "地雷") == 3, "每方恰3雷"
            for p9, n9 in lay9.items():
                assert not (n9 == "军旗" and cell_kind(*p9) != "hq"), "军旗必须在大本营"
                assert not (n9 == "地雷" and local_row(p9[0], s9) > 1), "地雷只能在本方后两排"
                assert not (n9 == "炸弹" and local_row(p9[0], s9) == 5), "炸弹不进本方前线排"
                assert sorted(lay9.values()) == sorted(n for n, k in PIECE_COUNTS.items()
                                                      for _ in range(k)), "每方25子齐全"
    # 10) 大本营封格: 腾空后任何子都进不去, 但夺旗(吃营内敌子)仍可行
    g10 = Game({}, {}, check=False)
    g10.cells[(10, 1)] = ("red", "连长")
    assert not any(m[1] in ((11, 1), (11, 3)) for m in g10.legal_moves("red")), "空大本营任何子都不得进入"
    g10.cells[(11, 1)] = ("blue", "军旗")
    assert ((10, 1), (11, 1)) in g10.legal_moves("red"), "夺旗仍须进营吃子"
    # 11) 暗棋选项描述不泄露敌子真名
    g11 = Game({}, {}, check=False)
    g11.cells[(7, 2)] = ("red", "团长")
    g11.cells[(6, 2)] = ("blue", "师长")
    d11 = g11.describe(((6, 2), (7, 2)), view="blue")
    assert "团长" not in d11 and "敌子未明" in d11, f"暗棋选项不得写出敌子真名: {d11}"
    assert g11.describe(((6, 2), (7, 2))) == "师长 c7→c8 吃红团长", "上帝视角仍如实写敌子真名"
    # 12) 双亡只保留两种可能: 同级 或 炸弹 (对掉与炸弹不可分辨)
    g12 = Game({(6, 2): "师长"}, {(7, 2): "师长"}, check=False)
    g12.apply(((6, 2), (7, 2)), "blue")
    for s in ("blue", "red"):
        dead = [set(v) for v in g12.knowledge[s].dead.values()]
        assert dead == [{"师长", "炸弹"}], f"{SIDE_CN[s]}方双亡后应只剩 同级/炸弹 两种可能: {dead}"
    assert "师长" in g12.history[-1][3] and "同级互亡" in g12.history[-1][3], "上帝视角棋谱如实记双亡"
    for s in ("blue", "red"):
        assert "同尽" in g12.history[-1][4][s] and "敌" in g12.history[-1][4][s], \
            f"暗棋历史应写'与敌子同尽'而不是真名: {g12.history[-1][4][s]}"
    # 12b) 低打高(攻方阵亡、守方原位) 必须是"阵亡"而不是"同尽", 且 poss 要按"比我弱"收缩
    g12b = Game({(6, 2): "工兵"}, {(7, 2): "营长"}, check=False)
    g12b.apply(((6, 2), (7, 2)), "blue")
    rec12b = g12b.history[-1]
    assert rec12b[3] == "工兵 c7→c8 攻方阵亡于营长", f"上帝视角: {rec12b[3]}"
    assert "阵亡" in rec12b[4]["blue"] and "同尽" not in rec12b[4]["blue"], \
        f"攻方视角应写阵亡: {rec12b[4]['blue']}"
    assert "阵亡" in rec12b[4]["red"] and "同尽" not in rec12b[4]["red"], \
        f"守方视角应写敌攻方阵亡: {rec12b[4]['red']}"
    kb12b = g12b.knowledge["red"]          # 红方(守方)对蓝方攻子的推断: 阵亡敌子只可能比我弱
    dead12b = [set(v) for v in kb12b.dead.values()]
    assert dead12b == [{"工兵", "排长", "连长"}], f"守方应推出敌攻子只可能比我弱: {dead12b}"
    g13 = Game({(0, 1): "军旗", (0, 3): "排长", (6, 1): "司令"}, {(7, 1): "炸弹"}, check=False)
    g13.apply(((6, 1), (7, 1)), "blue")
    assert g13.flags_public == {"blue": True, "red": False}, "蓝司令阵亡 → 只公开蓝方军旗"
    k13 = g13.knowledge["red"]        # 红方看到蓝军旗已公开 → 蓝方两个大本营身份随之确定
    hq13 = {p: k13.poss[k13.pid_at[p]] for p in sorted(k13.pid_at) if cell_kind(*p) == "hq"}
    assert [set(v) for v in hq13.values()] == [{"军旗"}, {"排长"}], f"公开后大本营身份应确定: {hq13}"
    g13b = Game({(0, 1): "军旗", (0, 3): "排长", (6, 1): "司令"}, {(7, 1): "司令"}, check=False)
    g13b.apply(((6, 1), (7, 1)), "blue")
    assert g13b.flags_public == {"blue": True, "red": True}, "双司令对掉 → 双方军旗同时公开"
    # 14) 敌子动过即排除军旗/地雷
    g14 = Game({(8, 1): "连长"}, {(8, 0): "排长", (10, 2): "连长"}, check=False)
    g14.apply(((8, 0), (7, 0)), "red")
    kb14 = g14.knowledge["blue"]
    poss14 = kb14.poss[kb14.pid_at[(7, 0)]]
    assert "地雷" not in poss14 and "军旗" not in poss14, "移动过的敌子不可能是雷或旗"
    # 15) 排除法剪枝: 敌雷已全部(3颗)被确证阵亡 → 所有敌子 poss 里的地雷都要消失
    for i in range(3):
        kb14.dead[9000 + i] = {"地雷"}
    kb14.prune()
    assert all("地雷" not in p for p in kb14.poss.values()), "雷全灭后应从所有 poss 中剔除地雷"
    assert "地雷" in seed_poss("red", (10, 2)) and "地雷" not in kb14.at((10, 2)), \
        "后两排的雷约束仍在, 但已确证雷全灭 → 活子不再可能是雷"
    # 16) 采样世界: 走子集合与真局面完全一致(暗棋不改变可走性), 但敌子牌面与真相不同
    rng16 = random.Random(5)
    g16 = Game(random_layout("blue", rng16), random_layout("red", rng16))
    ai16 = RuleSearchAI(time_budget=1.0, max_depth=2, samples=3)
    worlds16 = ai16._sample_worlds(g16, "red")
    truth_moves = set(g16.legal_moves("red"))
    for w in worlds16:
        assert set(w.legal_moves("red")) == truth_moves, "采样世界的走子集合必须与真局面一致"
    assert any(any(w.cells[p][1] != g16.cells[p][1]
                   for p, (s, _) in g16.cells.items() if s == "blue") for w in worlds16), \
        "采样世界应与真实敌子牌面存在差异(否则本测试无意义)"
    # 17) 公平性不变性: 把两个敌子的真实棋种对调(同格同排, 布局仍合法)后,
    #     RULE-AI 的选择必须完全不变 —— 它读不到真相。
    def build17(swap):
        rng = random.Random(21)
        b, r = random_layout("blue", rng), random_layout("red", rng)
        if swap:
            cand17 = sorted(p for p, n in b.items()
                            if n in ("师长", "旅长") and 2 <= local_row(p[0], "blue") <= 4)
            assert len(cand17) >= 2, "测试布局里应能找到两个可对调的敌子"
            b[cand17[0]], b[cand17[1]] = b[cand17[1]], b[cand17[0]]
        return Game(b, r)

    g17a, g17b = build17(False), build17(True)
    ai17a = RuleSearchAI(time_budget=60.0, max_depth=3, samples=3)
    ai17b = RuleSearchAI(time_budget=60.0, max_depth=3, samples=3)
    m17a, _ = ai17a.choose(g17a, "red", g17a.legal_moves("red"))
    m17b, _ = ai17b.choose(g17b, "red", g17b.legal_moves("red"))
    assert m17a == m17b, f"敌子真名对调后 AI 决策不应改变: {m17a} vs {m17b}"
    # 18) JEV 的 state 棋盘不得出现敌子真名(只出现 ? 与已确证者)
    rng18 = random.Random(9)
    g18 = Game(random_layout("blue", rng18), random_layout("red", rng18))
    txt18 = JevPlayer(url="http://127.0.0.1:1").state_text(g18, "red")
    board18 = txt18.split("【当前棋盘】\n", 1)[1]
    hidden = 0
    for (r, c), (s, name) in g18.cells.items():
        if s == "blue" and len(g18.knowledge["red"].at((r, c))) > 1:
            hidden += 1
            assert f"蓝{name}" not in board18, f"state 棋盘泄露敌子真名 {name}@{coord((r, c))}"
    assert hidden > 10, "开局时敌子大多不可辨识, 本测试才有意义"
    # 19) 知识库与真实占位表必须严格一致(随机走子全程体检: 敌子都有 pid / pid 都指着敌子 / 无幽灵 pid)
    def belief_ok(g):
        for s in SIDES:
            k = g.knowledge[s]
            live = set(k.pid_at.values())
            for cell, (os_, _name) in g.cells.items():
                if os_ != s and cell not in k.pid_at:
                    return f"{SIDE_CN[s]}方视角下敌子{coord(cell)}没有身份(pid)"
            for cell, pid in k.pid_at.items():
                p = g.cells.get(cell)
                if p is None or p[0] == s:
                    return f"{SIDE_CN[s]}方视角下 pid@{coord(cell)} 处没有敌子"
            if set(k.poss) - live - set(k.dead):
                return f"{SIDE_CN[s]}方视角下有幽灵 pid"
        return ""
    for sd in range(12):
        rg = random.Random(sd)
        g19 = Game(random_layout("blue", rg), random_layout("red", rg))
        for ply19 in range(60):
            s19 = "red" if ply19 % 2 == 0 else "blue"
            mv19 = g19.legal_moves(s19)
            if not mv19 or g19.winner:
                break
            g19.apply(rg.choice(mv19), s19)
            why = belief_ok(g19)
            assert not why, f"seed{sd} 第{ply19 + 1}手后知识库失同步: {why}"
            # 山界: 非工兵任何时候都不得在 a/e 两列跨越第6/7行(工兵可绕道桥后在前线行内走)
            for mv19 in g19.legal_moves(s19):
                (r1, c1), (r2, c2) = mv19
                if {r1, r2} == {5, 6} and (c1 in (0, 4) or c2 in (0, 4)):
                    assert g19.cells[(r1, c1)][1] == "工兵", \
                        f"seed{sd} 第{ply19 + 1}手出现越界着法 {coord((r1, c1))}→{coord((r2, c2))}"
    # 20) 军旗公开后禁入"占位大本营"(防白送子), 但旗营仍可进; 未公开时两营都可进(不泄露旗位)
    g20 = Game({(0, 1): "军旗", (0, 3): "排长"}, {(1, 1): "连长", (1, 3): "排长"}, check=False)
    dst20 = {m[1] for m in g20.legal_moves("red")}
    assert (0, 1) in dst20 and (0, 3) in dst20, "军旗未公开时两个大本营都可进(不能泄露旗在哪座)"
    g20.flags_public["blue"] = True
    dst20b = {m[1] for m in g20.legal_moves("red")}
    assert (0, 1) in dst20b, "旗营仍可进(夺旗)"
    assert (0, 3) not in dst20b, "军旗公开后不得再进占位大本营(白送一子)"
    # 评估函数: 军旗公开后应出现"贴旗/守旗"项
    ev_pub = evaluate(g20, "red")
    g20.flags_public["blue"] = False
    ev_hid = evaluate(g20, "red")
    assert ev_pub != ev_hid, "evaluate 应随军旗公开而改用贴旗项"
    # 20b) 采样世界必须继承"军旗已公开"(否则搜索里的走法集合与真局不一致)
    g20b = Game(random_layout("blue", random.Random(31)), random_layout("red", random.Random(32)))
    g20b.apply(g20b.legal_moves("red")[0], "red")      # 造一点着法史, 让种子确定
    ai20 = RuleSearchAI(time_budget=0.5, max_depth=2, samples=2)
    g20b.flags_public["blue"] = True
    for w20 in ai20._sample_worlds(g20b, "red"):
        assert w20.flags_public == g20b.flags_public, "采样世界应继承军旗公开状态"
        assert set(w20.legal_moves("red")) == set(g20b.legal_moves("red")), "世界走法集合须与真局一致"
    # 20c) 知识库对双方对称: 孪生局(行号镜像 + 红蓝互换)的每一步推断都必须与原局一致
    _MR = lambda cc: (11 - cc[0], cc[1])
    _SW = {"red": "blue", "blue": "red"}

    def _twin(g):
        return Game.from_cells({_MR((r, c)): (_SW[s], n) for (r, c), (s, n) in g.cells.items()})

    def _same_belief(tag, gs, gt):
        for (r, c), (s, _n) in gs.cells.items():
            assert gs.knowledge[s].at((r, c)) == gt.knowledge[_SW[s]].at(_MR((r, c))), \
                f"{tag}: {SIDE_CN[s]}方 {coord((r, c))} 推断不对称"
        for s in SIDES:
            ka, kb = gs.knowledge[s], gt.knowledge[_SW[s]]
            prof = lambda kb2, pids: sorted(sorted(kb2.poss.get(q, ())) for q in pids)
            assert prof(ka, ka.dead) == prof(kb, kb.dead), f"{tag}: {SIDE_CN[s]}方死亡分布不对称"
            assert prof(ka, ka.moved) == prof(kb, kb.moved), f"{tag}: {SIDE_CN[s]}方移动分布不对称"
        assert all(gs.flags_public[k] == gt.flags_public[_SW[k]] for k in gs.flags_public), \
            f"{tag}: 军旗公开不对称"
        assert (gs.quiet_plies, len(gs.history)) == (gt.quiet_plies, len(gt.history)), f"{tag}: 节奏不对称"

    rg20 = random.Random(97)
    gS = Game(random_layout("blue", rg20), random_layout("red", rg20))
    gT = _twin(gS)
    _same_belief("开局", gS, gT)

    def _do(gs, gt, side, mv):
        gs.apply(mv, side)
        gt.apply((_MR(mv[0]), _MR(mv[1])), _SW[side])
        _same_belief(f"{SIDE_CN[side]}走 {coord(mv[0])}->{coord(mv[1])}", gs, gt)

    mv0 = gS.legal_moves("red")[0]
    _do(gS, gT, "red", mv0)                                        # 静默一步
    for mv in gS.legal_moves("blue"):                             # 挑一步有子力阵亡的
        pr = gS.clone_belief(); pr.apply(mv, "blue")
        if len(pr.cells) < len(gS.cells):
            _do(gS, gT, "blue", mv)
            break
    for mv in gS.legal_moves("red"):                              # 挑一步铁路拐弯(强制明牌工兵)
        if is_rail(*mv[0]) and is_rail(*mv[1]) and abs(mv[0][0] - mv[1][0]) == 1 and abs(mv[0][1] - mv[1][1]) == 1:
            _do(gS, gT, "red", mv)
            assert gS.knowledge["blue"].at(mv[1]) == {"工兵"}, \
                f"铁路拐弯必须让观察方明牌工兵: {gS.knowledge['blue'].at(mv[1])}"
            break
    _r20 = random.Random(102)                                     # 再连走 24 手, 每手都比
    for ply in range(24):
        if gS.winner:
            break
        sd = "red" if ply % 2 == 0 else "blue"
        lm, lt = gS.legal_moves(sd), gT.legal_moves(_SW[sd])
        if not lm or not lt:
            break
        assert len(lm) == len(lt), f"孪生局合法着法数应相同: {len(lm)} vs {len(lt)}"
        _do(gS, gT, sd, _r20.choice(lm))
    # 20d) 概率分布随战局持续修正: 双方都要越打越窄, 且永不漏真值
    for seed in (31,):
        rg = random.Random(seed)
        gv = Game(random_layout("blue", rg), random_layout("red", rg))
        ais = {s: RuleSearchAI(0.02, 3, samples=2) for s in SIDES}
        sz = {s: [] for s in SIDES}
        last = {s: {} for s in SIDES}                 # 记住每个敌子上一手的大小, 用来验证"只收窄、不回涨"
        pinned = {s: 0 for s in SIDES}
        for ply in range(120):
            if gv.winner:
                break
            sd = "red" if ply % 2 == 0 else "blue"
            lm = gv.legal_moves(sd)
            if not lm:
                break
            gv.apply(ais[sd].choose(gv, sd, lm)[0], sd)
            for vs in SIDES:                       # 每手之后立刻重算双方可能集
                k = gv.knowledge[vs]
                sizes = [len(k.poss[q]) for q in k.pid_at.values()]
                sz[vs].append(sum(sizes) / max(1, len(sizes)))
                for cell, q in k.pid_at.items():
                    n = len(k.poss[q])
                    if q in last[vs]:
                        assert n <= last[vs][q], f"第{ply}手 {SIDE_CN[vs]}方 {coord(cell)} 可能集反而变大: {last[vs][q]} → {n}"
                    last[vs][q] = n
                    if n == 1:
                        pinned[vs] += 1
                for (rr, cc), (os_, nm) in gv.cells.items():      # 真值永远在可能集里
                    if os_ != vs:
                        assert nm in k.at((rr, cc)), f"第{ply}手泄漏: {SIDE_CN[vs]}方 {coord((rr, cc))} 真值{nm} 不在 {k.at((rr, cc))}"
        for vs in SIDES:
            assert len(sz[vs]) >= 40, f"样本太少: {len(sz[vs])}"
            early, late = sum(sz[vs][:20]) / 20, sum(sz[vs][-20:]) / 20
            assert late < early - 0.3, f"seed{seed} {SIDE_CN[vs]}方分布没收紧: {early:.2f} → {late:.2f}"
            assert pinned[vs] > 0, f"seed{seed} {SIDE_CN[vs]}方没有任何子被钉死成确定棋种"
    # 21) 铁路拐弯 = 工牌: 敌子一旦在铁路上拐弯, 该子身份自动明牌为工兵
    rg21 = random.Random(41)
    g21c = Game(random_layout("blue", rg21), random_layout("red", rg21))
    src21 = next((p for p, (s, n) in sorted(g21c.cells.items())
                  if s == "blue" and is_rail(*p) and n not in ("地雷", "军旗")
                  and all(g21c.cells.get(q) is None for q in ((p[0] + 2, p[1]),) if is_rail(*q))
                  and ((p[0] + 2, p[1]),) in g21c.legal_moves("blue")), None)
    if src21:                                     # 真实布局里: 直线长滑后身份仍应很宽
        g21c.apply((src21, (src21[0] + 2, src21[1])), "blue")
        kk21 = g21c.knowledge["red"]
        pidk = kk21.pid_at.get((src21[0] + 2, src21[1]))
        if pidk is not None:
            assert len(kk21.poss[pidk]) > 3, \
                f"直线长滑不该把身份压缩到 {kk21.poss[pidk]}"
    g21b = Game({(5, 2): "工兵"}, {}, check=False)
    assert ((5, 2), (6, 3)) in g21b.legal_moves("blue"), "工兵应能走 c6→d7(铁路拐弯)"
    g21b.cells[(5, 2)] = ("blue", "连长")
    assert ((5, 2), (6, 3)) not in g21b.legal_moves("blue"), "普通子不能这样走(测试前提)"
    g21b.cells[(5, 2)] = ("blue", "工兵")
    g21b.apply(((5, 2), (6, 3)), "blue")           # c6 → d7: 铁路上拐弯, 只有工兵能走
    kb21 = g21b.knowledge["red"]
    pidb = kb21.pid_at[(6, 3)]
    assert kb21.poss[pidb] == {"工兵"}, f"拐弯者应明牌工兵: {kb21.poss[pidb]}"
    assert "工兵" in g21b.render(view="red"), "红方视角应直接显示该子为工兵"
    # 22) 雷区警惕: 疑似撞雷的损失随子价值放大; 工兵军纪过滤非疑似目标
    g22 = Game({(1, 1): "司令"}, {(0, 0): "排长", (0, 2): "排长", (0, 4): "排长"}, check=False)
    kb22 = g22.knowledge["red"]
    for cell in ((0, 0), (0, 2), (0, 4)):          # 三颗雷都已被"确证阵亡" → 剩下的排长不是雷
        kb22.dead[5000 + cell[1]] = {"地雷"}
    kb22.prune()
    assert "地雷" not in kb22.at((1, 1)), "前排格本来就不可能有雷"
    g23 = Game({(11, 2): "工兵", (10, 1): "司令"}, {(10, 2): "排长"}, check=False)
    assert "地雷" in suspect_tags(g23, "blue", (10, 2)), \
        f"敌方后两排的独苗应疑似地雷: {suspect_tags(g23, 'blue', (10, 2))}"
    kept23 = JevPlayer._sapper_doctrine(g23, "blue", g23.legal_moves("blue"), g23.legal_moves("blue"))
    assert ((11, 2), (10, 2)) in kept23, "雷区里工兵仍可主动挖雷"
    n23 = JevPlayer.safety_note(g23, "blue", ((10, 1), (10, 2)))
    assert "疑似撞雷" in n23 and "100" in n23, f"司令撞疑似雷应按自身价值警示: {n23}"
    g23b = Game({(5, 2): "工兵"}, {(6, 2): "连长"}, check=False)   # 前线, 不可能有雷
    assert not suspect_tags(g23b, "blue", (6, 2)), "前线格不该疑似雷"
    kept23b = JevPlayer._sapper_doctrine(g23b, "blue", g23b.legal_moves("blue"), g23b.legal_moves("blue"))
    assert ((5, 2), (6, 2)) not in kept23b, "工兵军纪: 非疑似雷/弹的目标不许主动打"
    assert kept23b, "过滤后不能把着法删光"
    print("SELFTEST PASS: 铁路滑行/工兵转弯/不可越子/行营保护/地雷与炸弹/同级互亡/夺旗胜/"
          "大本营(只布旗+排长、冻结、封格、旗公开后禁入占位营)/山界/"
          "暗棋(选项不泄露敌名、双亡只留同级或炸弹两种可能、司令阵亡公开军旗、移动排除雷旗、排除法剪雷、"
          "采样世界走法一致且继承旗公开状态、真名对调后决策不变、state不泄露) 全部符合规则")


# ---------------------------------------------------------------- 主程序

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--url", default="http://127.0.0.1:8090", help="JEV 决策服务")
    ap.add_argument("--ai-time", type=float, default=2.0, help="AI 每步搜索时限(秒)")
    ap.add_argument("--ai-depth", type=int, default=4, help="AI 最大搜索深度")
    ap.add_argument("--ai-samples", type=int, default=4,
                    help="RULE-AI 每步从 belief 采样的敌子世界数(暗棋: 越多样本决策越稳, 越吃时间)")
    ap.add_argument("--ai-refine", type=int, default=3,
                    help="信念感知复评的候选手数(走完更新信念再重采样评估; 0 = 关闭)")
    ap.add_argument("--ai-procs", type=int, default=0,
                    help="并行子进程数(CPU 密集, 线程无效; 0 = 串行, 建议 <= CPU核数-1)")
    ap.add_argument("--max-plies", type=int, default=200, help="每局手数上限(达上限按子力判定)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).parent / "games"))
    ap.add_argument("--live-port", type=int, default=8097,
                    help="实时观战服务端口 (上帝视角, 对局中每手实时推演)")
    ap.add_argument("--no-live", action="store_true", help="关闭实时观战服务")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    live = None
    if not a.no_live:
        live = LiveBroadcaster(a.live_port)
        if live.start():
            print(f"实时观战 (上帝视角): {live.url}  ← 对局开始后浏览器打开即可实时看棋", flush=True)
        else:
            print(f"实时观战: 端口 {a.live_port} 被占用, 已跳过 (--no-live 可显式关闭)", flush=True)
            live = None
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    results = [play_game(i, a, log, live=live) for i in range(a.games)]
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
    if live and sys.stdout.isatty():
        print("实时观战页保留 30 分钟 (Ctrl+C 可提前退出)…", flush=True)
        try:
            time.sleep(1800)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
