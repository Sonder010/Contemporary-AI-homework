from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random
from typing import Callable, Iterable, Sequence

Position = tuple[int, int]
Direction = tuple[int, int]

NEIGHBOR_ORDERS: dict[str, tuple[Direction, ...]] = {
    "up_right_down_left": ((-1, 0), (0, 1), (1, 0), (0, -1)),
    "down_left_up_right": ((1, 0), (0, -1), (-1, 0), (0, 1)),
    "right_left_down_up": ((0, 1), (0, -1), (1, 0), (-1, 0)),
}


@dataclass(frozen=True)
class SearchResult:
    path: list[Position]
    cost: int
    expanded: int
    generated: int
    max_frontier: int


@dataclass(frozen=True)
class SearchMetrics:
    nodes: int = 0
    leaves: int = 0
    cutoffs: int = 0


@dataclass
class TreeNode:
    value: int | float | None = None
    children: list["TreeNode"] | None = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []


def _validate_grid(grid: Sequence[Sequence[object]]) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid rows must have equal length")
    return len(grid), width


def _is_obstacle(cell: object) -> bool:
    return str(cell).upper() == "X"


def _entry_cost(cell: object, zero_goal_cost: bool = False) -> int:
    text = str(cell)
    if text.upper() == "X":
        raise ValueError("obstacles cannot be entered")
    if text.upper() == "E" and zero_goal_cost:
        return 0
    if text.upper() == "E":
        return 1
    if text in {"S", "A", "B"} or text.startswith("T"):
        return 1
    if text == ".":
        return 1
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"unsupported map cell: {text!r}") from exc
    if value < 0:
        raise ValueError("terrain costs must be non-negative")
    return value


def _neighbors(
    node: Position,
    rows: int,
    cols: int,
    neighbor_order: Sequence[Direction] = NEIGHBOR_ORDERS["up_right_down_left"],
) -> Iterable[Position]:
    row, col = node
    if len(neighbor_order) != 4 or len(set(neighbor_order)) != 4:
        raise ValueError("neighbor_order must contain four unique directions")
    if any(abs(dr) + abs(dc) != 1 for dr, dc in neighbor_order):
        raise ValueError("neighbor_order must contain cardinal directions only")
    for dr, dc in neighbor_order:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def zero_heuristic(_: Position, __: Position) -> int:
    return 0


def manhattan_heuristic(node: Position, goal: Position) -> int:
    return _manhattan(node, goal)


def goal_zero_heuristic(
    grid: Sequence[Sequence[object]], goal: Position
) -> Callable[[Position, Position], int]:
    """Admissible bound when entering E costs zero.

    Every move except the final one enters a non-goal cell with cost at least
    c_min. Therefore (d - 1) * c_min is a lower bound for a path of d moves.
    """

    costs = [
        _entry_cost(cell, zero_goal_cost=True)
        for row in grid
        for cell in row
        if not _is_obstacle(cell) and str(cell).upper() != "E"
    ]
    minimum = min(costs, default=1)
    minimum = max(1, minimum)

    def heuristic(node: Position, target: Position) -> int:
        del target
        distance = _manhattan(node, goal)
        return max(0, distance - 1) * minimum

    return heuristic


def _reconstruct_path(
    parents: dict[Position, Position | None], goal: Position
) -> list[Position]:
    path: list[Position] = []
    node: Position | None = goal
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    return path


def astar(
    grid: Sequence[Sequence[object]],
    start: Position,
    goal: Position,
    heuristic: Callable[[Position, Position], int] = manhattan_heuristic,
    zero_goal_cost: bool = False,
    neighbor_order: Sequence[Direction] = NEIGHBOR_ORDERS["up_right_down_left"],
) -> SearchResult:
    """Run A* and return the path and instrumented search statistics."""

    rows, cols = _validate_grid(grid)
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError("start is outside the grid")
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        raise ValueError("goal is outside the grid")
    if _is_obstacle(grid[start[0]][start[1]]):
        return SearchResult([], -1, 0, 0, 0)
    if _is_obstacle(grid[goal[0]][goal[1]]):
        return SearchResult([], -1, 0, 0, 0)
    if start == goal:
        return SearchResult([start], 0, 1, 1, 1)

    counter = 0
    frontier: list[tuple[int, int, Position]] = []
    heapq.heappush(frontier, (heuristic(start, goal), counter, start))
    parents: dict[Position, Position | None] = {start: None}
    best_g: dict[Position, int] = {start: 0}
    closed: set[Position] = set()
    expanded = 0
    generated = 1
    max_frontier = 1

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1

        if node == goal:
            return SearchResult(
                _reconstruct_path(parents, goal),
                best_g[goal],
                expanded,
                generated,
                max_frontier,
            )

        for neighbor in _neighbors(node, rows, cols, neighbor_order):
            if _is_obstacle(grid[neighbor[0]][neighbor[1]]):
                continue
            new_g = best_g[node] + _entry_cost(
                grid[neighbor[0]][neighbor[1]],
                zero_goal_cost=zero_goal_cost,
            )
            if new_g >= best_g.get(neighbor, math.inf):
                continue
            best_g[neighbor] = new_g
            parents[neighbor] = node
            generated += 1
            counter += 1
            priority = new_g + heuristic(neighbor, goal)
            heapq.heappush(frontier, (priority, counter, neighbor))
            max_frontier = max(max_frontier, len(frontier))

    return SearchResult([], -1, expanded, generated, max_frontier)


def shortest_maze_path(
    grid: Sequence[Sequence[object]], start: Position, goal: Position
) -> list[Position]:
    return astar(grid, start, goal, manhattan_heuristic).path


# 此函数对应题目二加权地形
def weighted_path_cost(
    grid: Sequence[Sequence[object]], start: Position, goal: Position
) -> int:
    heuristic = goal_zero_heuristic(grid, goal)
    return astar(grid, start, goal, heuristic, zero_goal_cost=True).cost


def build_sample_tree() -> TreeNode:
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    # The supplied construction intentionally reuses max2 in both MIN branches.
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


class _TreeOrder:
    def __init__(self, mode: str, seed: int = 0) -> None:
        self.mode = mode
        self.rng = random.Random(seed)

    @staticmethod
    def _leaf_bounds(node: TreeNode) -> tuple[float, float]:
        values: list[float] = []

        def visit(current: TreeNode) -> None:
            if not current.children:
                if current.value is None:
                    raise ValueError("leaf node has no value")
                values.append(float(current.value))
                return
            for child in current.children:
                visit(child)

        visit(node)
        return min(values), max(values)

    def children(self, node: TreeNode, maximizing: bool) -> list[TreeNode]:
        children = list(node.children or [])
        if self.mode == "left_to_right":
            return children
        if self.mode == "right_to_left":
            return list(reversed(children))
        if self.mode == "random":
            self.rng.shuffle(children)
            return children
        if self.mode == "oracle":
            def key(child: TreeNode) -> float:
                low, high = self._leaf_bounds(child)
                average = (low + high) / 2
                return -average if maximizing else average

            return sorted(children, key=key)
        raise ValueError(f"unknown tree order: {self.mode}")


def tree_search(
    root: TreeNode,
    algorithm: str = "alpha_beta",
    order: str = "left_to_right",
    seed: int = 0,
) -> tuple[float, SearchMetrics]:
    """Search a tree whose root is a MAX node."""

    if algorithm not in {"minimax", "alpha_beta"}:
        raise ValueError("algorithm must be minimax or alpha_beta")
    orderer = _TreeOrder(order, seed)
    metrics = {"nodes": 0, "leaves": 0, "cutoffs": 0}

    def search(
        node: TreeNode, maximizing: bool, alpha: float, beta: float
    ) -> float:
        metrics["nodes"] += 1
        if not node.children:
            if node.value is None:
                raise ValueError("leaf node has no value")
            metrics["leaves"] += 1
            return float(node.value)

        children = orderer.children(node, maximizing)
        if maximizing:
            value = -math.inf
            for child in children:
                value = max(value, search(child, False, alpha, beta))
                if algorithm == "alpha_beta":
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        metrics["cutoffs"] += 1
                        break
            return value

        value = math.inf
        for child in children:
            value = min(value, search(child, True, alpha, beta))
            if algorithm == "alpha_beta":
                beta = min(beta, value)
                if alpha >= beta:
                    metrics["cutoffs"] += 1
                    break
        return value

    score = search(root, True, -math.inf, math.inf)
    return score, SearchMetrics(**metrics)


TREASURE_MAP: tuple[tuple[object, ...], ...] = (
    ("A", "1", "1", "X", "2", "1", "T1"),
    ("1", "X", "6", "X", "1", "X", "1"),
    ("1", "1", "T2", "1", "1", "1", "1"),
    ("X", "1", "X", "8", "X", "1", "X"),
    ("1", "1", "1", "1", "T3", "1", "1"),
    ("1", "X", "9", "X", "1", "X", "1"),
    ("T4", "1", "1", "X", "1", "1", "B"),
)

TREASURE_VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
TREASURE_POSITIONS: dict[str, Position] = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}


@dataclass(frozen=True)
class GameState:
    a_pos: Position
    b_pos: Position
    remaining: tuple[str, ...]
    score_a: int
    score_b: int
    current_player: str


@dataclass(frozen=True)
class GameSolution:
    best_target: str | None
    best_value: float
    metrics: SearchMetrics


def compute_distance_matrix(
    grid: Sequence[Sequence[object]] = TREASURE_MAP,
    positions: Sequence[str] = ("A", "B", "T1", "T2", "T3", "T4"),
) -> tuple[list[list[int]], int]:
    matrix: list[list[int]] = []
    total_expanded = 0
    for source_name in positions:
        row: list[int] = []
        source = TREASURE_POSITIONS[source_name]
        for target_name in positions:
            target = TREASURE_POSITIONS[target_name]
            result = astar(
                grid,
                source,
                target,
                goal_zero_heuristic(grid, target),
            )
            total_expanded += result.expanded
            row.append(result.cost)
        matrix.append(row)
    return matrix, total_expanded


def _game_order(
    state: GameState,
    distances: dict[tuple[Position, Position], int],
    mode: str,
) -> tuple[str, ...]:
    remaining = list(state.remaining)
    if mode == "natural":
        return tuple(remaining)
    if mode == "reverse":
        return tuple(reversed(remaining))
    if mode == "value_desc":
        return tuple(sorted(remaining, key=lambda name: -TREASURE_VALUES[name]))
    if mode == "best_proxy":
        player_pos = state.a_pos if state.current_player == "A" else state.b_pos
        key = lambda name: TREASURE_VALUES[name] - distances[
            (player_pos, TREASURE_POSITIONS[name])
        ]
        return tuple(sorted(remaining, key=key, reverse=state.current_player == "A"))
    raise ValueError(f"unknown game order: {mode}")


def solve_treasure_game(
    depth: int,
    algorithm: str = "alpha_beta",
    order: str = "natural",
    distances: dict[tuple[Position, Position], int] | None = None,
) -> GameSolution:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    if algorithm not in {"minimax", "alpha_beta"}:
        raise ValueError("algorithm must be minimax or alpha_beta")
    if distances is None:
        distance_matrix, _ = compute_distance_matrix()
        names = ("A", "B", "T1", "T2", "T3", "T4")
        distances = {
            (TREASURE_POSITIONS[source], TREASURE_POSITIONS[target]): distance
            for i, source in enumerate(names)
            for j, target in enumerate(names)
            for distance in [distance_matrix[i][j]]
        }

    initial = GameState(
        a_pos=TREASURE_POSITIONS["A"],
        b_pos=TREASURE_POSITIONS["B"],
        remaining=("T1", "T2", "T3", "T4"),
        score_a=0,
        score_b=0,
        current_player="A",
    )
    metrics = {"nodes": 1, "leaves": 0, "cutoffs": 0}

    def transition(state: GameState, treasure: str) -> GameState:
        player_pos = state.a_pos if state.current_player == "A" else state.b_pos
        travel = distances[(player_pos, TREASURE_POSITIONS[treasure])]
        if travel < 0:
            raise ValueError(f"treasure {treasure} is unreachable")
        gain = TREASURE_VALUES[treasure] - travel
        remaining = tuple(name for name in state.remaining if name != treasure)
        if state.current_player == "A":
            return GameState(
                TREASURE_POSITIONS[treasure],
                state.b_pos,
                remaining,
                state.score_a + gain,
                state.score_b,
                "B",
            )
        return GameState(
            state.a_pos,
            TREASURE_POSITIONS[treasure],
            remaining,
            state.score_a,
            state.score_b + gain,
            "A",
        )

    def search(
        state: GameState, remaining_depth: int, alpha: float, beta: float
    ) -> float:
        metrics["nodes"] += 1
        if remaining_depth == 0 or not state.remaining:
            metrics["leaves"] += 1
            return state.score_a - state.score_b

        maximizing = state.current_player == "A"
        moves = _game_order(state, distances, order)
        best = -math.inf if maximizing else math.inf
        for treasure in moves:
            value = search(
                transition(state, treasure),
                remaining_depth - 1,
                alpha,
                beta,
            )
            if maximizing:
                best = max(best, value)
                if algorithm == "alpha_beta":
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        metrics["cutoffs"] += 1
                        break
            else:
                best = min(best, value)
                if algorithm == "alpha_beta":
                    beta = min(beta, best)
                    if alpha >= beta:
                        metrics["cutoffs"] += 1
                        break
        return best

    root_best = -math.inf
    root_target: str | None = None
    alpha = -math.inf
    moves = _game_order(initial, distances, order)
    for treasure in moves:
        value = search(transition(initial, treasure), depth - 1, alpha, math.inf)
        if value > root_best:
            root_best = value
            root_target = treasure
        if algorithm == "alpha_beta":
            alpha = max(alpha, root_best)

    return GameSolution(root_target, root_best, SearchMetrics(**metrics))


def distance_lookup(
    matrix: Sequence[Sequence[int]],
    names: Sequence[str] = ("A", "B", "T1", "T2", "T3", "T4"),
) -> dict[tuple[Position, Position], int]:
    return {
        (TREASURE_POSITIONS[source], TREASURE_POSITIONS[target]): matrix[i][j]
        for i, source in enumerate(names)
        for j, target in enumerate(names)
    }
