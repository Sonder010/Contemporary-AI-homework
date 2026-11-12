"""Deterministic checks for the supplied test cases."""

from experiment2 import (
    NEIGHBOR_ORDERS,
    TREASURE_MAP,
    astar,
    build_sample_tree,
    compute_distance_matrix,
    distance_lookup,
    goal_zero_heuristic,
    manhattan_heuristic,
    solve_treasure_game,
    tree_search,
    weighted_path_cost,
)

MAZE = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]

WEIGHTED = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]


def main() -> None:
    result = astar(MAZE, (0, 0), (3, 4), manhattan_heuristic)
    assert len(result.path) == 8
    assert result.path[0] == (0, 0)
    assert result.path[-1] == (3, 4)
    assert result.cost == 7
    assert weighted_path_cost(WEIGHTED, (0, 0), (4, 4)) == 7
    for order in NEIGHBOR_ORDERS.values():
        result = astar(
            WEIGHTED,
            (0, 0),
            (4, 4),
            goal_zero_heuristic(WEIGHTED, (4, 4)),
            zero_goal_cost=True,
            neighbor_order=order,
        )
        assert result.cost == 7
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (4, 4)
    assert tree_search(build_sample_tree(), "minimax")[0] == 5
    assert tree_search(build_sample_tree(), "alpha_beta")[0] == 5

    matrix, _ = compute_distance_matrix()
    distances = distance_lookup(matrix)
    h2 = solve_treasure_game(2, "alpha_beta", "natural", distances)
    h4 = solve_treasure_game(4, "alpha_beta", "natural", distances)
    assert (h2.best_target, h2.best_value) == ("T2", -1)
    assert (h4.best_target, h4.best_value) == ("T3", 3)

    # Minimax and alpha-beta must agree under every tested order.
    for depth in (2, 4):
        for order in ("natural", "value_desc", "best_proxy", "reverse"):
            plain = solve_treasure_game(depth, "minimax", order, distances)
            pruned = solve_treasure_game(depth, "alpha_beta", order, distances)
            assert plain.best_value == pruned.best_value

    assert (
        astar(
            WEIGHTED,
            (0, 0),
            (4, 4),
            goal_zero_heuristic(WEIGHTED, (4, 4)),
            zero_goal_cost=True,
        ).cost
        == 7
    )
    assert weighted_path_cost(TREASURE_MAP, (0, 0), (0, 6)) >= 0
    print("All supplied and cross-check tests passed.")


if __name__ == "__main__":
    main()
