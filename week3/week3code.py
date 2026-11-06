mport numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# ==================================================
# ANT COLONY OPTIMIZATION FOR ROBOT PATH PLANNING
# Graph representation + convergence analysis
# ==================================================

random.seed(42)
np.random.seed(42)

# Environment
ROWS, COLS = 12, 12
START = (0, 0)
GOAL = (11, 11)

# ACO parameters
NUM_ANTS = 30
ITERATIONS = 100
ALPHA = 1.0
BETA = 3.0
EVAPORATION = 0.30
Q = 100.0

# Exploration probability: encourages alternative routes
EXPLORATION_RATE = 0.15

# ==================================================
# 1. CREATE GRID AND OBSTACLES
# ==================================================

grid = np.zeros((ROWS, COLS), dtype=int)

obstacles = [
    (1, 2), (2, 2), (3, 2), (4, 2),
    (6, 2), (7, 2), (8, 2), (9, 2),
    (2, 4), (3, 4), (4, 4), (5, 4),
    (7, 4), (8, 4), (9, 4), (10, 4),
    (0, 6), (1, 6), (2, 6), (3, 6),
    (5, 6), (6, 6), (7, 6), (8, 6),
    (3, 8), (4, 8), (5, 8), (6, 8),
    (8, 8), (9, 8), (10, 8),
    (1, 10), (2, 10), (3, 10),
    (5, 10), (6, 10), (7, 10)
]

for cell in obstacles:
    grid[cell] = 1

grid[START] = 0
grid[GOAL] = 0

# ==================================================
# 2. CONSTRUCT GRAPH G = (V, E)
# ==================================================

nodes = [
    (r, c)
    for r in range(ROWS)
    for c in range(COLS)
    if grid[r, c] == 0
]

graph = {node: [] for node in nodes}

MOVES = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1)
]

for r, c in nodes:
    for dr, dc in MOVES:
        neighbor = (r + dr, c + dc)

        if neighbor in graph:
            graph[(r, c)].append(neighbor)

# Pheromone is associated with directed graph edges
pheromone = defaultdict(lambda: 1.0)

# ==================================================
# 3. HEURISTIC FUNCTION
# ==================================================

def heuristic(node):
    """Estimate desirability based on distance to the goal."""
    distance = (
        abs(node[0] - GOAL[0])
        + abs(node[1] - GOAL[1])
    )
    return 1.0 / (distance + 1e-6)


# ==================================================
# 4. CONSTRUCT A PATH FOR ONE ANT
# ==================================================

def construct_path():
    current = START
    path = [START]
    visited = {START}

    while current != GOAL:

        candidates = [
            neighbor
            for neighbor in graph[current]
            if neighbor not in visited
        ]

        # Ant has reached a dead end
        if not candidates:
            return None

        # Occasionally explore a random valid neighbor
        if random.random() < EXPLORATION_RATE:
            next_node = random.choice(candidates)

        else:
            weights = []

            for neighbor in candidates:
                edge = (current, neighbor)

                tau = pheromone[edge] ** ALPHA
                eta = heuristic(neighbor) ** BETA

                weights.append(tau * eta)

            weights = np.asarray(weights, dtype=float)
            total = weights.sum()

            if total <= 0 or not np.isfinite(total):
                probabilities = np.ones(
                    len(candidates)
                ) / len(candidates)
            else:
                probabilities = weights / total

            next_node = random.choices(
                candidates,
                weights=probabilities,
                k=1
            )[0]

        path.append(next_node)
        visited.add(next_node)
        current = next_node

        # Prevent excessively long paths
        if len(path) > len(nodes):
            return None

    return path


# ==================================================
# 5. UPDATE PHEROMONES
# ==================================================

def update_pheromones(paths, best_path):
    # Evaporate pheromones on all edges
    for node in graph:
        for neighbor in graph[node]:
            edge = (node, neighbor)
            pheromone[edge] *= (1.0 - EVAPORATION)

    # Reinforce successful paths
    for path in paths:
        length = len(path) - 1
        deposit = Q / max(length, 1)

        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            pheromone[edge] += deposit

    # Extra reinforcement for the best path found
    if best_path is not None:
        length = len(best_path) - 1
        deposit = Q / max(length, 1)

        for i in range(len(best_path) - 1):
            edge = (best_path[i], best_path[i + 1])
            pheromone[edge] += deposit

    # Limit values to maintain numerical stability
    for node in graph:
        for neighbor in graph[node]:
            edge = (node, neighbor)
            pheromone[edge] = np.clip(
                pheromone[edge], 1e-6, 1e6
            )


# ==================================================
# 6. RUN ACO AND RECORD CONVERGENCE
# ==================================================

def run_aco():
    best_path = None
    best_length = float("inf")

    best_history = []
    average_history = []
    explored_paths = []

    for iteration in range(ITERATIONS):

        paths = []
        lengths = []

        for _ in range(NUM_ANTS):
            path = construct_path()

            if path is not None:
                length = len(path) - 1

                paths.append(path)
                lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_path = path.copy()

        # Record the final iteration's successful paths
        explored_paths = paths.copy()

        update_pheromones(paths, best_path)

        # Best-so-far path length
        if best_path is not None:
            best_history.append(best_length)
        else:
            best_history.append(np.nan)

        # Mean path length of successful ants in this iteration
        if lengths:
            average_history.append(float(np.mean(lengths)))
        else:
            average_history.append(np.nan)

        if (iteration + 1) % 10 == 0:
            print(
                f"Iteration {iteration + 1:3d} | "
                f"Best = {best_history[-1]} | "
                f"Average = {average_history[-1]:.2f}",
                flush=True
            )

    return (
        best_path,
        best_length,
        best_history,
        average_history,
        explored_paths
    )


# ==================================================
# 7. DRAW THE GRAPH ENVIRONMENT
# ==================================================

def draw_environment(ax, best_path, explored_paths):
    # Draw valid graph edges
    for node in graph:
        for neighbor in graph[node]:
            if node < neighbor:
                ax.plot(
                    [node[1], neighbor[1]],
                    [node[0], neighbor[0]],
                    color="lightgray",
                    linewidth=0.6,
                    zorder=1
                )

    # Draw obstacles
    for r, c in obstacles:
        ax.scatter(
            c, r,
            color="black",
            marker="s",
            s=170,
            zorder=3
        )

    # Draw free nodes
    ax.scatter(
        [c for r, c in nodes],
        [r for r, c in nodes],
        color="steelblue",
        s=12,
        label="Free nodes",
        zorder=2
    )

    # Draw sample ant paths from the final iteration
    for path in explored_paths[:12]:
        ax.plot(
            [p[1] for p in path],
            [p[0] for p in path],
            color="orange",
            linewidth=0.8,
            alpha=0.35,
            zorder=2
        )

    # Highlight best path
    if best_path is not None:
        ax.plot(
            [p[1] for p in best_path],
            [p[0] for p in best_path],
            color="red",
            linewidth=2.8,
            marker="o",
            markersize=3,
            label="Best ACO path",
            zorder=4
        )

    # Start and goal
    ax.scatter(
        START[1], START[0],
        color="limegreen",
        marker="s",
        s=180,
        edgecolors="black",
        label="Start",
        zorder=5
    )

    ax.scatter(
        GOAL[1], GOAL[0],
        color="blue",
        marker="*",
        s=230,
        edgecolors="black",
        label="Goal",
        zorder=5
    )

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_aspect("equal")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_title("ACO Graph-Based Robot Path Planning")
    ax.legend(loc="upper left", fontsize=8)


# ==================================================
# 8. PLOT PATH AND CONVERGENCE CURVES
# ==================================================

def plot_results(
    best_path,
    best_length,
    best_history,
    average_history,
    explored_paths
):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 7)
    )

    # Left graph: robot environment
    draw_environment(
        axes[0],
        best_path,
        explored_paths
    )

    # Right graph: convergence
    iterations = range(1, ITERATIONS + 1)

    axes[1].plot(
        iterations,
        best_history,
        color="red",
        linewidth=2,
        label="Best-so-far path length"
    )

    axes[1].plot(
        iterations,
        average_history,
        color="blue",
        linewidth=1.4,
        alpha=0.85,
        label="Average path length"
    )

    axes[1].set_title("ACO Convergence Analysis")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Path length (grid moves)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save before showing
    filename = "aco_graph_results.png"

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"\nGraph saved as: {filename}")

    plt.show()


# ==================================================
# 9. MAIN PROGRAM
# ==================================================

if __name__ == "__main__":

    print("Starting ACO robot path planning...\n")

    print("Graph nodes:", len(nodes))
    print(
        "Undirected graph edges:",
        sum(len(v) for v in graph.values()) // 2
    )

    (
        best_path,
        best_length,
        best_history,
        average_history,
        explored_paths
    ) = run_aco()

    print("\n========== FINAL RESULTS ==========")

    if best_path is not None:
        print("Status: Feasible path found")
        print("Start:", START)
        print("Goal:", GOAL)
        print("Best path length:", best_length)
        print("Number of cells in path:", len(best_path))
        print("Path coordinates:", best_path)
    else:
        print("No feasible path found.")

    plot_results(
        best_path,
        best_length,
        best_history,
        average_history,
        explored_paths
    )