import numpy as np

# 1. Smart grid data
np.random.seed(7)

# Electricity price for 6 time slots
price = np.array([3, 3, 5, 8, 10, 6])

# Normal electricity demand
base = np.array([2, 2, 3, 5, 6, 3])

# Solar energy availability
solar = np.array([0, 0, 1, 2, 1, 0])

# Power consumed by 5 appliances
appliance = np.array([1.0, 1.5, 2.0, 1.0, 2.5])

# 2. GWO parameters
n_wolves = 20
iterations = 100
dim = len(appliance)

lb, ub = 0, 5

wolves = np.random.uniform(
    lb, ub, (n_wolves, dim)
)

# 3. Fitness function
def evaluate(position):

    slots = np.clip(
        np.rint(position).astype(int), lb, ub
    )

    grid = base - solar

    for i, slot in enumerate(slots):
        grid[slot] += appliance[i]

    cost = np.sum(grid * price)

    peak_penalty = 2.0 * np.max(grid)**2

    fitness = cost + peak_penalty

    return fitness, cost, np.max(grid), slots, grid


# 4. Grey Wolf Optimization
for t in range(iterations):

    scores = np.array([
        evaluate(w)[0] for w in wolves
    ])

    order = np.argsort(scores)

    alpha = wolves[order[0]].copy()
    beta = wolves[order[1]].copy()
    delta = wolves[order[2]].copy()

    a = 2 - 2 * t / iterations

    for i in range(n_wolves):

        new_positions = []

        for leader in (alpha, beta, delta):

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A = 2 * a * r1 - a
            C = 2 * r2

            D = np.abs(C * leader - wolves[i])

            new_positions.append(
                leader - A * D
            )

        wolves[i] = np.clip(
            np.mean(new_positions, axis=0),
            lb, ub
        )


# 5. Final optimized result
best = evaluate(alpha)

# Original appliance schedule
baseline = evaluate(np.array([3, 4, 4, 3, 4]))

print("Baseline grid demand:", baseline[4])
print("Baseline electricity cost:", baseline[1])
print("Baseline peak demand:", baseline[2])

print("\nOptimized appliance time slots:", best[3] + 1)
print("Optimized grid demand:", best[4])
print("Optimized electricity cost:", best[1])
print("Optimized peak demand:", best[2])