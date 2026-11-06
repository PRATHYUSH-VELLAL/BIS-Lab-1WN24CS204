import numpy as np
import math
# --------------------------------------------------
# 1. Mechanical shaft design parameters
# --------------------------------------------------
T = 500.0               # Torque (N.m)
L = 1.0                 # Shaft length (m)
rho = 7850.0            # Density (kg/m^3)
G = 80e9                # Shear modulus (Pa)
tau_max = 80e6          # Allowable shear stress (Pa)
theta_max = 0.02        # Allowable twist (rad)

lower = 0.020           # Minimum diameter (m)
upper = 0.080           # Maximum diameter (m)

# --------------------------------------------------
# 2. Objective and constraint functions
# --------------------------------------------------
def evaluate(d):
    # Shaft mass
    mass = rho * (np.pi / 4) * d**2 * L

    # Maximum torsional shear stress
    tau = 16 * T / (np.pi * d**3)

    # Angle of twist
    theta = 32 * T * L / (np.pi * G * d**4)

    # Normalized constraint violations
    g1 = tau / tau_max - 1.0
    g2 = theta / theta_max - 1.0

    violation = max(0.0, g1) + max(0.0, g2)

    return mass, tau, theta, violation


# Feasible designs are always preferred.
# Among feasible designs, minimize mass.
def score(d):
    mass, _, _, violation = evaluate(d)
    return (violation, mass)


def is_better(d1, d2):
    return score(d1) < score(d2)


# --------------------------------------------------
# 3. Generate Levy-flight steps (Mantegna method)
# --------------------------------------------------

def levy_flight(beta=1.5):
    sigma_u = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (
            math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)

    return u / (abs(v) ** (1 / beta) + 1e-12)



# --------------------------------------------------
# 4. Cuckoo Search parameters
# --------------------------------------------------
np.random.seed(42)

n = 25                  # Number of nests
pa = 0.25                # Abandonment probability
iterations = 500
alpha = 0.01 * (upper - lower)

# Initialize random candidate shaft diameters
nests = np.random.uniform(lower, upper, n)

best = min(nests, key=score)
history = []

# --------------------------------------------------
# 5. Main Cuckoo Search loop
# --------------------------------------------------
for t in range(iterations):

    # Generate new solutions using Levy flights
    for i in range(n):
        step = levy_flight()

        candidate = nests[i] + alpha * step
        candidate = np.clip(candidate, lower, upper)

        # Select a random nest
        j = np.random.randint(n)

        # Replace if the candidate is better
        if is_better(candidate, nests[j]):
            nests[j] = candidate

    # Abandon a fraction of nests and introduce
    # new random candidates
    abandon = np.random.random(n) < pa

    for i in np.where(abandon)[0]:
        candidate = np.random.uniform(lower, upper)

        if is_better(candidate, nests[i]):
            nests[i] = candidate

    # Preserve the best solution found so far
    current_best = min(nests, key=score)

    if is_better(current_best, best):
        best = current_best

    history.append(evaluate(best)[0])

# --------------------------------------------------
# 6. Display the final design
# --------------------------------------------------
mass, tau, theta, violation = evaluate(best)

print("CUCKOO SEARCH: SHAFT DESIGN OPTIMIZATION")
print("----------------------------------------")
print(f"Optimal diameter : {best * 1000:.3f} mm")
print(f"Minimum mass     : {mass:.4f} kg")
print(f"Shear stress     : {tau / 1e6:.3f} MPa")
print(f"Angle of twist   : {theta:.6f} rad")
print(f"Feasible design  : {violation < 1e-9}")
print(f"Iterations       : {iterations}")