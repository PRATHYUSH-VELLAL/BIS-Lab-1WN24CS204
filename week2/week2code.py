import random
import numpy as np


# ============================================================
# XOR DATASET
# ============================================================

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# ============================================================
# ACTIVATION FUNCTION
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ============================================================
# NEURAL NETWORK
# Architecture:
# Input Layer  -> 2 neurons
# Hidden Layer -> 3 neurons
# Output Layer -> 1 neuron
# ============================================================

def neural_network(X, weights):

    # --------------------------------------------------------
    # Extract weights and biases from particle position
    # --------------------------------------------------------

    # Input -> Hidden weights
    W1 = weights[0:6].reshape(2, 3)

    # Hidden layer biases
    b1 = weights[6:9].reshape(1, 3)

    # Hidden -> Output weights
    W2 = weights[9:12].reshape(3, 1)

    # Output layer bias
    b2 = weights[12]

    # --------------------------------------------------------
    # Hidden layer
    # --------------------------------------------------------

    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    # --------------------------------------------------------
    # Output layer
    # --------------------------------------------------------

    output_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(output_input)

    return output


# ============================================================
# FITNESS FUNCTION
# ============================================================

def objective_function(weights):

    predictions = neural_network(X, weights)

    # Mean Squared Error
    mse = np.mean((y - predictions) ** 2)

    return mse


# ============================================================
# PARTICLE CLASS
# ============================================================

class Particle:

    def __init__(self, dim, bounds):

        self.dim = dim
        self.bounds = bounds

        # Random position
        self.position = np.random.uniform(
            bounds[0],
            bounds[1],
            dim
        )

        # Random velocity
        velocity_range = abs(bounds[1] - bounds[0]) / 10

        self.velocity = np.random.uniform(
            -velocity_range,
            velocity_range,
            dim
        )

        # Personal best
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def update_position(self):

        # Update position
        self.position += self.velocity

        # Keep position inside bounds
        self.position = np.clip(
            self.position,
            self.bounds[0],
            self.bounds[1]
        )


# ============================================================
# PSO CLASS
# ============================================================

class PSO:

    def __init__(
        self,
        objective_func,
        dim,
        num_particles=30,
        max_iter=100,
        bounds=(-1.0, 1.0),
        w=0.5,
        c1=2.0,
        c2=2.0
    ):

        self.objective_func = objective_func
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds

        # PSO parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Create swarm
        self.particles = [
            Particle(dim, bounds)
            for _ in range(num_particles)
        ]

        # Global best
        self.global_best_position = np.zeros(dim)
        self.global_best_score = float('inf')

    def optimize(self):

        # ====================================================
        # PSO ITERATIONS
        # ====================================================

        for iteration in range(self.max_iter):

            # ------------------------------------------------
            # Evaluate particles
            # ------------------------------------------------

            for particle in self.particles:

                score = self.objective_func(
                    particle.position
                )

                # Update personal best
                if score < particle.best_score:

                    particle.best_score = score
                    particle.best_position = np.copy(
                        particle.position
                    )

                # Update global best
                if score < self.global_best_score:

                    self.global_best_score = score
                    self.global_best_position = np.copy(
                        particle.position
                    )

            # ------------------------------------------------
            # Update velocity and position
            # ------------------------------------------------

            for particle in self.particles:

                # Random values
                r1 = random.random()
                r2 = random.random()

                # Cognitive component
                cognitive = (
                    self.c1
                    * r1
                    * (
                        particle.best_position
                        - particle.position
                    )
                )

                # Social component
                social = (
                    self.c2
                    * r2
                    * (
                        self.global_best_position
                        - particle.position
                    )
                )

                # Velocity update
                particle.velocity = (
                    self.w * particle.velocity
                    + cognitive
                    + social
                )

                # Position update
                particle.update_position()

            # ------------------------------------------------
            # Display progress
            # ------------------------------------------------

            if (
                (iteration + 1) % 10 == 0
                or iteration == 0
            ):

                print(
                    f"Iteration {iteration + 1:3d}/"
                    f"{self.max_iter} | "
                    f"Best MSE: "
                    f"{self.global_best_score:.6f}"
                )

        return (
            self.global_best_position,
            self.global_best_score
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Neural network parameters
    #
    # 2 -> 3:
    # 2 * 3 = 6 weights
    #
    # Hidden biases:
    # 3
    #
    # 3 -> 1:
    # 3 * 1 = 3 weights
    #
    # Output bias:
    # 1
    #
    # Total = 13
    # --------------------------------------------------------

    dimension = 13

    # --------------------------------------------------------
    # Create PSO
    # --------------------------------------------------------

    pso = PSO(
        objective_func=objective_function,
        dim=dimension,
        num_particles=30,
        max_iter=200,
        bounds=(-5.0, 5.0),
        w=0.5,
        c1=2.0,
        c2=2.0
    )

    # --------------------------------------------------------
    # Run PSO
    # --------------------------------------------------------

    best_weights, best_score = pso.optimize()

    # --------------------------------------------------------
    # Final predictions
    # --------------------------------------------------------

    predictions = neural_network(
        X,
        best_weights
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print("OPTIMIZATION FINISHED")
    print("=" * 50)

    print(f"Best MSE: {best_score:.6f}")

    print("\nPredictions:")

    for i in range(len(X)):

        predicted = predictions[i][0]

        print(
            f"Input: {X[i].astype(int)} | "
            f"Actual: {int(y[i][0])} | "
            f"Predicted: {predicted:.4f}"
        )