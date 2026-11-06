
import random
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 2: Initialize GA parameters
population_size = 20
mutation_rate = 0.02
crossover_rate = 0.8
generations = 30

num_features = X.shape[1]

# Step 3: Generate random population
def generate_population():
    population = []
    for i in range(population_size):
        chromosome = [
            random.randint(0, 1) for _ in range(num_features)
        ]
        if sum(chromosome) == 0:
            chromosome[random.randrange(num_features)] = 1
        population.append(chromosome)
    return population

# Step 4: Fitness function using SVM
def fitness(chromosome):
    selected = np.where(np.array(chromosome) == 1)[0]

    if len(selected) == 0:
        return 0

    model = SVC(kernel='linear')
    model.fit(X_train[:, selected], y_train)

    prediction = model.predict(X_test[:, selected])
    accuracy = accuracy_score(y_test, prediction)

    # Reward accuracy and fewer selected features
    score = accuracy - 0.01 * (
        len(selected) / num_features
    )

    return score

# Step 5: Select best parents
def select_parent(population, fitness_values):
    index = random.sample(range(len(population)), 3)
    winner = max(index, key=lambda i: fitness_values[i])
    return population[winner]

# Step 6: Crossover
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, num_features - 1)

        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        return child1, child2

    return parent1[:], parent2[:]

# Step 7: Mutation
def mutate(chromosome):
    for i in range(num_features):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]

    if sum(chromosome) == 0:
        chromosome[random.randrange(num_features)] = 1

    return chromosome

# Step 8: Main Genetic Algorithm
population = generate_population()

for generation in range(generations):

    fitness_values = [
        fitness(chromosome) for chromosome in population
    ]

    best_index = np.argmax(fitness_values)
    best_chromosome = population[best_index][:]
    best_fitness = fitness_values[best_index]

    offspring = []

    while len(offspring) < population_size:

        parent1 = select_parent(population, fitness_values)
        parent2 = select_parent(population, fitness_values)

        child1, child2 = crossover(parent1, parent2)

        offspring.append(mutate(child1))
        if len(offspring) < population_size:
            offspring.append(mutate(child2))

    # Preserve best chromosome (elitism)
    offspring[0] = best_chromosome

    population = offspring

    print(
        "Generation:", generation + 1,
        "Best Fitness:", round(best_fitness, 4)
    )

# Step 9: Display final result
fitness_values = [fitness(c) for c in population]

best_index = np.argmax(fitness_values)
best_chromosome = population[best_index]

selected_features = np.where(
    np.array(best_chromosome) == 1
)[0]

final_model = SVC(kernel='linear')
final_model.fit(X_train[:, selected_features], y_train)

prediction = final_model.predict(X_test[:, selected_features])
final_accuracy = accuracy_score(y_test, prediction)

print("\nFINAL RESULTS")
print("-------------------------")
print("Best Chromosome:", best_chromosome)
print("Selected Features:", selected_features.tolist())
print("Number of Selected Features:", len(selected_features))
print("Classification Accuracy:", round(final_accuracy * 100, 2), "%")
print("Best Fitness:", round(max(fitness_values), 4))
