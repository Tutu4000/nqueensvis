import random
import numpy as np
from tabulate import tabulate
from datetime import datetime
from typing import List, Tuple, Set, Optional
import json

def calculate_fitness(solution, N):
    chromosome = solution["chromosome"]
    
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Coluna
            if chromosome[i] == chromosome[j]:
                conflicts += 1
            # Diagonais
            if abs(chromosome[i] - chromosome[j]) == abs(i - j):
                conflicts += 1
    return np.exp(-conflicts)

def crossover(parent1, parent2, N):
    p1_chromosome = parent1["chromosome"]
    p2_chromosome = parent2["chromosome"]
    
    crossover_point = random.randint(1, N-2)
    
    child = [-1] * N
    child[:crossover_point] = p1_chromosome[:crossover_point]
    
    # conjunto dos valores do pai 1 que ja estao no filho
    used_values = set(child[:crossover_point])

    j = 0
    for i in range(crossover_point, N):
        while j < N and p2_chromosome[j] in used_values:
            j += 1
        child[i] = p2_chromosome[j]
        used_values.add(p2_chromosome[j])
    
    return child, crossover_point

def mutate(solution, N, mutation_rate):
    chromosome = solution["chromosome"]
    
    mutated = False
    mutation_positions = None
    
    if random.random() < mutation_rate:
        i, j = random.sample(range(N), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        mutated = True
        mutation_positions = (i, j)
    
    return chromosome, mutated, mutation_positions

# Torneio nao funciona com probabilidades q nem slides
def select_parent(population, fitnesses):
    tournament_size = 3
    tournament = random.sample(list(enumerate(population)), tournament_size)
    winner_idx = max(tournament, key=lambda x: fitnesses[x[0]])[0]
    return population[winner_idx]


def run_experiment(N: int, population_size: int, generations: int, mutation_rate: float) -> Tuple[int, bool, dict, List[int], List[List[int]], dict]:
    population = []
    for _ in range(population_size):
        chromosome = random.sample(range(N), N)
        individual = {
            "chromosome": chromosome,
            "generation": 0,
            "parents": None,
            "crossover_point": None,
            "mutation": None
        }
        population.append(individual)
    

    fitnesses = [calculate_fitness(individual, N) for individual in population]
    best_idx = np.argmax(fitnesses)
    best_solution = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]
    
    solution_found = best_fitness == 1.0
    generation = 0
    

    all_populations = {0: population.copy()}

    while not solution_found and generation < generations:
        new_population = []
        
        new_population.append(best_solution)
        
        while len(new_population) < population_size:
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            

            child_chromosome, crossover_point = crossover(parent1, parent2, N)
            
            temp_solution = {"chromosome": child_chromosome}
            child_chromosome, was_mutated, mutation_positions = mutate(temp_solution, N, mutation_rate)
            
            child = {
                "chromosome": child_chromosome,
                "generation": generation + 1,
                "parents": {
                    "parent1": parent1.copy(),
                    "parent2": parent2.copy()
                },
                "crossover_point": crossover_point,
                "mutation": {
                    "applied": was_mutated,
                    "positions": mutation_positions
                }
            }
            
            new_population.append(child)
        
        population = new_population
        generation += 1
        all_populations[generation] = population.copy()
        
        fitnesses = [calculate_fitness(individual, N) for individual in population]
        current_best_idx = np.argmax(fitnesses)
        
        if fitnesses[current_best_idx] > best_fitness:
            best_solution = population[current_best_idx].copy()
            best_fitness = fitnesses[current_best_idx]
            
        if best_fitness == 1.0:
            solution_found = True
    
    return generation, solution_found, all_populations, best_solution

def main():
    global POPULATION_SIZE, GENERATIONS, MUTATION_RATE
    POPULATION_SIZE = 100
    GENERATIONS = 3000
    MUTATION_RATE = 0.2
    
    N = 4
    
    generations, found, all_populations, best_solution = run_experiment(N, POPULATION_SIZE, GENERATIONS, MUTATION_RATE)
    
    print(f"Solution found: {found}")
    
    if generations > 0:
        print("\nBest solution:")
        print(best_solution["chromosome"])
        print(best_solution["generation"])
        print("parent1: ", best_solution["parents"]["parent1"]["chromosome"])
        print("parent2: ", best_solution["parents"]["parent2"]["chromosome"])
        print("crossover_point: ", best_solution["crossover_point"])
        print("mutation: ", best_solution["mutation"])

        
        fitness = calculate_fitness(best_solution, N)
        print(f"\nFitness of solution: {fitness}")
    else:
        if found:
            print("Solution found on initial population:")
            print(best_solution["chromosome"])


if __name__ == "__main__":
    main() 