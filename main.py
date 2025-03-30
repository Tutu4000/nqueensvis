import random
import numpy as np
from tabulate import tabulate
import time
import os
from datetime import datetime
from typing import List, Tuple, Set, Optional

def calculate_fitness(solution, N):
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Coluna
            if solution[i] == solution[j]:
                conflicts += 1
            # Diagonais
            if abs(solution[i] - solution[j]) == abs(i - j):
                conflicts += 1
    return np.exp(-conflicts)

def crossover(parent1, parent2, N):
    crossover_point = random.randint(1, N-2)
    
    child = [-1] * N
    child[:crossover_point] = parent1[:crossover_point]
    
    # conjunto dos valores do pai 1 que ja estao no filho
    used_values = set(child[:crossover_point])

    j = 0
    for i in range(crossover_point, N):
        while j < N and parent2[j] in used_values:
            j += 1
        child[i] = parent2[j]
        used_values.add(parent2[j])
    return child

def mutate(solution, N, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(N), 2)
        solution[i], solution[j] = solution[j], solution[i]
    return solution

# Torneio nao funciona com probabilidades q nem slides
def select_parent(population, fitnesses):
    tournament_size = 3
    tournament = random.sample(list(enumerate(population)), tournament_size)
    winner_idx = max(tournament, key=lambda x: fitnesses[x[0]])[0]
    return population[winner_idx]

def bad_population_repository(population, fitnesses):
    return 0

def run_experiment(N: int, population_size: int, generations: int, mutation_rate: float) -> Tuple[int, bool]:
    population: List[List[int]] = [random.sample(range(N), N) for _ in range(population_size)]
    
    fitnesses: List[float] = [calculate_fitness(individual, N) for individual in population]
    best_idx = np.argmax(fitnesses)
    best_solution: List[int] = population[best_idx].copy()
    best_fitness: float = fitnesses[best_idx]
    
    solution_found = best_fitness == 1.0
    generation = 0
    
    while not solution_found and generation < generations:
        new_population = []
        
        # Elitismo melhor individuo sempre vai pra nova populacao
        new_population.append(best_solution)
        

        while len(new_population) < population_size:
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            
            child = crossover(parent1, parent2, N)
            
            child = mutate(child, N, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
        
        fitnesses = [calculate_fitness(individual, N) for individual in population]
        current_best_idx = np.argmax(fitnesses)
        
        if fitnesses[current_best_idx] > best_fitness:
            best_solution = population[current_best_idx].copy()
            best_fitness = fitnesses[current_best_idx]
            
        if best_fitness == 1.0:
            solution_found = True
        
        generation += 1
    
    return generation, solution_found


def save_results(results, filename):
    with open(filename, 'w') as f:
        f.write("Resultados\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}\n")
        f.write(f"- Population size: {POPULATION_SIZE}\n")
        f.write(f"- Max it: {GENERATIONS}\n")
        f.write(f"- Mutation probability: {MUTATION_RATE}\n")
        f.write(f"- Number of experiments per N: {NUM_EXPERIMENTS}\n\n")

        f.write("Detailed results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'N':<5} {'Success rate':<15} {'Average generations':<20} {'Average time':<15}\n")
        f.write("-" * 80 + "\n")
        
        for row in results:
            f.write(f"{row[0]:<5} {row[1]:<15} {row[2]:<20} {row[3]:<15}\n")
        
        f.write("-" * 80 + "\n\n")
        
        f.write("Analysis:\n")
        f.write("-" * 80 + "\n")
        success_rates = [float(r[1].strip('%')) for r in results]
        avg_success = np.mean(success_rates)
        min_success = min(success_rates)
        f.write(f"Average success: {avg_success:.1f}%\n")
        f.write(f"Minimum success: {min_success:.1f}%\n")
        

def main():
    global POPULATION_SIZE, GENERATIONS, MUTATION_RATE, NUM_EXPERIMENTS
    POPULATION_SIZE = 100
    GENERATIONS = 3000
    MUTATION_RATE = 0.2
    NUM_EXPERIMENTS = 10
    NVALUES = range(8, 30)
    
    results: List[Tuple[int, str, str, str]] = [] # "N", "Success rate", "Average generations", "Average time"
    for N in NVALUES:
        print(f"\nCurrent N = {N}")
        iterations: List[int] = []
        success_rate = 0
        total_time = 0 
        for experiments in range(NUM_EXPERIMENTS):
            start_time = time.time()
            generations, found = run_experiment(N, POPULATION_SIZE, GENERATIONS, MUTATION_RATE)
            end_time = time.time()
            execution_time = end_time - start_time
            
            if found:
                success_rate += 1
                iterations.append(generations)
            
            total_time += execution_time
            
            avg_iterations = np.mean(iterations)
            success_percentage = (success_rate / (experiments + 1)) * 100
            
            result_row: Tuple[int, str, str, str] = [
                N,
                f"{success_percentage:.1f}%",
                f"{avg_iterations:.1f}",
                f"{total_time/(experiments+1):.1f}s"
            ]
            
            if N in [r[0] for r in results]:
                results[[r[0] for r in results].index(N)] = result_row
            else:
                results.append(result_row)
            print(f"Experiment {experiments + 1}/{NUM_EXPERIMENTS}: {'Success' if found else 'Failure'}")
            print(f"Generations: {generations}, Time: {execution_time:.1f}s")

    print("\nFinal results:")
    headers = ["N", "Success rate", "Average generations", "Average time"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

    filename = f"resultados_n_rainhas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    save_results(results, filename)

if __name__ == "__main__":
    main() 