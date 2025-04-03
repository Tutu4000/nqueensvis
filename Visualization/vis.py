"""
N-Queens Genetic Algorithm - Visualization Interface

This module adapts the genetic algorithm to work with the visualization system.
It translates between the core algorithm and the visualization requirements.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from visualization import run_visualization, update_visualization_data

# Global dictionary to store all individuals by ID
all_individuals_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}

# Default parameters 
population_size = 30
generations = 150
mutation_rate = 0.1

def calculate_fitness(solution, N):
    """Calculate fitness of a solution."""
    chromosome = solution["chromosome"]
    
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Column conflicts
            if chromosome[i] == chromosome[j]:
                conflicts += 1
            # Diagonal conflicts
            if abs(chromosome[i] - chromosome[j]) == abs(i - j):
                conflicts += 1
    return np.exp(-conflicts)

def crossover(parent1, parent2, N):
    """Perform crossover between two parents."""
    p1_chromosome = parent1["chromosome"]
    p2_chromosome = parent2["chromosome"]
    
    crossover_point = random.randint(1, N-2)
    
    child = [-1] * N
    child[:crossover_point] = p1_chromosome[:crossover_point]
    
    # Keep track of values from parent1 already in child
    used_values = set(child[:crossover_point])

    j = 0
    for i in range(crossover_point, N):
        while j < N and p2_chromosome[j] in used_values:
            j += 1
        child[i] = p2_chromosome[j]
        used_values.add(p2_chromosome[j])
    
    return child, crossover_point

def mutate(solution, N, mutation_rate):
    """Potentially mutate a solution."""
    chromosome = solution["chromosome"]
    
    mutated = False
    mutation_positions = None
    
    if random.random() < mutation_rate:
        i, j = random.sample(range(N), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        mutated = True
        mutation_positions = (i, j)
    
    return chromosome, mutated, mutation_positions

def select_parent(population, fitnesses):
    """Select a parent using tournament selection."""
    tournament_size = 3
    tournament = random.sample(list(enumerate(population)), tournament_size)
    winner_idx = max(tournament, key=lambda x: fitnesses[x[0]])[0]
    return population[winner_idx]

def run_experiment(N: int, pop_size: int, gen_count: int, mut_rate: float, callback: Optional[Callable] = None) -> Tuple[int, bool, Optional[Tuple[int, int]], Dict]:
    """Run the genetic algorithm with visualization callbacks."""
    global all_individuals_lookup
    all_individuals_lookup = {} # Reset for new experiment
    
    # Initialize population
    population = []
    for i in range(pop_size):
        chromosome = random.sample(range(N), N)
        individual_id = (0, i)
        individual = {
            "id": individual_id,
            "chromosome": chromosome,
            "generation": 0,
            "parents": None, # Parents are None for initial population
            "crossover_point": None,
            "mutation": None
        }
        population.append(individual)
        all_individuals_lookup[individual_id] = individual # Store initial pop

    # Calculate initial fitness values
    fitnesses = [calculate_fitness(individual, N) for individual in population]
    best_idx = np.argmax(fitnesses)
    best_solution_id = population[best_idx]["id"] # Store the ID of the best
    best_fitness = fitnesses[best_idx]

    solution_found = best_fitness == 1.0
    generation = 0

    # Initial callback for starting state
    if callback:
        callback(all_individuals_lookup[best_solution_id], N, finished=False, all_individuals=all_individuals_lookup)

    # Main GA loop
    while not solution_found and generation < gen_count:
        new_population = []

        # Elitism: Carry over the best solution
        best_individual_data = all_individuals_lookup[best_solution_id]
        new_population.append(best_individual_data)

        next_individual_idx = 1 # Start index for new children in this generation

        # Create new population through selection, crossover, and mutation
        while len(new_population) < pop_size:
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)

            child_chromosome, crossover_point = crossover(parent1, parent2, N)

            temp_solution = {"chromosome": child_chromosome}
            child_chromosome, was_mutated, mutation_positions = mutate(temp_solution, N, mut_rate)

            child_id = (generation + 1, next_individual_idx)
            next_individual_idx += 1

            child = {
                "id": child_id,
                "chromosome": child_chromosome,
                "generation": generation + 1,
                "parents": {
                    # Store parent IDs
                    "parent1_id": parent1["id"],
                    "parent2_id": parent2["id"]
                },
                "crossover_point": crossover_point,
                "mutation": {
                    "applied": was_mutated,
                    "positions": mutation_positions
                }
            }

            new_population.append(child)
            all_individuals_lookup[child_id] = child # Store the new child

        population = new_population
        generation += 1

        # Recalculate fitness values
        fitnesses = [calculate_fitness(individual, N) for individual in population]
        current_best_idx = np.argmax(fitnesses)

        current_best_id = population[current_best_idx]["id"]
        current_best_fitness = fitnesses[current_best_idx]

        # Update best solution if we found a better one
        if current_best_fitness > best_fitness:
            best_solution_id = current_best_id
            best_fitness = current_best_fitness
            # Call callback with the new best
            if callback:
                callback(all_individuals_lookup[best_solution_id], N, finished=False, all_individuals=all_individuals_lookup)

        # Check if we found a solution
        if best_fitness == 1.0:
            solution_found = True
            # Ensure the best ID is captured if found on this iteration
            best_solution_id = current_best_id

    # Final callback after loop finishes
    final_best_solution_data = all_individuals_lookup.get(best_solution_id)
    if callback and final_best_solution_data:
        callback(final_best_solution_data, N, finished=True, all_individuals=all_individuals_lookup)
    elif callback:
         callback(None, N, finished=True, all_individuals=all_individuals_lookup)

    return generation, solution_found, best_solution_id, all_individuals_lookup

def main():
    """Run the visualization with default parameters."""
    global population_size, generations, mutation_rate
    
    N = 6

    # Define a wrapper function that calls run_experiment with the callback
    def ga_runner():
        print(f"Starting GA for N={N}...")
        # Capture the final ID and the lookup table
        generation, found, final_best_id, lookup_table = run_experiment(
            N, population_size, generations, mutation_rate,
            callback=update_visualization_data
        )
        print(f"GA finished after {generation} generations. Solution found: {found}")
        print(f"Final best individual ID: {final_best_id}")
        print(f"Total individuals created: {len(lookup_table)}")
        
        # Print detailed information about the final solution
        if final_best_id and final_best_id in lookup_table:
            final_solution = lookup_table[final_best_id]
            fitness = calculate_fitness(final_solution, N)
            
            print("\n--- FINAL SOLUTION DETAILS ---")
            print(f"Chromosome: {final_solution['chromosome']}")
            print(f"Fitness: {fitness}")
            print(f"Generation: {final_solution['generation']}")
            
            # Print parent information if available
            parents = final_solution.get('parents')
            if parents:
                p1_id = parents.get('parent1_id')
                p2_id = parents.get('parent2_id')
                
                if p1_id and p1_id in lookup_table:
                    p1 = lookup_table[p1_id]
                    print(f"\nParent 1 (ID: {p1_id}):")
                    print(f"Chromosome: {p1['chromosome']}")
                    print(f"Fitness: {calculate_fitness(p1, N)}")
                    print(f"Generation: {p1['generation']}")
                
                if p2_id and p2_id in lookup_table:
                    p2 = lookup_table[p2_id]
                    print(f"\nParent 2 (ID: {p2_id}):")
                    print(f"Chromosome: {p2['chromosome']}")
                    print(f"Fitness: {calculate_fitness(p2, N)}")
                    print(f"Generation: {p2['generation']}")
            
            # Print crossover and mutation information
            if 'crossover_point' in final_solution and final_solution['crossover_point'] is not None:
                crossover_point = final_solution['crossover_point']
                print(f"\nCrossover point: {crossover_point}")
                print("Crossover pattern: ", end="")
                for i in range(N):
                    if i < crossover_point:
                        print("1", end="")
                    else:
                        print("2", end="")
                print(" (1=from parent1, 2=from parent2)")
            
            if 'mutation' in final_solution:
                mutation_info = final_solution['mutation']
                if mutation_info:
                    print(f"\nMutation applied: {mutation_info.get('applied', False)}")
                    if mutation_info.get('applied'):
                        positions = mutation_info.get('positions')
                        if positions:
                            print(f"Mutation positions: {positions}")

    # Run the visualization, passing the GA runner function
    run_visualization(ga_runner)

if __name__ == "__main__":
    main() 