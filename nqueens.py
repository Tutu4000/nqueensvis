"""
N-Queens Genetic Algorithm Implementation

This module provides a genetic algorithm implementation for solving the N-Queens problem.
The N-Queens problem requires placing N queens on an N×N chessboard such that 
no two queens threaten each other.
"""

import random
import numpy as np
from typing import List, Tuple, Set, Optional

def calculate_fitness(solution: List[int], N: int) -> float:
    """
    Calculate fitness of a solution. Higher is better, with 1.0 indicating a perfect solution.
    
    Args:
        solution: A list of integers representing queen positions
        N: The size of the board
        
    Returns:
        Fitness score between 0.0 and 1.0
    """
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Column conflicts (redundant with permutation encoding)
            if solution[i] == solution[j]:
                conflicts += 1
            # Diagonal conflicts
            if abs(solution[i] - solution[j]) == abs(i - j):
                conflicts += 1
    return np.exp(-conflicts)

def crossover(parent1: List[int], parent2: List[int], N: int) -> List[int]:
    """
    Perform crossover between two parents to create a child.
    Uses order-based crossover maintaining permutation validity.
    
    Args:
        parent1: First parent's chromosome
        parent2: Second parent's chromosome
        N: The size of the board
        
    Returns:
        A new child chromosome
    """
    crossover_point = random.randint(1, N-2)
    
    child = [-1] * N
    child[:crossover_point] = parent1[:crossover_point]
    
    # Keep track of values already used from parent1
    used_values = set(child[:crossover_point])

    j = 0
    for i in range(crossover_point, N):
        while j < N and parent2[j] in used_values:
            j += 1
        child[i] = parent2[j]
        used_values.add(parent2[j])
    return child

def mutate(solution: List[int], N: int, mutation_rate: float) -> List[int]:
    """
    Potentially mutate a solution by swapping two positions.
    
    Args:
        solution: The chromosome to mutate
        N: The size of the board
        mutation_rate: Probability of mutation (0.0-1.0)
        
    Returns:
        The mutated chromosome
    """
    if random.random() < mutation_rate:
        i, j = random.sample(range(N), 2)
        solution[i], solution[j] = solution[j], solution[i]
    return solution

def select_parent(population: List[List[int]], fitnesses: List[float]) -> List[int]:
    """
    Select a parent using tournament selection.
    
    Args:
        population: List of chromosomes
        fitnesses: List of fitness values
        
    Returns:
        The selected parent chromosome
    """
    tournament_size = 3
    tournament = random.sample(list(enumerate(population)), tournament_size)
    winner_idx = max(tournament, key=lambda x: fitnesses[x[0]])[0]
    return population[winner_idx]

def run_experiment(N: int, population_size: int, generations: int, mutation_rate: float) -> Tuple[int, bool, List[int]]:
    """
    Run a genetic algorithm experiment to solve the N-Queens problem.
    
    Args:
        N: The size of the board
        population_size: Number of individuals in the population
        generations: Maximum number of generations to run
        mutation_rate: Probability of mutation (0.0-1.0)
        
    Returns:
        A tuple of (generations_used, solution_found, best_solution)
    """
    # Initialize population with random permutations
    population: List[List[int]] = [random.sample(range(N), N) for _ in range(population_size)]
    
    # Calculate initial fitness values
    fitnesses: List[float] = [calculate_fitness(individual, N) for individual in population]
    best_idx = np.argmax(fitnesses)
    best_solution: List[int] = population[best_idx].copy()
    best_fitness: float = fitnesses[best_idx]
    
    solution_found = best_fitness == 1.0
    generation = 0
    
    # Main GA loop
    while not solution_found and generation < generations:
        new_population = []
        
        # Elitism: preserve best solution
        new_population.append(best_solution)
        
        # Create new population through selection, crossover, and mutation
        while len(new_population) < population_size:
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            
            child = crossover(parent1, parent2, N)
            child = mutate(child, N, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
        
        # Recalculate fitness values
        fitnesses = [calculate_fitness(individual, N) for individual in population]
        current_best_idx = np.argmax(fitnesses)
        
        # Update best solution if we found a better one
        if fitnesses[current_best_idx] > best_fitness:
            best_solution = population[current_best_idx].copy()
            best_fitness = fitnesses[current_best_idx]
            
        # Check if we found a solution
        if best_fitness == 1.0:
            solution_found = True
        
        generation += 1
    
    return generation, solution_found, best_solution 