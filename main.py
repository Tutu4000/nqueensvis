"""
N-Queens Problem Solver - Command Line Interface

This module provides a command-line interface for running N-Queens experiments
and collecting statistics on the genetic algorithm's performance.
"""

import time
import os
import argparse
import numpy as np
from datetime import datetime
from typing import List, Tuple
from tabulate import tabulate

from nqueens import run_experiment, calculate_fitness

def save_results(results, filename, population_size, generations, mutation_rate, num_experiments):
    """Save experiment results to a file."""
    with open(filename, 'w') as f:
        f.write("N-Queens Genetic Algorithm Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}\n")
        f.write(f"- Population size: {population_size}\n")
        f.write(f"- Maximum generations: {generations}\n")
        f.write(f"- Mutation rate: {mutation_rate}\n")
        f.write(f"- Number of experiments per N: {num_experiments}\n\n")

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
        f.write(f"Average success rate: {avg_success:.1f}%\n")
        f.write(f"Minimum success rate: {min_success:.1f}%\n")
        
def print_solution(solution, N):
    """Print a readable representation of the solution."""
    print("\nSolution:")
    print("-" * (N * 2 + 1))
    for row in range(N):
        line = "|"
        for col in range(N):
            if solution[col] == row:
                line += "Q|"
            else:
                line += " |"
        print(line)
    print("-" * (N * 2 + 1))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="N-Queens Genetic Algorithm")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--gen", type=int, default=5000, help="Maximum generations")
    parser.add_argument("--mut", type=float, default=0.2, help="Mutation rate")
    parser.add_argument("--exp", type=int, default=10, help="Number of experiments per N")
    parser.add_argument("--min", type=int, default=5, help="Minimum N to test")
    parser.add_argument("--max", type=int, default=30, help="Maximum N to test")
    parser.add_argument("--show", action="store_true", help="Show solution boards")
    args = parser.parse_args()
    
    population_size = args.pop
    generations = args.gen
    mutation_rate = args.mut
    num_experiments = args.exp
    n_values = range(args.min, args.max + 1)
    show_solutions = args.show
    
    print(f"Running N-Queens experiments with:")
    print(f"- Population size: {population_size}")
    print(f"- Maximum generations: {generations}")
    print(f"- Mutation rate: {mutation_rate}")
    print(f"- Number of experiments per N: {num_experiments}")
    print(f"- Testing N values from {args.min} to {args.max}")
    
    results = []  # "N", "Success rate", "Average generations", "Average time"
    
    for N in n_values:
        print(f"\nRunning experiments for N = {N}")
        iterations = []
        success_count = 0
        total_time = 0
        last_solution = None
        
        for experiment in range(num_experiments):
            start_time = time.time()
            generations_used, found, solution = run_experiment(N, population_size, generations, mutation_rate)
            end_time = time.time()
            execution_time = end_time - start_time
            
            if found:
                success_count += 1
                iterations.append(generations_used)
                last_solution = solution
            
            total_time += execution_time
            
            avg_iterations = np.mean(iterations) if iterations else 0
            success_percentage = (success_count / (experiment + 1)) * 100
            
            result_row = [
                N,
                f"{success_percentage:.1f}%",
                f"{avg_iterations:.1f}" if iterations else "N/A",
                f"{total_time/(experiment+1):.1f}s"
            ]
            
            # Update results if we already have an entry for this N
            if N in [r[0] for r in results]:
                results[[r[0] for r in results].index(N)] = result_row
            else:
                results.append(result_row)
                
            print(f"  Experiment {experiment + 1}/{num_experiments}: {'Success' if found else 'Failure'}")
            print(f"  Generations: {generations_used}, Time: {execution_time:.2f}s")
        
        # Show the last found solution for this N value if requested
        if show_solutions and last_solution:
            print_solution(last_solution, N)
            print(f"Fitness: {calculate_fitness(last_solution, N)}")

    print("\nFinal results:")
    headers = ["N", "Success rate", "Average generations", "Average time"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

    # Save results to a file
    filename = f"resultados_n_rainhas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    save_results(results, filename, population_size, generations, mutation_rate, num_experiments)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main() 