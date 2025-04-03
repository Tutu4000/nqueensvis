# N-Queens Genetic Algorithm Project

A comprehensive solution to the N-Queens problem using genetic algorithms, with both console-based execution and interactive visualization.

## Overview

The N-Queens problem requires placing N queens on an N×N chessboard such that no two queens threaten each other. This implementation uses a genetic algorithm approach to find solutions.

## Project Structure

This project is organized into two main components:

### 1. Core Algorithm (`/`)

The root directory contains the core genetic algorithm implementation without visualization:

- `main.py` - Command-line interface for running experiments
- `nqueens.py` - Core genetic algorithm implementation

### 2. Visualization (`/Visualization`)

The Visualization directory contains all the code related to graphical representation:

- `executor.py` - GUI for configuring and running the algorithm with custom parameters
- `visualization.py` - Visualization engine for displaying algorithm progress
- `run.bat` - Helper script to install dependencies and launch the GUI

## Features

### Core Algorithm

- Permutation encoding for chromosomes
- Tournament selection
- Order-based crossover with permutation preservation
- Swap mutation
- Elitism preservation
- Configurable parameters (population size, generations, mutation rate)
- Comprehensive statistics and experiment tracking

### Visualization

- Real-time display of the evolving solution
- Lineage tracking of individuals (parent-child relationships)
- Crossover and mutation operations visualization
- Full ancestry tree view

## Installation

1. Clone this repository
2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

### Running Core Algorithm

Run from the root directory:

```
python main.py
```

This will execute a series of experiments and save the results to a timestamped file.

### Running with Visualization

Navigate to the Visualization directory and run:

```
python executor.py
```

Or use the included batch file:

```
cd Visualization
run.bat
```

## Requirements

- Python 3.8+
- NumPy
- Pygame (for visualization)
- Tabulate (for formatted output) 