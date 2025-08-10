# Flipper Puzzle Solver

A Python implementation for solving the classic "Flipper" puzzle using linear algebra over finite fields.

## Overview

The flipper puzzle consists of a 3×3 grid of cells, each of which can be in one of two states:
- **Up**: represented by `1` (green in visualizations)
- **Down**: represented by `-1` (red in visualizations)

The goal is to start from an initial state and reach the all-up state by touching a sequence of cells. When you touch a cell, it flips the value of that cell and its immediate neighbors (up, down, left, right).

## Mathematical Approach

This solver uses a mathematical approach by modeling the puzzle as a system of linear equations over the field of integers modulo 2 (GF(2)). Each of the 9 possible operations (touching each cell) can be represented by a 3×3 matrix, and the solution involves finding which operations to apply using Gaussian elimination.

## Requirements

- Python 3.7+
- `numpy`
- `matplotlib`
- `seaborn`
- **`galois`** - For finite field arithmetic (GF(2))

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd flipper
```

2. Install the required dependencies:
```bash
pip install numpy matplotlib seaborn galois
```

## Usage

### Basic Usage

```python
import numpy as np
from flipper import solve, show_board, demo_solution

# Define an initial state
init_state = np.array([
    [ 1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1, -1]
])

# Solve the puzzle
solution = solve(init_state)

# Visualize the solution
demo_solution(init_state, solution)
```

Now you can dominate this game in [the Moriarty Console at Murdle](https://murdle.com/console/).

### Available Functions

#### Core Functions

- **`solve(init_state)`**: Solves the flipper puzzle for a given initial state
- **`generate_ops()`**: Generates the 9 operation matrices
- **`show_board(state, count_steps=False, highlight_cell=None)`**: Visualizes board states
- **`demo_solution(init_state, solution)`**: Shows step-by-step solution
- **`solve_and_demo(init_state)`**: Combines solving and demonstration

#### Alternative Solving Methods

- **`elemental_solutions()`**: Generates elemental solutions (flipping exactly one cell)
- **`solve_from_elemental(init_state)`**: Solves using elemental solution combinations

### Example: Interactive Jupyter Notebook

The repository includes `flipper.ipynb` which demonstrates:
- Visualization of all 9 possible operations
- Solving a sample puzzle
- Step-by-step solution demonstration
- Alternative solving methods using elemental solutions

## How It Works

### Mathematical Foundation

The puzzle can be modeled as a system of linear equations:
```
∑(j=1 to 9) (o_ij == -1) * n_j = (b_i == -1) mod 2
```

Where:
- `o_ij` represents the operation matrix for touching the j-th cell
- `n_j` is 0 or 1 (whether to touch cell j)
- `b_i` is the initial state of cell i

### Solution Process

1. **Generate Operations**: Create 9 matrices representing each possible touch operation
2. **Build System**: Convert to binary system over GF(2)
3. **Solve**: Use Gaussian elimination to find which cells to touch
4. **Apply**: Execute the solution sequence

### Elemental Solutions

An alternative approach uses "elemental solutions" - solutions that flip exactly one cell. Any puzzle solution can be constructed by combining these elemental solutions for cells that are initially in the "down" position.

## Visualization

The solver includes rich visualization capabilities:
- **Green cells**: Up state (`1`)
- **Red cells**: Down state (`-1`)
- **Step-by-step animations**: Show how the board changes with each operation
- **Operation highlighting**: Visual indication of which cell is being touched

## File Structure

```
flipper/
├── README.md           # This file
├── flipper.py          # Main solver implementation
├── flipper.ipynb       # Interactive Jupyter notebook
└── __pycache__/        # Python cache files
```

## Theory

The flipper puzzle is a classic example of:
- **Linear algebra over finite fields**
- **Game theory and puzzle solving**
- **Matrix operations in GF(2)**
- **Gaussian elimination applications**

Each puzzle state can be uniquely solved (if solvable) using this mathematical approach, making it much more efficient than brute-force methods.

## Contributing

Feel free to contribute by:
- Adding new visualization features
- Implementing different grid sizes
- Optimizing the solving algorithms
- Adding more comprehensive tests

## License

This project is open source. Please check the repository for specific license information.

## References

- Linear algebra over finite fields
- Galois field theory (GF(2))
- Classic puzzle solving techniques
