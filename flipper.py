"""Flipper Puzzle Solver.

The flipper puzzle consists of a 3x3 grid of cells, each of which can be in one of two
states:

    up: represented by 1
    down: represented by -1

Touching a cell flips the value of the cell and its immediate neighbors
(up, down, left, right). The goal of the puzzle is to start from an initial state
and reach the all-up state by touching a sequence of cells.

To solve this puzzle, consider the 9 possible operations that can be performed on the
board. Each operation can be represented by a 3x3 matrix with values in {-1, 1}. With -1
representing a flip and 1 representing no change.

Let $o_i$ be the operation matrrix for touching the $i$-th cell, and that the solution
involves touch the $j$-th cell $n_j = 0, 1$ times. Let b be the initial state of the
board. Then the solution can be written as
a linear system of equations:

    $$
    \sum_{j=1}^9 (o_ij == -1) n_j = (b_i == -1) mod 2
    $$

This is a system of linear equations over the field of integers modulo 2. We can solve
this system using Gaussian elimination.
"""

import galois
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def generate_ops():
    """
    Generates 9 operations for a 3x3 grid.

    Each operation is a 3x3 matrix with values in {-1, 1}.
    The operations are designed to flip the values of the grid in the following pattern:

    Touching a cell flips the value of the cell and its immediate neighbors
    (up, down, left, right).
    """
    ops = [ np.ones((3,3)) for _ in range(9) ]
    for i in range(3):
        for j in range(3):
            ops[3*i + j][i, j] = -1
            if i > 0:
                ops[3*i + j][i-1, j] = -1
            if i < 2:
                ops[3*i + j][i+1, j] = -1
            if j > 0:
                ops[3*i + j][i, j-1] = -1
            if j < 2:
                ops[3*i + j][i, j+1] = -1
    return ops

def show_board(state : np.ndarray|list[np.ndarray], count_steps=False,
               highlight_cell=None):
    """
    Displays the state of the board using a heatmap.

    Args:
        state (np.ndarray|list[np.ndarray]): The state of the board to display.
            Can be a single 3x3 array or a list of 3x3 arrays.
    """
    if isinstance(state, list):
        fig, axs = plt.subplots(1, len(state), figsize=(5*len(state), 5))
        for i, ax in enumerate(axs):
            sns.heatmap(state[i], linewidth=1, linecolor='white', cmap='RdYlGn',
                        center=0,
                        vmin=-1, vmax=1,
                        cbar=False, square=True, ax=ax)
            if count_steps:
                ax.set_title(f"Step {i+1}")
        if highlight_cell is not None:
            for ax, cell in zip(axs[:len(highlight_cell)], highlight_cell):
                ax.add_patch(plt.Rectangle((cell%3, cell//3), 1, 1, fill=False,
                                           edgecolor='black', lw=5))
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(state, linewidth=1, linecolor='white', cmap='RdYlGn', center=0,
                    vmin=-1, vmax=1,
                    cbar=False, square=True, ax=ax)
        if highlight_cell is not None:
            ax.add_patch(plt.Rectangle((highlight_cell%3, highlight_cell//3), 1, 1,
                                        fill=False,
                                        edgecolor='black', lw=5))
    return fig

def apply_op( state, op):
    """
    Applies an operation to the state of the board.

    Args:
        state (np.ndarray): The state of the board.
        op (np.ndarray): The operation to apply.

    Returns:
        np.ndarray: The new state of the board.
    """
    return state * op

def solve( init_state: np.ndarray):
    """
    Solves the flipper puzzle for a given initial state.

    Args:
        init_state (np.ndarray): The initial state of the board.

    Returns:
        np.ndarray: a binary array indicating which cells to touch to solve the puzzle.
    """
    ops = generate_ops()
    A = np.array([ op.flatten()==-1 for op in ops ]).astype(np.uint8)
    b = (init_state.flatten()==-1).astype(np.uint8)

    # Solve using Galois field for modulo 2 arithmetic.
    GF2 = galois.GF(2)
    A_gf = galois.GF2(A)
    b_gf = galois.GF2(b)
    x = np.linalg.solve(A_gf, b_gf)
    return np.array(x.astype(np.uint8))

def demo_solution( init_state: np.ndarray, solution: np.ndarray):
    """
    Demonstrates the solution step by step.

    Args:
        init_state (np.ndarray): The initial state of the board.
        solution (np.ndarray): A binary array indicating which cells to touch.
    """
    ops = generate_ops()
    state = init_state.copy()
    states = [state]
    cells_touched = np.where(solution == 1)[0]
    for cell in cells_touched:
        state = apply_op(state, ops[cell])
        states.append(state)
    return show_board(states, count_steps=True, highlight_cell=cells_touched)

def solve_and_demo( init_state: np.ndarray):
    """
    Solves the flipper puzzle for a given initial state and
    demonstrates the solution step by step.

    Args:
        init_state (np.ndarray): The initial state of the board.
    """
    solution = solve(init_state)
    return demo_solution(init_state, solution)

def elemental_solutions():
    """
    Generates the elemental solutions for the flipper puzzle.  An elemental solution is
    a solution that flips exactly exactly one cell.

    Returns:
        np.ndarray: An array of 9 elemental solutions.
    """
    init_states = [ np.ones((3,3)) for _ in range(9) ]
    for i in range(9):
        init_states[i].flat[i] = -1
    return np.array([ solve(init_state) for init_state in init_states ])

def solve_from_elemental( init_state):
    """
    Solves the flipper puzzle for a given initial state by combining the elemental
    solutions.  Each elemental solution flips exactly one cell.  The solution to the
    puzzle is the sum of the elemental solutions that flip the cells that are in the
    down position, modulo 2.

    Args:
        init_state (np.ndarray): The initial state of the board.

    Returns:
        np.ndarray: a binary array indicating which cells to touch to solve the puzzle.
    """
    elemsols = elemental_solutions()
    idx = np.where(init_state.flatten() == -1)[0]
    sol = elemsols[idx].sum(axis=0) % 2
    return sol
