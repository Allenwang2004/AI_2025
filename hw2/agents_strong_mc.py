
import numpy as np
import random
import game
from copy import deepcopy

def print_INFO():
    print("""
========================================
  DATE: 2025/04/04
  STUDENT NAME: 王子儀
  STUDENT ID: 111705068
========================================
    """)

#
# Heuristic functions
#

def get_heuristic(board):
    num_twos       = game.count_windows(board, 2, 1)
    num_threes     = game.count_windows(board, 3, 1)
    num_twos_opp   = game.count_windows(board, 2, 2)
    num_threes_opp = game.count_windows(board, 3, 2)

    score = (
          1e10 * board.win(1)
        + 1e6  * num_threes
        + 10   * num_twos
        - 10   * num_twos_opp
        - 1e6  * num_threes_opp
        - 1e10 * board.win(2)
    )
    return score

def get_heuristic_strong(board):
    player = board.mark
    opponent = 3 - player
    score = 0

    center_col = board.table[:, board.column // 2]
    center_count = np.count_nonzero(center_col == player)
    score += center_count * 6

    num_twos       = game.count_windows(board, 2, player)
    num_threes     = game.count_windows(board, 3, player)
    num_twos_opp   = game.count_windows(board, 2, opponent)
    num_threes_opp = game.count_windows(board, 3, opponent)

    score += (
        1e9  * board.win(player)
        + 5e5 * num_threes
        + 60   * num_twos
        - 5e5 * num_threes_opp
        - 1e9 * board.win(opponent)
        - 80   * num_twos_opp
    )

    return score

#
# Minimax and AlphaBeta
#

def minimax(grid, depth, maximizingPlayer, dep=4):
    if grid.terminate() or depth == 0:
        return get_heuristic(grid), set()
    bestValue = -np.inf if maximizingPlayer else np.inf
    bestMoves = set()
    for col in grid.valid:
        next_grid = game.drop_piece(grid, col)
        value, _ = minimax(next_grid, depth - 1, not maximizingPlayer, dep)
        if maximizingPlayer:
            if value > bestValue:
                bestValue = value
                bestMoves = {col}
            elif value == bestValue:
                bestMoves.add(col)
        else:
            if value < bestValue:
                bestValue = value
                bestMoves = {col}
            elif value == bestValue:
                bestMoves.add(col)
    return bestValue, bestMoves

def alphabeta(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    if grid.terminate() or depth == 0:
        return get_heuristic(grid), set()
    bestValue = -np.inf if maximizingPlayer else np.inf
    bestMoves = set()
    for col in grid.valid:
        next_grid = game.drop_piece(grid, col)
        value, _ = alphabeta(next_grid, depth - 1, not maximizingPlayer, alpha, beta, dep)
        if maximizingPlayer:
            if value > bestValue:
                bestValue = value
                bestMoves = {col}
            elif value == bestValue:
                bestMoves.add(col)
            alpha = max(alpha, bestValue)
        else:
            if value < bestValue:
                bestValue = value
                bestMoves = {col}
            elif value == bestValue:
                bestMoves.add(col)
            beta = min(beta, bestValue)
        if beta <= alpha:
            break
    return bestValue, bestMoves

def your_function(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    if grid.terminate() or depth == 0:
        return get_heuristic_strong(grid), set()
    bestValue = -np.inf if maximizingPlayer else np.inf
    bestMoves = set()
    for col in grid.valid:
        next_grid = game.drop_piece(grid, col)
        value, _ = your_function(next_grid, depth - 1, not maximizingPlayer, alpha, beta, dep)
        if maximizingPlayer:
            if value > bestValue:
                bestValue = value
                bestMoves = {col}
            elif value == bestValue:
                bestMoves.add(col)
            alpha = max(alpha, bestValue)
        else:
            if value < bestValue:
                bestValue = value
                bestMoves = {col}
            elif value == bestValue:
                bestMoves.add(col)
            beta = min(beta, bestValue)
        if beta <= alpha:
            break
    return bestValue, bestMoves

#
# Monte Carlo Simulation
#

def simulate_random_game(board, current_player):
    while not board.terminate():
        valid_cols = board.valid
        if not valid_cols:
            break
        col = random.choice(valid_cols)
        board = game.drop_piece(board, col)
        current_player = 3 - current_player
    if board.win(1): return 1
    elif board.win(2): return 2
    else: return 0

def monte_carlo_score(board, player, num_simulations=15):
    scores = []
    for _ in range(num_simulations):
        sim_board = deepcopy(board)
        winner = simulate_random_game(sim_board, player)
        if winner == player:
            scores.append(1)
        elif winner == 0:
            scores.append(0.5)
        else:
            scores.append(0)
    return np.mean(scores)

#
# Agents
#

def agent_minimax(grid):
    return random.choice(list(minimax(grid, 4, True)[1]))

def agent_alphabeta(grid):
    return random.choice(list(alphabeta(grid, 4, True, -np.inf, np.inf)[1]))

def agent_reflex(grid):
    wins = [c for c in grid.valid if game.check_winning_move(grid, c, grid.mark)]
    if wins:
        return random.choice(wins)
    return random.choice(grid.valid)

def agent_strong(grid):
    _, candidate_moves = your_function(grid, 4, False, -np.inf, np.inf)
    best_score = -1
    best_move = None
    for col in candidate_moves:
        new_grid = game.drop_piece(grid, col)
        score = monte_carlo_score(new_grid, grid.mark, num_simulations=15)
        if score > best_score:
            best_score = score
            best_move = col
    return best_move
