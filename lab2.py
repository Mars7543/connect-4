# AI Lab 2: Games and ConnectFour 

# Name(s): Sebastian Wittrock     and   Miguel Roberts
# Email(s): sebwit20@bergen.org   and   migrob20@bergen.org

from game_api import *
from boards import *
from toytree import GAME1
from time import time


INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board=None):
    """Returns True if game is over, otherwise False."""
    if not board:  # if board doesn't exist throw an exception
        raise Exception('Board Not Defined')

    """
    - Loop through every chain and check if a player won (len(chain) >= 4)
    - Check if every column is full
    """

    chains = board.get_all_chains()

    for chain in chains:
        if len(chain) >= 4:
            return True

    all_columns_full = True
    for col in range(7):
        if not board.is_column_full(col):
            all_columns_full = False

    return True if all_columns_full else False

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""

    """
    If the game isn't over:
        Loop through each column and if it isn't full:
            add a piece to the column and append the new board to the list of "next boards"  
    """

    next_boards = []

    if is_game_over_connectfour(board):
        return next_boards

    for col in range(7):
        if not board.is_column_full(col):
            next_boards.append(board.add_piece(col))

    return next_boards


def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    if not is_game_over_connectfour(board):
        raise Exception("Game not over") # Error if game is not over

    """ 
    1) Loop through every chain to find the winning chain (len(chain) >= 4)
        * the last player was the winning player * therefore
    
        if the current player is the maximizer:
            then return -1000 because the minimizer won 
            
        else if the current player is the minimizer:
            return 1000 because the maximizer won
        
        else return 0 - tie
    """

    chains = board.get_all_chains()
    for chain in chains:
        if len(chain) >= 4:
            if is_current_player_maximizer:
                return -1000
            else:
                return 1000

    return 0


def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""

    """
    The maximum number of moves a player can make is 21 ((6 rows * 7 cols) / (2 players))
    A player will lose 5 points per move made (not including the necessary 4 needed to win)
        Lower # of moves = greater points
    Maximum Points = 1000 (minimum) + 85 (maximimum # of moves needed * 5 points per move)
    """
    maxPoints = 1085

    """
    Find the winning chain and count the # of moves made - the 4 necessary moves
    Subtract 5 * # of excess moves from max score
    """

    chains = board.get_all_chains()
    for chain in chains: # get every chain
        if len(chain) >= 4: # find winning chain
            numPieces = board.count_pieces(not is_current_player_maximizer) # count pieces of the winner
            leftOver = numPieces - 4 # find how many extra moves they did other than the necessary moves needed to win
            for i in range(0,leftOver):
                maxPoints = maxPoints - 5 # subtract 5 points for every extra move

            return maxPoints
    return 0


# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

# min and max use index 0 b/c a tuple is being passed in with the format (score, state) to preserve states
def min(x, y):
    return x if x[0] <= y[0] else y


def max(x, y):
    return x if x[0] >= y[0] else y


static_evaluations = 0


def minimax(state, maximize, path=[], dfs_maximizing=False):

    # if game is over return a tuple with the score, current state
    if state.is_game_over():
        global static_evaluations
        static_evaluations += 1

        path.append(state)

        return state.get_endgame_score(maximize), state, path

    # array of next states for maximize and minimize
    children = state.generate_next_states()

    """ 
    (Min/Max)imize use tuples with the format: 
       (score, state) to preserve the best state
    """

    # if dfs_maximizing always put True for maximize arg
    if dfs_maximizing:
        maxEval = (-INF, '', '')

        for child in children:
            eval = minimax(child, True, path, True)
            maxEval = max(maxEval, eval)

        return maxEval

    elif maximize:
        maxEval = (-INF, '', '')

        for child in children:
            eval = minimax(child, False, path)
            maxEval = max(maxEval, eval)

        return maxEval

    else:
        minEval = (INF, '', '')

        for child in children:
            eval = minimax(child, True, path)
            minEval = min(minEval, eval)

        return minEval

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""

    global static_evaluations
    static_evaluations = 0

    score, state, path = minimax(state=state, maximize=True, path=[], dfs_maximizing=True)

    return path, score, static_evaluations


# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

# pretty_print_dfs_type(dfs_maximizing(GAME1))


def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Returns the same as dfs_maximizing:
    a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""

    global static_evaluations
    static_evaluations = 0

    score, state, path = minimax(state=state, maximize=maximize)

    return path, score, static_evaluations

# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


#### Part 3: Cutting off and Pruning search #############################################


def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    raise NotImplementedError
    

## Note that the signature of heuristic_fn is heuristic_fn(board, maximize=True)

def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs h-minimax, cutting off search at depth_limit and using heuristic_fn
    to evaluate non-terminal states. 
    Same return type as dfs_maximizing, a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    raise NotImplementedError

# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=2))

def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. 
    Same return type as dfs_maximizing, a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    raise NotImplementedError


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True, time_limit=INF) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. 
    Returns anytime_value."""
    raise NotImplementedError


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented




#
# If you want to enter the tournament, implement your final contestant 
# in this function. You may write other helper functions/classes 
# but the function must take these arguments (though it can certainly ignore them)
# and must return an AnytimeValue.
#
def tournament_search(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True, time_limit=INF) :
    """Runs some kind of search (probably progressive deepening).
    Returns an AnytimeValue."""
    raise NotImplementedError

