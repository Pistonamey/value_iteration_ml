#Amey Shinde
#1001844387

def load_environment(file_path):
    environment = []
    with open(file_path, 'r') as file:
        for line in file:
            # Remove newline characters and split the line by comma to get the elements
            elements = line.strip().split(',')
            environment.append(elements)
    return environment

def value_iteration(environment_file, non_terminal_reward, gamma, K):
    """
    Perform the value iteration algorithm on a grid environment to determine the optimal policy.

    Args:
    environment_file (str): Path to a file containing the grid environment. The environment is 
                            expected to be a grid where each cell can be '.', representing a 
                            free state, 'X', representing a blocked state, or a numeric value 
                            representing a terminal state's reward.
    non_terminal_reward (float): The reward received for non-terminal states.
    gamma (float): The discount factor used in the value iteration algorithm.
    K (int): The number of iterations to perform in the value iteration algorithm.

    The function reads the environment from a file, initializes utilities of states to zero, and 
    then iteratively updates the utilities based on the Bellman equation over K iterations. The 
    Bellman equation considers the actions that can be taken from each state, the probability of 
    each action leading to various outcomes, and the utilities of those outcomes.

    The function also calculates and prints the optimal policy based on the final utilities. The 
    policy maps each state to an action that maximizes the expected utility. The function prints 
    both the final utilities of each state and the optimal actions at each state in the grid.

    Returns:
    None: This function does not return any value. It prints the utilities and optimal policy 
          for the given environment directly.
    """
    # Load the environment from the file
    environment = load_environment(environment_file)
    
    # Define action probabilities
    action_probs = {
        'up': {'up': 0.8, 'left': 0.1, 'right': 0.1},
        'down': {'down': 0.8, 'left': 0.1, 'right': 0.1},
        'left': {'left': 0.8, 'up': 0.1, 'down': 0.1},
        'right': {'right': 0.8, 'up': 0.1, 'down': 0.1}
    }

    # Define a mapping from actions to symbols
    action_symbols = {
        'up': '^',
        'down': 'v',
        'left': '<',
        'right': '>'
    }
    
    # Initialize utilities
    U = {s: 0 for s in all_states(environment)}
    
    # Main loop
    for k in range(K):
        U_prime = U.copy()
        for s in all_states(environment):
            if environment[s[0]][s[1]] in ('1.0', '-1.0'):  # Check for terminal states
                U_prime[s] = float(environment[s[0]][s[1]])
            elif environment[s[0]][s[1]] != 'X':  # Skip blocked states
                action_values = []
                for action in actions(s, environment):
                    action_value = sum(action_probs[action][outcome] * U[next_states(s, outcome, environment)]
                                       for outcome in action_probs[action])
                    action_values.append(non_terminal_reward + gamma * action_value)
                U_prime[s] = max(action_values)
        U = U_prime.copy()
    
    # Compute the policy
    policy = {s: None for s in all_states(environment)}
    for s in all_states(environment):
        if environment[s[0]][s[1]] == '.':
            # Calculate the expected utility for each action
            action_utilities = {}
            for action in actions(s, environment):
                action_utilities[action] = sum(action_probs[action][outcome] * U[next_states(s, outcome, environment)]
                                               for outcome in action_probs[action])

            # Find the action with the maximum expected utility and map it to its symbol
            best_action = max(action_utilities, key=action_utilities.get)
            policy[s] = action_symbols[best_action]
        elif environment[s[0]][s[1]] == 'X':
            # Blocked states are marked with 'x'
            policy[s] = 'x'
        else:
            # Terminal states are marked with 'o'
            policy[s] = 'o'
    
    # Print utilities and policy
    print_utilities_and_policy(U, policy, environment)



def print_utilities_and_policy(U, policy, environment):
    rows = len(environment)  # assuming environment is a list of lists
    cols = len(environment[0]) if rows else 0

    # Printing utilities
    print("utilities:")
    for i in range(rows):
        for j in range(cols):
            # Check for blocked states
            if environment[i][j] == 'X':
                print("%6.3f" % 0, end=' ')
            else:
                state = (i, j)
                print("%6.3f" % U[state], end=' ')
        print()  # Newline for next row

    # Printing policy
    print("\npolicy:")
    for i in range(rows):
        for j in range(cols):
            state = (i, j)
            # Check for blocked and terminal states
            if environment[i][j] == 'X':
                print("%6s" % 'x', end=' ')
            elif environment[i][j] in ('1.0', '-1.0'):  # Assuming these are the terminal states' rewards
                print("%6s" % 'o', end=' ')
            else:
                print("%6s" % policy[state], end=' ')
        print()  # Newline for next row





def all_states(environment):
    """
    Generate a list of all possible states in a grid environment.

    Args:
    environment (list of list of str): A 2D grid environment represented as a list of lists,
                                       where each sublist represents a row in the environment.
    
    Returns:
    list of tuples: A list of tuples, where each tuple represents a state (i, j) in the grid.
                    Here, i is the row index and j is the column index in the grid.
    """
    rows = len(environment)
    cols = len(environment[0]) if rows else 0
    all_states_list = []

    for i in range(rows):
        for j in range(cols):
            all_states_list.append((i, j))

    return all_states_list


def non_terminal_states(environment):
    """
    Identify all non-terminal states in a grid environment.

    Args:
    environment (list of list of str): A 2D grid environment represented as a list of lists.
                                       Each cell in the grid can be '.', representing a free cell,
                                       or 'X', representing a blocked or terminal cell.

    Returns:
    list of tuples: A list containing tuples of non-terminal (free) states. Each tuple represents
                    a coordinate (i, j) in the grid, where i is the row index and j is the column index.
    """
    non_terminals = []
    rows = len(environment)
    cols = len(environment[0]) if rows else 0
    
    for i in range(rows):
        for j in range(cols):
            # Add the state as a non-terminal if it is not a terminal state or blocked
            if environment[i][j] == '.':
                non_terminals.append((i, j))
    return non_terminals

def next_states(state, action, environment):
    """
    Determine the next state in the environment given a current state and an action.

    Args:
    state (tuple): The current state represented as a tuple (i, j), where i is the row index
                   and j is the column index in the grid.
    action (str): The action to be taken from the current state. Actions include 'up', 'down',
                  'left', and 'right'.
    environment (list of list of str): The 2D grid environment, where each cell can be either '.'
                                       (free) or 'X' (blocked or terminal).

    Returns:
    tuple: The next state as a tuple (i, j) after performing the action. If the action leads to
           a blocked cell or out of bounds, the current state is returned.
    """
    rows = len(environment)
    cols = len(environment[0]) if rows else 0
    i, j = state

    # Define the possible moves
    moves = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1),
    }

    # Get the change in coordinates corresponding to the given action
    di, dj = moves[action]

    # Calculate the new potential coordinates
    new_i, new_j = i + di, j + dj

    # Check if the new coordinates are within the grid bounds and not a blocked state
    if 0 <= new_i < rows and 0 <= new_j < cols and environment[new_i][new_j] != 'X':
        return (new_i, new_j)
    else:
        # If it's a blocked state or out of bounds, stay in the current state
        return state


def actions(state, environment):
    """
    Determine the valid actions that can be taken from a given state in the environment.

    Args:
    state (tuple): The current state represented as a tuple (i, j), where i is the row index
                   and j is the column index in the grid.
    environment (list of list of str): A 2D grid environment where each cell can be either '.'
                                       (free) or 'X' (blocked).

    Returns:
    list of str: A list of valid actions ('up', 'down', 'left', 'right') that can be taken from
                 the current state. Actions that would lead to a blocked state or move out of the
                 grid boundaries are excluded.
    """
    # Define all possible actions
    possible_actions = ['up', 'down', 'left', 'right']
    i, j = state
    rows = len(environment)
    cols = len(environment[0]) if rows else 0

    # Remove actions that lead to a blocked state or move out of the grid
    if i == 0 or environment[i-1][j] == 'X':
        possible_actions.remove('up')
    if i == rows - 1 or environment[i+1][j] == 'X':
        possible_actions.remove('down')
    if j == 0 or environment[i][j-1] == 'X':
        possible_actions.remove('left')
    if j == cols - 1 or environment[i][j+1] == 'X':
        possible_actions.remove('right')

    return possible_actions


def expected_utility(action, state, U, environment):
    """
    Calculate the expected utility of taking a given action from a state in the environment.

    Args:
    action (str): The action to be evaluated ('up', 'down', 'left', 'right').
    state (tuple): The current state represented as a tuple (i, j), where i is the row index
                   and j is the column index in the grid.
    U (dict): A dictionary mapping states (tuples) to their respective utilities.
    environment (list of list of str): A 2D grid environment, where each cell can be either '.'
                                       (free) or 'X' (blocked).

    Returns:
    float: The expected utility of performing the given action in the given state, considering
           the probabilities of ending up in different states as a result of the action.
    """
    # Define the move probabilities
    move_probs = {
        'up': {'up': 0.8, 'left': 0.1, 'right': 0.1},
        'down': {'down': 0.8, 'left': 0.1, 'right': 0.1},
        'left': {'left': 0.8, 'up': 0.1, 'down': 0.1},
        'right': {'right': 0.8, 'up': 0.1, 'down': 0.1},
    }
    
    # Initialize expected utility
    exp_utility = 0.0

    # Get all possible actions from the current state
    for act in move_probs[action]:
        # Probability of the intended action
        prob = move_probs[action][act]

        # Get the next state given the action
        next_state = next_states(state, act, environment)

        # Add the expected utility of the action to the total
        exp_utility += prob * U[next_state]

    return exp_utility
