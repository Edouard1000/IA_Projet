import random
import copy

class Agent:
    def __init__(self, player):
        self.player = player
    
    def act(self, state, remaining_time):
        raise NotImplementedError
    



class MCTSAgent(Agent):
    def __init__(self, player, simulations=1):
        super().__init__(player)
        self.simulations = simulations  # nombre de simulations K

    def act(self, state, remaining_time):
        actions = state.actions()[:len(state.actions())]
        

        best_action = None
        best_score = float('-inf')

        for action in actions: # selection
            total_score = 0

            for _ in range(self.simulations):  
                
                sim_state = copy.deepcopy(state)
                sim_state = sim_state.result(action) # expansion

                # Simulation 
                while not sim_state.is_terminal():
                    sim_actions = sim_state.actions()
                    if not sim_actions:
                        break
                    sim_action = random.choice(sim_actions)
                    sim_state = sim_state.result(sim_action)

                # backpropagation
                score = sim_state.utility(self.player)
                total_score += score

            avg_score = total_score / self.simulations

            if avg_score > best_score:
                best_score = avg_score
                best_action = action

        return best_action

class Agent2:
    def __init__(self, player, depth=2):
        self.player = player
        self.depth = depth

    def evaluate(self, state):
        piece_weights = {
            1: 1,   # Soldat
            2: 5,   # Général
            3: 20   # Roi
        }

        score = 0
        for (pos, value) in state.pieces.items():
            owner = 1 if value > 0 else -1
            piece_type = abs(value)
            weight = piece_weights.get(piece_type, 0)
            score += weight if owner == self.player else -weight

        return score

    def max_value(self, state, alpha, beta, depth):
        if state.is_terminal():
            return state.utility(self.player), None
        if depth == 0:
            return self.evaluate(state), None

        value = -float('inf')
        best_action = None
        for action in state.actions():
            new_state = state.result(action)
            v, _ = self.min_value(new_state, alpha, beta, depth - 1)
            if v > value:
                value = v
                best_action = action
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value, best_action

    def min_value(self, state, alpha, beta, depth):
        if state.is_terminal():
            return state.utility(self.player), None
        if depth == 0:
            return self.evaluate(state), None

        value = float('inf')
        best_action = None
        for action in state.actions():
            new_state = state.result(action)
            v, _ = self.max_value(new_state, alpha, beta, depth - 1)
            if v < value:
                value = v
                best_action = action
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_action

    def act(self, state, remaining_time):
        if state.to_move() == self.player:
            _, action = self.max_value(state, -float('inf'), float('inf'), self.depth)
        else:
            _, action = self.min_value(state, -float('inf'), float('inf'), self.depth)
        return action