import random
import copy

class Agent:
    def __init__(self, player):
        self.player = player
    
    def act(self, state, remaining_time):
        raise NotImplementedError
    

class Agent3(Agent):
    def __init__(self, player, depth=8, max_killer_moves=5, futility=20):
        self.player = player
        self.depth = depth
        self.killer_moves = []
        self.max_killer_moves = max_killer_moves
        self.futility = futility  # seuil de futility pruning

    def evaluate(self, state):
        piece_weights = {
            1: 1,    # Soldat
            2: 5,    # Général
            3: 20    # Roi
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
        sorted_actions = sorted(state.actions(), key=lambda action: self.evaluate(state.result(action)), reverse=True)

        # Futility pruning: if the best possible value is below the futility threshold, prune the branch
        if value + self.futility < alpha:
            return value, best_action

        if self.killer_moves:
            sorted_actions = [a for a in sorted_actions if a in self.killer_moves] + \
                             [a for a in sorted_actions if a not in self.killer_moves]

        for action in sorted_actions:
            new_state = state.result(action)
            v, _ = self.min_value(new_state, alpha, beta, depth - 1)
            if v > value:
                value = v
                best_action = action
            alpha = max(alpha, value)
            if beta <= alpha:
                break

        if best_action is not None and best_action not in self.killer_moves:
            self.killer_moves.append(best_action)
            if len(self.killer_moves) > self.max_killer_moves:
                self.killer_moves.pop(0)

        return value, best_action

    def min_value(self, state, alpha, beta, depth):

        if state.is_terminal():
            return state.utility(self.player), None
        if depth == 0:
            return self.evaluate(state), None

        value = float('inf')
        best_action = None
        sorted_actions = sorted(state.actions(), key=lambda action: self.evaluate(state.result(action)))

        # Futility pruning: if the best possible value is above the futility threshold, prune the branch
        if value - self.futility > beta :
            return value, best_action

        if self.killer_moves:
            sorted_actions = [a for a in sorted_actions if a in self.killer_moves] + \
                             [a for a in sorted_actions if a not in self.killer_moves]

        for action in sorted_actions:
            new_state = state.result(action)
            v, _ = self.max_value(new_state, alpha, beta, depth - 1)
            if v < value:
                value = v
                best_action = action
            beta = min(beta, value)
            if beta <= alpha:
                break

        if best_action is not None and best_action not in self.killer_moves:
            self.killer_moves.append(best_action)
            if len(self.killer_moves) > self.max_killer_moves:
                self.killer_moves.pop(0)

        return value, best_action

    def act(self, state, remaining_time):
        if state.to_move() == self.player:
            _, action = self.max_value(state, -float('inf'), float('inf'), self.depth)
        else:
            _, action = self.min_value(state, -float('inf'), float('inf'), self.depth)

        # Sécurité finale
        if action is None or action not in state.actions():
            print("⚠️ Action None ou invalide retournée ! Choix par défaut.")
            actions = state.actions()
            if actions:
                action = actions[0]

        return action
    



class MCTSAgent(Agent):
    def __init__(self, player, simulations=10):
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
from agent import Agent
import math
import random
import time
from collections import defaultdict

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = state.actions()
        self.action_visits = defaultdict(int)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4): # UCT
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.result(action)
        child_node = Node(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward

class MCTSAgent2(Agent):
    def __init__(self, rollout_depth=10, max_time=1.0):
        self.rollout_depth = rollout_depth
        self.max_time = max_time

    def act(self, state, remaining_time):
        root = Node(state)
        start_time = time.time()

        while time.time() - start_time < min(self.max_time, remaining_time):
            node = root
            # SELECTION
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # EXPANSION with Progressive Widening
            if node.untried_actions and len(node.children) < math.sqrt(node.visits + 1):
                node = node.expand()

            # SIMULATION
            reward = self.rollout(node.state, state.to_move())

            # BACKPROPAGATION
            while node is not None:
                node.update(reward)
                node = node.parent

        return max(root.children, key=lambda n: n.visits).action

    def rollout(self, state, player):
        current_state = state
        depth = 0
        while not current_state.is_terminal() and depth < self.rollout_depth:
            actions = current_state.actions()
            actions = self.prioritize_actions(actions, current_state)
            action = random.choice(actions[:min(len(actions), 5)])  # limit to top 5
            current_state = current_state.result(action)
            depth += 1
        return self.evaluate_state(current_state, player)

    def prioritize_actions(self, actions, state):
        def action_score(action):
            to_pos = action.end
            promotion_bonus = 0
            neighbors = [(to_pos[0] + dx, to_pos[1] + dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]]
            for n in neighbors:
                if n in state.pieces and state.pieces[n] == state.to_move():
                    promotion_bonus += 0.5
            return promotion_bonus
        return sorted(actions, key=action_score, reverse=True)

    def evaluate_state(self, state, player):
        opponent = -player
        score = 0
        center_weight = 0.2

        for (r, c), piece in state.pieces.items():
            abs_val = abs(piece)
            value = 1 if abs_val == 1 else 2 if abs_val == 2 else 3
            if piece * player > 0:
                score += value
                score += center_weight * (3.5 - abs(3.5 - r)) * (4 - abs(4 - c))
            elif piece * opponent > 0:
                score -= value

        if state.can_create_king and state.current_player == opponent:
            score -= 5
        compactness_bonus = 0

        for (r, c), piece in state.pieces.items():
            if piece * player > 0:
                neighbors = [(r + dx, c + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
                for nr, nc in neighbors:
                    if (nr, nc) in state.pieces and state.pieces[(nr, nc)] * player > 0:
                        compactness_bonus += 0.2

        score += compactness_bonus

        return score
    
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = state.actions()
        self.action_visits = defaultdict(int)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4): # UCT
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.result(action)
        child_node = Node(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward

class MCTSAgent3(Agent): # C'est lui le big boss !!!!
    def __init__(self, rollout_depth=5, max_time=5):
        self.rollout_depth = rollout_depth
        self.max_time = max_time

    def act(self, state, remaining_time):
        root = Node(state)
        start_time = time.time()
        player = state.to_move()
        actions = state.actions()

        # Revive le roi des que possible
        if state.can_create_king and state.turn >= 5:
            king_res = []
            for action in actions:
                from_val = state.pieces.get(action.start, 0)
                to_val = state.pieces.get(action.end, 0)

                # Roi = général + soldat du même camp
                values = sorted([abs(from_val), abs(to_val)])
                if from_val * to_val > 0 and values == [1, 2]:
                    king_res.append(action)
            
            actions = self.prioritize_actions(king_res, state)
            if actions:
                return actions[0]
        
        time_budget = min(self.max_time, max(1.0, remaining_time / 60)) 
        while time.time() - start_time < time_budget:
            node = root
            # SELECTION
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # EXPANSION with Progressive Widening
            if node.untried_actions and len(node.children) < math.sqrt(node.visits + 1):
                node = node.expand()

            # SIMULATION
            reward = self.rollout(node.state, state.to_move())

            # BACKPROPAGATION
            while node is not None:
                node.update(reward)
                node = node.parent

        return max(root.children, key=lambda n: n.visits).action

    def rollout(self, state, player):
        current_state = state
        depth = 0
        while not current_state.is_terminal() and depth < self.rollout_depth:
            actions = current_state.actions()
            actions = self.prioritize_actions(actions, current_state)
            top_k = 10 if depth < self.rollout_depth / 2 else 5
            action = random.choice(actions[:min(len(actions), top_k)])  # limit to top top_k
            current_state = current_state.result(action)
            depth += 1
        return self.evaluate_state(current_state, player)

    def prioritize_actions(self, actions, state):
        def action_score(action):
            to_pos = action.end
            from_pos = action.start
            score = 0

            #  Favoriser les promotions 
            from_val = state.pieces.get(from_pos, 0)
            to_val = state.pieces.get(to_pos, 0)
            if from_val * to_val > 0:  # Deux pièces du même camp
                stack_val = abs(from_val + to_val)
                if stack_val == 2:
                    score += 5

            #  Favoriser les captures (surtout roi/général)
            for r_captured in action.removed:
                captured_val = abs(state.pieces.get(r_captured, 0))
                score += 0.5 * captured_val  

            #  Éviter d’isoler une pièce
            isolated_penalty = 0
            nearby_allies = sum(1 for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)] 
                                if (to_pos[0]+dx, to_pos[1]+dy) in state.pieces and 
                                state.pieces[(to_pos[0]+dx, to_pos[1]+dy)] == state.to_move())
            if nearby_allies == 0:
                isolated_penalty = -0.5

            return score + isolated_penalty

        return sorted(actions, key=action_score, reverse=True)


    def evaluate_state(self, state, player):
        opponent = -player
        score = 0
        center_weight = 0.1
        border_bonus = 0
        king_exposed_penalty = 0
        fortress_bonus = 0

        if state.is_terminal():
            utility = state.utility(player)
            if utility == 1:
                return 30
            elif utility == -1:
                return -30
            else:
                return 0

        if not state._has_king and state.turn >= 5:
            score -= 10

        # Évaluer la phase de jeu
        total_pieces = len(state.pieces)
        if total_pieces > 18:
            phase = "early"
        elif total_pieces > 10:
            phase = "mid"
        else:
            phase = "late"

        for (r, c), piece in state.pieces.items():
            abs_val = abs(piece)
            value = 1 if abs_val == 1 else 10 if abs_val == 2 else 15

            if piece * player > 0:
                # Valeur brute
                score += value

                # Bonus position centrale
                score += center_weight * (3.5 - abs(3.5 - r)) * (4 - abs(4 - c))

                # Bonus pour pièce sur le bord
                if r == 0 or r == 6 or c == 0 or c == 7:
                    border_bonus += 0.3

                # Bonus pour structure en carré (forteresse)
                if ((r+1, c) in state.pieces and state.pieces[(r+1, c)] * player > 0 and
                    (r, c+1) in state.pieces and state.pieces[(r, c+1)] * player > 0 and
                    (r+1, c+1) in state.pieces and state.pieces[(r+1, c+1)] * player > 0):
                    fortress_bonus += 0.6

                # Roi : pénaliser s’il est au front selon la phase
                if abs_val == 3:
                    # roi proche de son coin à lui
                    def is_adjacent(pos1, pos2):
                        return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1

                    target_corner = (0, 0) if player == 1 else (6, 7)

                    if phase == "early":
                        if is_adjacent((r, c), target_corner):
                            king_exposed_penalty += 3

                    if phase in ["early", "mid"]:
                            if player == 1 and not (0 <= r <= 2 and 0 <= c <= 2):
                                king_exposed_penalty -= 2  
                            elif player == -1 and not (4 <= r <= 6 and 5 <= c <= 7):
                                king_exposed_penalty -= 2  

            elif piece * opponent > 0:
                score -= value

        # résurrection du roi adverse
        if state.can_create_king and state.current_player == opponent:
            score -= 1

        if not state.is_terminal() and 3 * opponent not in state.pieces.values():
            score += 10  

        # Total des bonus/penalités
        score += border_bonus + fortress_bonus + king_exposed_penalty

        

        return score



