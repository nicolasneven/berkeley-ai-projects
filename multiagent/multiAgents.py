# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent,Actions
import math

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return childGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        def MinValue(state,depth,ghost_num):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = math.inf

            for action in state.getLegalActions(ghost_num):
                if ghost_num == gameState.getNumGhost(): ## last ghost
                    v = min(v, MaxValue(state.getNextState(ghost_num, action), depth + 1))
                else:
                    v = min(v, MinValue(state.getNextState(ghost_num, action), depth, ghost_num + 1))

            return v






        def MaxValue(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = -math.inf
            for action in state.getLegalActions(0):
                v = max(v, MinValue(state.getNextState(0, action),depth,1))
            return v


        action_score = []
        for action in gameState.getLegalActions(0):
            action_score.append((action,MinValue(gameState.getNextState(0, action), 0, 1)))

        sorted_action = sorted(action_score, key = lambda x: x[1],reverse=True)

        return sorted_action[0][0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()


        def MinValue(state,depth,ghost_num,alpha,beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = math.inf

            for action in state.getLegalActions(ghost_num):
                if ghost_num == gameState.getNumGhost(): ## last ghost
                    retV = MaxValue(state.getNextState(ghost_num, action), depth + 1,alpha,beta)
                    if type(retV) is tuple:
                        retV = retV[1]
                    v = min(v,retV)
                    if v < alpha:
                        return v
                    beta = min(v,beta)
                else:
                    retV = MinValue(state.getNextState(ghost_num, action), depth, ghost_num + 1,alpha,beta)
                    if type(retV) is tuple:
                        retV = retV[1]
                    v = min(v, retV )
                    if v < alpha:
                        return (action,v)
                    beta = min(v,beta)

            return (action,v)






        def MaxValue(state, depth,alpha,beta):
            cur_action = None
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = -math.inf
            for action in state.getLegalActions(0):
                retV =MinValue(state.getNextState(0, action),depth,1,alpha,beta)
                if type(retV) is tuple:
                    retV = retV[1]
                if retV >=v: # get new high value for max and  associated action
                    v = retV
                    cur_action = action
                if v > beta:

                    return (cur_action,v)
                alpha = max(alpha,v)
            return (cur_action,v)



        alpha = -1*math.inf
        beta = math.inf

        v = MaxValue(gameState, 0, alpha,beta)

        return v[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()


        def ExpMinValue(state, depth, ghost_num):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = 0

            prob = 1/ len(gameState.getLegalActions(ghost_num))

            for action in state.getLegalActions(ghost_num):
                if ghost_num == gameState.getNumGhost():  ## last ghost
                    retV = MaxValue(state.getNextState(ghost_num, action), depth + 1)
                else:
                    retV = ExpMinValue(state.getNextState(ghost_num, action), depth, ghost_num + 1)

                if type(retV) is tuple:
                    retV = retV[1]


                v += retV

            v *= prob
            return (action,v)

        def MaxValue(state, depth):

            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = -math.inf
            cur_action = None
            for action in state.getLegalActions(0):
                retV = ExpMinValue(state.getNextState(0, action), depth, 1)

                if type(retV) is tuple:
                   retV = retV[1]
                if retV >=v: # get new high value for max and  associated action
                    v = retV
                    cur_action = action
            return (cur_action,v)

        retV = MaxValue(gameState, 0)
        return retV[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    position = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    ghostStates = currentGameState.getGhostStates()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()

    if currentGameState.isWin(): return math.inf
    if currentGameState.isLose(): return -math.inf

    min_ghost = math.inf
    for ghost in ghostStates:
        d = manhattanDistance(position, ghost.getPosition())
        min_ghost = min(d,min_ghost)
        if ghost.scaredTimer > 6 and d < 2:
            return math.inf
        elif ghost.scaredTimer < 5 and d < 2:
            return -math.inf

    # Distance to closest food pellet
    # Note that at least one food pellet must exist,
    # otherwise we would have already won!
    foodDistance = 1.0/closestItemDistance(currentGameState, foodList)

    # Distance to closest capsule
    capsuleDistance = closestItemDistance(currentGameState, capsuleList)
    capsuleDistance = 0.0 if capsuleDistance is None else 1.0/capsuleDistance

    return 10.0*foodDistance + 5.0*score + 0.5*capsuleDistance - 1.0 * min_ghost


## bfs to find cloest maze distance in the item list
def closestItemDistance(currentGameState, items):
    """Returns the maze distance to the closest item present in items"""

    # BFS to find the maze distance from position to closest item
    walls = currentGameState.getWalls()

    start = currentGameState.getPacmanPosition()

    # Dictionary storing the maze distance from start to any given position
    distance = {start: 0}

    # Set of visited positions in order to avoid revisiting them again
    visited = {start}

    queue = util.Queue()
    queue.push(start)

    while not queue.isEmpty():
        position = x, y = queue.pop()
        if position in items: return distance[position]
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            next_position = nextx, nexty = int(x + dx), int(y + dy)

            if not walls[nextx][nexty] and next_position not in visited:
                queue.push(next_position)
                visited.add(next_position)
                # A single action separates position from next_position, so the distance is 1
                distance[next_position] = distance[position] + 1

    return None

# Abbreviation
better = betterEvaluationFunction
