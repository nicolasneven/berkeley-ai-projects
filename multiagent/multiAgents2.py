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

from game import Agent

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
    Your minimax agent (question 1)
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

        gameState.getNumGhost():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = -float('inf')
            LegalActions = state.getLegalActions(0)  # action of pacman
            for action in LegalActions:
                v = max(v, min_value(state.getNextState(0, action), depth, 1))
            return v

        def min_value(state, depth, ghostIdx):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = float('inf')
            LegalActions = state.getLegalActions(ghostIdx)  # action of ghost of ghostIdx
            for action in LegalActions:
                newState = state.getNextState(ghostIdx, action)
                if ghostIdx == state.getNumGhost():  # if this is the last ghost
                    v = min(v, max_value(newState, depth+1))
                else:
                    v = min(v, min_value(newState, depth, ghostIdx+1))
            return v

        v = -float('inf')
        LegalActions = gameState.getLegalActions(0)
        best_a = LegalActions[0]
        for action in LegalActions:
            new_v = min_value(gameState.getNextState(0, action), 0, 1)
            if new_v > v:
                v = new_v
                best_a = action
        return best_a


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = -float('inf')
            LegalActions = state.getLegalActions(0)  # action of pacman
            for action in LegalActions:
                newState = state.getNextState(0, action)
                v = max(v, min_value(newState, depth, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, depth, ghostIdx, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = float('inf')
            LegalActions = state.getLegalActions(ghostIdx)  # action of ghost of ghostIdx
            for action in LegalActions:
                newState = state.getNextState(ghostIdx, action)
                if ghostIdx == gameState.getNumGhost():  # if this is the last ghost
                    v = min(v, max_value(newState, depth + 1, alpha, beta))
                else:
                    v = min(v, min_value(newState, depth, ghostIdx + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        alpha, beta = -float('inf'), float('inf')
        v = -float('inf')
        LegalActions = gameState.getLegalActions(0)
        best_a = LegalActions[0]
        for action in LegalActions:
            newState = gameState.getNextState(0, action)
            new_v = min_value(newState, 0, 1, alpha, beta)
            if new_v > v:
                v = new_v
                best_a = action
            alpha = max(alpha, v)
        return best_a


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = -float('inf')
            LegalActions = state.getLegalActions(0)  # action of pacman
            for action in LegalActions:
                v = max(v, exp_value(state.getNextState(0, action), depth, 1))
            return v

        def exp_value(state, depth, ghostIdx):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = 0
            LegalActions = state.getLegalActions(ghostIdx)
            p = 1.0 / len(LegalActions)
            for action in LegalActions:
                newState = state.getNextState(ghostIdx, action)
                if ghostIdx == gameState.getNumGhost():  # if this is the last ghost
                    v += max_value(newState, depth+1) * p
                else:
                    v += exp_value(newState, depth, ghostIdx + 1) * p
            return v

        LegalActions = gameState.getLegalActions(0)
        v = -float('inf')
        best_a = LegalActions[0]
        for action in LegalActions:
            new_v = exp_value(gameState.getNextState(0, action), 0, 1)
            if new_v > v:
                v = new_v
                best_a = action
        return best_a


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 4).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    closestFoodDis = min(manhattanDistance(position, food) for food in foods) if foods else 0.5
    capsuleList = currentGameState.getCapsules()
    closestCapsuleDis = min(manhattanDistance(position, capsule) for capsule in capsuleList) if capsuleList else 0.5
    score = currentGameState.getScore()

    evaluation = 10 * 1.0 / closestFoodDis + 35 * 1.0 / closestCapsuleDis + 5.0 * score
    return evaluation


# Abbreviation
better = betterEvaluationFunction
