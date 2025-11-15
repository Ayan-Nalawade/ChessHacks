import torch
import numpy as np
import chess

class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, fen):
        self.fen = fen
        self.planes = None
        
    def boardToPlanes(self, board):
        planes = np.zeros((18,8,8),dtype=np.float32)
        piece_to_plane = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        # Planes 0-11: Piece positions
        piece_map = board.piece_map()
        for square, piece in piece_map.items():
            rank = square // 8  # 0-7 (bottom to top)
            file = square % 8   # 0-7 (left to right)
            
            piece_type = piece.piece_type
            color = piece.color
            
            # White pieces: planes 0-5, Black pieces: planes 6-11
            plane_idx = piece_to_plane[piece_type] + (0 if color == chess.WHITE else 6)
            planes[plane_idx, rank, file] = 1
        
        # Plane 12: White kingside castling
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[12, :, :] = 1
        
        # Plane 13: White queenside castling
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[13, :, :] = 1
        
        # Plane 14: Black kingside castling
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[14, :, :] = 1
        
        # Plane 15: Black queenside castling
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[15, :, :] = 1
        
        # Plane 16: Side to move (1 if White's turn, 0 if Black's turn)
        if board.turn == chess.WHITE:
            planes[16, :, :] = 1
        
        # Plane 17: En passant square
        if board.ep_square is not None:
            ep_rank = board.ep_square // 8
            ep_file = board.ep_square % 8
            planes[17, ep_rank, ep_file] = 1
        
        return planes

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        board = chess.Board(self.fen) #Starting position
        self.planes = self.boardToPlanes
        return self.planes
        #Convert fen into 18x8x8
        

    def getBoardSize(self):
        """
        Returns:
            (x,y,z): a tuple of board dimensions
        """
        return self.planes.shape()

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 4672 #The number of moves we will design for (typical)

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        pass

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass