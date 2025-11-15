import torch
import numpy as np
import chess
from move_encoding import move_to_index, ACTION_SIZE

class Game():
    """
    Chess game implementation for AlphaZero-style training.
    """
    def __init__(self, fen=None):
        self.fen = fen if fen else chess.STARTING_FEN
        self.action_size = ACTION_SIZE  # 4672
        
    def boardToPlanes(self, board):
        """Convert chess board to 18x8x8 tensor representation"""
        planes = np.zeros((18, 8, 8), dtype=np.float32)
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
            rank = square // 8
            file = square % 8
            piece_type = piece.piece_type
            color = piece.color
            plane_idx = piece_to_plane[piece_type] + (0 if color == chess.WHITE else 6)
            planes[plane_idx, rank, file] = 1
        
        # Planes 12-15: Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[12, :, :] = 1
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[13, :, :] = 1
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[14, :, :] = 1
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[15, :, :] = 1
        
        # Plane 16: Side to move
        if board.turn == chess.WHITE:
            planes[16, :, :] = 1
        
        # Plane 17: En passant square
        if board.ep_square is not None:
            ep_rank = board.ep_square // 8
            ep_file = board.ep_square % 8
            planes[17, ep_rank, ep_file] = 1
        
        return planes

    def getInitBoard(self):
        """Returns starting board state as planes"""
        board = chess.Board(self.fen)
        return self.boardToPlanes(board)

    def getBoardSize(self):
        """Returns (planes, height, width)"""
        return (18, 8, 8)

    def getActionSize(self):
        """Returns number of possible actions"""
        return self.action_size

    def getNextState(self, board_planes, player, action):
        """
        Apply action to board and return new state.
        
        Args:
            board_planes: 18x8x8 numpy array
            player: 1 for white, -1 for black
            action: move index (0-4671)
            
        Returns:
            (nextBoard_planes, nextPlayer)
        """
        # Reconstruct chess.Board from planes
        board = self._planes_to_board(board_planes)
        
        # Convert action to move
        move = self._action_to_move(action, board)
        
        if move is None or move not in board.legal_moves:
            raise ValueError(f"Invalid action {action}")
        
        # Apply move
        board.push(move)
        
        # Return new board state and next player
        return self.boardToPlanes(board), -player

    def getValidMoves(self, board_planes, player):
        """
        Returns binary vector of valid moves.
        
        Args:
            board_planes: 18x8x8 numpy array
            player: 1 or -1 (not used since board encodes turn)
            
        Returns:
            valid_moves: binary numpy array of length 4672
        """
        board = self._planes_to_board(board_planes)
        valid_moves = np.zeros(self.action_size, dtype=np.float32)
        
        for move in board.legal_moves:
            action = move_to_index(board, move)
            if action is not None:
                valid_moves[action] = 1
        
        return valid_moves

    def getGameEnded(self, board_planes, player):
        """
        Check if game has ended.
        
        Args:
            board_planes: 18x8x8 numpy array
            player: 1 for white, -1 for black
            
        Returns:
            0 if game continues
            1 if player won
            -1 if player lost
            0.001 for draw (small non-zero value)
        """
        board = self._planes_to_board(board_planes)
        
        if not board.is_game_over():
            return 0
        
        result = board.result()
        
        if result == "1-0":  # White won
            return 1 if player == 1 else -1
        elif result == "0-1":  # Black won
            return -1 if player == 1 else 1
        else:  # Draw
            return 0.001

    def getCanonicalForm(self, board_planes, player):
        """
        Return board from perspective of player.
        For chess, we flip the board when it's black's turn.
        
        Args:
            board_planes: 18x8x8 numpy array
            player: 1 for white, -1 for black
            
        Returns:
            canonical_board: 18x8x8 numpy array from player's perspective
        """
        if player == 1:
            return board_planes
        
        # Flip board for black's perspective
        canonical = np.zeros_like(board_planes)
        
        # Flip piece planes (swap white/black pieces)
        canonical[0:6] = np.flip(board_planes[6:12], axis=1)  # Black pieces -> white
        canonical[6:12] = np.flip(board_planes[0:6], axis=1)  # White pieces -> black
        
        # Flip castling rights
        canonical[12] = np.flip(board_planes[14], axis=1)  # Black kingside -> white
        canonical[13] = np.flip(board_planes[15], axis=1)  # Black queenside -> white
        canonical[14] = np.flip(board_planes[12], axis=1)  # White kingside -> black
        canonical[15] = np.flip(board_planes[13], axis=1)  # White queenside -> black
        
        # Side to move (flip since we're changing perspective)
        canonical[16] = 1 - board_planes[16]
        
        # En passant square (flip vertically)
        canonical[17] = np.flip(board_planes[17], axis=1)
        
        return canonical

    def getSymmetries(self, board_planes, pi):
        """
        Chess has no symmetries that preserve move legality.
        Return original board and policy.
        """
        return [(board_planes, pi)]

    def stringRepresentation(self, board_planes):
        """Convert board to string for hashing in MCTS"""
        return board_planes.tobytes()

    def _planes_to_board(self, planes):
        """Reconstruct chess.Board from plane representation"""
        board = chess.Board(fen=None)  # Empty board
        board.clear()
        
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                       chess.ROOK, chess.QUEEN, chess.KING]
        
        # Place pieces
        for plane_idx in range(12):
            piece_type = piece_types[plane_idx % 6]
            color = chess.WHITE if plane_idx < 6 else chess.BLACK
            
            piece_positions = np.argwhere(planes[plane_idx] == 1)
            for rank, file in piece_positions:
                square = rank * 8 + file
                board.set_piece_at(square, chess.Piece(piece_type, color))
        
        # Set turn
        board.turn = chess.WHITE if planes[16, 0, 0] == 1 else chess.BLACK
        
        # Set castling rights
        board.castling_rights = 0
        if planes[12, 0, 0] == 1:
            board.castling_rights |= chess.BB_H1
        if planes[13, 0, 0] == 1:
            board.castling_rights |= chess.BB_A1
        if planes[14, 0, 0] == 1:
            board.castling_rights |= chess.BB_H8
        if planes[15, 0, 0] == 1:
            board.castling_rights |= chess.BB_A8
        
        # Set en passant
        ep_squares = np.argwhere(planes[17] == 1)
        if len(ep_squares) > 0:
            rank, file = ep_squares[0]
            board.ep_square = rank * 8 + file
        
        return board

    def _action_to_move(self, action, board):
        """Convert action index to chess.Move"""
        for move in board.legal_moves:
            if move_to_index(board, move) == action:
                return move
        return None