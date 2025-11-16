import chess.pgn
import os
import json
from huggingface_hub import hf_hub_download

def train_model():
    """
    This function creates a placeholder model by processing a few games
    and saves it to a JSON file.
    """
    model = {}
    
    repo_id = "AN707/ChessHacks"
    pgn_filename = "lichess_elite_2024-04.pgn"
    
    # Download the PGN file from Hugging Face
    downloaded_pgn_path = hf_hub_download(repo_id=repo_id, filename=pgn_filename, repo_type="dataset")
    
    with open(downloaded_pgn_path) as pgn:
        for i in range(10000): # Limit to the first 10000 games for a small model
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            
            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()
                model[fen] = move.uci()
                board.push(move)

    model_path = os.path.join(os.path.dirname(__file__), "..", "model.json")
    with open(model_path, "w") as f:
        json.dump(model, f)
        
    print(f"Placeholder model saved to {model_path}")

if __name__ == "__main__":
    train_model()
