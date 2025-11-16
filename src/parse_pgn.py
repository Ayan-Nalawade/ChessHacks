import chess.pgn
import os
from huggingface_hub import hf_hub_download

def parse_pgn_file(file_path, output_path):
    """
    Parses a PGN file and saves the move sequences of all games to a file.
    """
    with open(file_path) as pgn, open(output_path, "w") as output_file:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            
            moves = [move.uci() for move in game.mainline_moves()]
            output_file.write(" ".join(moves) + "\n")

if __name__ == "__main__":
    repo_id = "AN707/ChessHacks"
    pgn_filename = "lichess_elite_2024-04.pgn"
    
    # Download the PGN file from Hugging Face
    downloaded_pgn_path = hf_hub_download(repo_id=repo_id, filename=pgn_filename, repo_type="dataset")
    
    output_file_path = os.path.join(os.path.dirname(__file__), "..", "games.txt")
    parse_pgn_file(downloaded_pgn_path, output_file_path)
    print(f"Move sequences saved to {output_file_path}")
