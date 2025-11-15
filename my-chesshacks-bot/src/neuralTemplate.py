import torch
from chessModel import ChessModel
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tools import dotdict


DEFAULT_ARGS = dotdict({
    'lr': 1e-3,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 256,
})

class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.board_x, self.board_y = game.getBoardSize()[1:]
        self.action_size = game.getActionSize()
        
        self.model = ChessModel(game, args)
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        #Train network on batch of samples
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        for epoch in range(self.args.epochs):
            self.model.train()
            
            batch_idx = 0
            total_loss = 0
            policy_loss_total = 0
            value_loss_total = 0
            
            # Shuffle examples
            np.random.shuffle(examples)
            
            # Train on batches
            for i in range(0, len(examples), self.args.batch_size):
                batch = examples[i:i + self.args.batch_size]
                
                # Prepare batch
                boards, pis, vs = list(zip(*batch))
                boards = torch.FloatTensor(np.array(boards)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).to(self.device)
                
                # Forward pass
                policy_logits, values = self.model(boards)
                
                # Compute loss
                value_loss = F.mse_loss(values.view(-1), target_vs)
                policy_loss = F.cross_entropy(policy_logits, target_pis)
                loss = value_loss + policy_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track losses
                total_loss += loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                batch_idx += 1
            
            # Print epoch stats
            avg_loss = total_loss / batch_idx
            avg_policy_loss = policy_loss_total / batch_idx
            avg_value_loss = value_loss_total / batch_idx
            
            print(f'Epoch {epoch+1}/{self.args.epochs} | '
                  f'Loss: {avg_loss:.4f} | '
                  f'Policy: {avg_policy_loss:.4f} | '
                  f'Value: {avg_value_loss:.4f}')

        pass

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        self.model.eval()
        
        with torch.no_grad():
            board = torch.FloatTensor(board).unsqueeze(0).to(self.device)
            policy_logits, value = self.model(board)
            
            # Convert logits to probabilities
            pi = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            v = value.cpu().numpy()[0][0]
        
        return pi, v

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        import os
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        import os
        filepath = os.path.join(folder, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")


class NNetWrapper(NeuralNet):
    """
    Compatibility wrapper so legacy training code can request `NNetWrapper`.
    """

    def __init__(self, game, args=None):
        super().__init__(game, args or DEFAULT_ARGS)
