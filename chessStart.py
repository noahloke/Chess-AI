import chess
from chessboard import display

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def fen_to_bitboard(fen):
    board = chess.Board(fen)
    
    bitboard = ''

    wP = board.pieces(chess.PAWN, chess.WHITE)
    wN = board.pieces(chess.KNIGHT, chess.WHITE)
    wB = board.pieces(chess.BISHOP, chess.WHITE)
    wR = board.pieces(chess.ROOK, chess.WHITE)
    wQ = board.pieces(chess.QUEEN, chess.WHITE)
    wK = board.pieces(chess.KING, chess.WHITE)

    bP = board.pieces(chess.PAWN, chess.BLACK)
    bN = board.pieces(chess.KNIGHT, chess.BLACK)
    bB = board.pieces(chess.BISHOP, chess.BLACK)
    bR = board.pieces(chess.ROOK, chess.BLACK)
    bQ = board.pieces(chess.QUEEN, chess.BLACK)
    bK = board.pieces(chess.KING, chess.BLACK)

    bitboard += ''.join(['1' if 63 - i in wP else '0' for i in range(64)])
    bitboard += ''.join(['1' if 63 - i in wN else '0' for i in range(64)])
    bitboard += ''.join(['1' if 63 - i in wB else '0' for i in range(64)])
    bitboard += ''.join(['1' if 63 - i in wR else '0' for i in range(64)])
    bitboard += ''.join(['1' if 63 - i in wQ else '0' for i in range(64)])
    bitboard += ''.join(['1' if 63 - i in wK else '0' for i in range(64)])

    bitboard += ''.join(['1' if 63 - i in bP else '0' for i in range(64)])
    bitboard += ''.join(['1' if 63 - i in bN else '0' for i in range(64)])
    bitboard += ''.join(['1' if 63 - i in bB else '0' for i in range(64)])
    bitboard += ''.join(['1' if 63 - i in bR else '0' for i in range(64)])
    bitboard += ''.join(['1' if 63 - i in bQ else '0' for i in range(64)])
    bitboard += ''.join(['1' if 63 - i in bK else '0' for i in range(64)])

    bitboard += ''.join(['1' if board.turn == chess.WHITE else '0'])

    bitboard += ''.join(['1' if board.has_kingside_castling_rights(chess.WHITE) else '0',])
    bitboard += ''.join(['1' if board.has_queenside_castling_rights(chess.WHITE) else '0'])
    bitboard += ''.join(['1' if board.has_kingside_castling_rights(chess.BLACK) else '0'])
    bitboard += ''.join(['1' if board.has_queenside_castling_rights(chess.BLACK) else '0'])

    bitboard += ''.join(['1' if board.ep_square == 63 - i else '0' for i in range(64)]) if board.ep_square is not None else '0' * 64

    return bitboard

class EvaluationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(837, 837)
        self.fc2 = nn.Linear(837, 837)
        self.fc3 = nn.Linear(837, 837)
        self.fc4 = nn.Linear(837, 837)
        self.fc5 = nn.Linear(837, 837)
        self.fc6 = nn.Linear(837, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def load_model(model_path):
    model = EvaluationModel()
    state_dict = torch.load(model_path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model

def evaluation(board, model):
    bitboard = fen_to_bitboard(board.fen())
    bitboard = np.array(list(bitboard)).astype(np.float32)
    input_tensor = torch.tensor(bitboard)
    return model(input_tensor)

def minimax(board, depth, alpha, beta, white_turn, model):
    if depth == 0 or board.is_game_over():
        return evaluation(board, model)

    if white_turn:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, model)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, model)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def computer_move(board, depth, model):
    best_move = None
    best_value = float('-inf') if board.turn else float('inf')

    for move in board.legal_moves:
        board.push(move)
        value = minimax(board, depth - 1, float('-inf'), float('inf'), not board.turn, model)
        board.pop()

        if board.turn and value > best_value:
            best_value = value
            best_move = move
        elif not board.turn and value < best_value:
            best_value = value
            best_move = move

    return best_move

def main(player_color, depth):
    model = load_model("chess_model.pth")
    model.eval()
    
    board = chess.Board()
    window = display.start()

    if player_color == "black":
        display.flip(window)

    while not board.is_game_over():
        display.update(board.fen(), window)

        if board.turn == chess.WHITE:
            print("White to Play, ", end='')
        else:
            print("Black to Play, ", end='')

        if (board.turn == chess.WHITE and player_color == "white") or (board.turn == chess.BLACK and player_color == "black"):
            while True:
                try:
                    move_input = input("Please enter your move: ")
                    move = board.parse_san(move_input)
                    board.push(move)
                    break
                except ValueError as e:
                    print("Illegal Move, Please try again.")
        else:
            move = computer_move(board, depth, model)
            print(f"The computer plays: {board.san(move)}")
            board.push(move)

    display.update(board.fen(), window)
    print("Game Over: ", end='')
    if board.is_checkmate():
        print("Checkmate")
    elif board.is_stalemate():
        print("Stalemate")
    elif board.is_insufficient_material():
        print("Draw Due to Insufficient Material")
    elif board.is_seventyfive_moves():
        print("Draw Due to 75-Move Rule")
    elif board.is_fivefold_repetition():
        print("Draw Due to Fivefold Repetition")

    while True:
        if display.check_for_quit():
            break
    display.terminate()

if __name__ == "__main__":
    #player_color = "white"
    player_color = "black"
    depth = 3

    main(player_color, depth)