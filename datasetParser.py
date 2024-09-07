import json
import csv
import chess

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
    bitboard += ''.join(['1' if  63 - i in wB else '0' for i in range(64)])
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
    
def parse_jsonl(file_path, output_csv_prefix):
    file_count = 1
    line_count = 0
    max_lines = 500000

    csv_file = open(f"{output_csv_prefix}_{file_count}.csv", 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['BITBOARD', 'EVAL'])

    with open(file_path, 'r') as file:
        for line in file:
            if line_count == max_lines:
                csv_file.close()
                file_count += 1
                line_count = 0
                csv_file = open(f"{output_csv_prefix}_{file_count}.csv", 'w', newline='')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['BITBOARD', 'EVAL'])

            try:
                data = json.loads(line)
                fen = data['fen']
                bitboard = fen_to_bitboard(fen)

                if bitboard is None:
                    continue

                eval_value = None
                for eval in data['evals']:
                    for pv in eval['pvs']:
                        if 'mate' in pv:
                            mate = pv['mate']
                            eval_value = 16 if mate > 0 else -16
                            break
                        if 'cp' in pv:
                            cp_value = pv['cp']
                            if cp_value > 15:
                                cp_value = 15
                            elif cp_value < -15:
                                cp_value = -15
                            eval_value = cp_value
                            break
                    if eval_value is not None:
                        break

                eval_value /= 16
                csv_writer.writerow([bitboard, eval_value])
                line_count += 1

            except json.JSONDecodeError:
                print("Failed to decode a line")
            except KeyError:
                print("Necessary data missing in the line")

    csv_file.close()

file_path = 'lichess_db_eval.jsonl'
output_csv_prefix = 'data'
parse_jsonl(file_path, output_csv_prefix)