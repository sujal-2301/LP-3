# Title: Design 8-Queens Matrix Having First Queen Placed
# Objective: Use backtracking to place remaining queens so that no two attack each other

# Size of the chessboard
N = 8

def print_board(board):
    """Function to print the chessboard in a readable format"""
    for row in board:
        print(" ".join("Q" if x else "." for x in row))
    print()

def is_safe(board, row, col):
    """Check if placing a queen at board[row][col] is safe"""

    # Check column above
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper left diagonal
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1

    # Check upper right diagonal
    i, j = row - 1, col + 1
    while i >= 0 and j < N:
        if board[i][j] == 1:
            return False
        i -= 1
        j += 1

    return True

def solve(board, row):
    """Recursive function to place queens row by row"""
    if row == N:
        print("One possible solution:")
        print_board(board)
        return True

    for col in range(N):
        if is_safe(board, row, col):
            board[row][col] = 1  # Place the queen
            if solve(board, row + 1):  # Recursively place the rest
                return True
            board[row][col] = 0  # Backtrack

    return False

# ----------- Main Logic Starts Here -----------

# Create 8x8 board filled with 0
board = [[0 for _ in range(N)] for _ in range(N)]

# Ask user to place the first queen
first_row = 0
first_col = int(input("Enter the column number (1-8) to place the first Queen in row 1: ")) - 1

# Validate input
if first_col < 0 or first_col >= N:
    print("Invalid column. Please enter a value between 1 and 8.")
else:
    board[first_row][first_col] = 1  # Place first queen
    # Start solving from the second row
    if not solve(board, first_row + 1):
        print("No solution found.")
