import tkinter as tk
from tkinter import messagebox
import sys

class ReversiGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Reversi")

        size_option = sys.argv[1] # CLASSIC, LARGE, MASSIVE, INFINITE

        if size_option=="CLASSIC":
            self.board_size = 8
            self.tile_size = 100
        elif size_option=="LARGE":
            self.board_size = 12
            self.tile_size = 75
        elif size_option=="MASSIVE":
            self.board_size = 16
            self.tile_size = 55
        elif size_option=="INFINITE":
            self.board_size = 20
            self.tile_size = 45          

        # Game constants
        self.colors = {
            'empty': '#2e8b57',  # sea green
            'black': 'black',
            'white': 'white',
            'highlight': '#98fb98'  # pale green
        }
        
        # Game state
        self.board = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 'black'
        self.valid_moves = []
        
        # Initialize UI
        self.create_board()
        self.setup_initial_pieces()
        self.update_valid_moves()
        self.draw_board()
        
        # Score display
        self.score_label = tk.Label(master, text="Black: 2 - White: 2", font=('Arial', 14))
        self.score_label.pack()
        
        # Current player display
        self.player_label = tk.Label(master, text="Current Player: Black", font=('Arial', 14))
        self.player_label.pack()
    
    def create_board(self):
        """Create the game board canvas"""
        canvas_size = self.board_size * self.tile_size
        self.canvas = tk.Canvas(
            self.master, 
            width=canvas_size, 
            height=canvas_size, 
            bg=self.colors['empty']
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)
        
        # Draw grid lines
        for i in range(self.board_size + 1):
            # Vertical lines
            self.canvas.create_line(
                i * self.tile_size, 0,
                i * self.tile_size, canvas_size,
                fill="black"
            )
            # Horizontal lines
            self.canvas.create_line(
                0, i * self.tile_size,
                canvas_size, i * self.tile_size,
                fill="black"
            )
    
    def setup_initial_pieces(self):
        """Place the initial 4 pieces in the center"""
        mid1 = self.board_size // 2 - 1
        mid2 = self.board_size // 2
        
        # Initial pieces
        self.board[mid1][mid1] = 'white'
        self.board[mid1][mid2] = 'black'
        self.board[mid2][mid1] = 'black'
        self.board[mid2][mid2] = 'white'
    
    def draw_board(self):
        """Draw all pieces and highlights on the board"""
        for row in range(self.board_size):
            for col in range(self.board_size):
                x1 = col * self.tile_size
                y1 = row * self.tile_size
                x2 = x1 + self.tile_size
                y2 = y1 + self.tile_size
                
                # Clear the tile
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=self.colors['empty'],
                    outline="black"
                )
                
                # Draw highlight for valid moves
                if (row, col) in self.valid_moves:
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=self.colors['highlight'],
                        outline="black"
                    )
                
                # Draw the piece if it exists
                if self.board[row][col]:
                    self.canvas.create_oval(
                        x1 + 5, y1 + 5,
                        x2 - 5, y2 - 5,
                        fill=self.colors[self.board[row][col]]
                    )
    
    def update_valid_moves(self):
        """Find all valid moves for the current player"""
        self.valid_moves = []
        opponent = 'white' if self.current_player == 'black' else 'black'
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] is not None:
                    continue  # Skip occupied squares
                
                # Check all 8 directions
                for dr, dc in [(-1,-1), (-1,0), (-1,1),
                              (0,-1),          (0,1),
                              (1,-1),  (1,0),  (1,1)]:
                    r, c = row + dr, col + dc
                    if (0 <= r < self.board_size and 0 <= c < self.board_size and 
                        self.board[r][c] == opponent):
                        # Continue in this direction
                        r += dr
                        c += dc
                        found = False
                        while 0 <= r < self.board_size and 0 <= c < self.board_size:
                            if self.board[r][c] == self.current_player:
                                found = True
                                break
                            elif self.board[r][c] is None:
                                break
                            r += dr
                            c += dc
                        
                        if found:
                            self.valid_moves.append((row, col))
                            break  # No need to check other directions
    
    def handle_click(self, event):
        """Handle player clicks on the board"""
        col = event.x // self.tile_size
        row = event.y // self.tile_size
        
        if (row, col) in self.valid_moves:
            self.make_move(row, col)
    
    def make_move(self, row, col):
        """Place a piece and flip opponent's pieces"""
        opponent = 'white' if self.current_player == 'black' else 'black'
        self.board[row][col] = self.current_player
        
        # Flip opponent's pieces in all valid directions
        for dr, dc in [(-1,-1), (-1,0), (-1,1),
                      (0,-1),          (0,1),
                      (1,-1),  (1,0),  (1,1)]:
            r, c = row + dr, col + dc
            to_flip = []
            
            while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                   self.board[r][c] == opponent):
                to_flip.append((r, c))
                r += dr
                c += dc
            
            if (0 <= r < self.board_size and 0 <= c < self.board_size and 
                self.board[r][c] == self.current_player):
                for (fr, fc) in to_flip:
                    self.board[fr][fc] = self.current_player
        
        # Switch players
        self.current_player = opponent
        self.update_valid_moves()
        self.draw_board()
        self.update_scores()
        
        # Check for game over
        if not self.valid_moves:
            # Current player has no moves, check if game is over
            self.current_player = 'white' if self.current_player == 'black' else 'black'
            self.update_valid_moves()
            
            if not self.valid_moves:
                # Game over
                self.game_over()
            else:
                messagebox.showinfo("No Valid Moves", 
                                  f"{opponent.capitalize()} has no valid moves. {self.current_player.capitalize()} plays again.")
    
    def update_scores(self):
        """Update the score display"""
        black = sum(row.count('black') for row in self.board)
        white = sum(row.count('white') for row in self.board)
        
        self.score_label.config(text=f"Black: {black} - White: {white}")
        self.player_label.config(text=f"Current Player: {self.current_player.capitalize()}")
    
    def game_over(self):
        """Handle game over condition"""
        black = sum(row.count('black') for row in self.board)
        white = sum(row.count('white') for row in self.board)
        
        if black > white:
            winner = "Black wins!"
        elif white > black:
            winner = "White wins!"
        else:
            winner = "It's a tie!"
        
        messagebox.showinfo("Game Over", f"Game over! {winner}\nFinal score: Black {black} - White {white}")
        
        # Ask to play again
        if messagebox.askyesno("Play Again?", "Would you like to play again?"):
            self.reset_game()
        else:
            self.master.quit()
    
    def reset_game(self):
        """Reset the game to initial state"""
        self.board = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 'black'
        self.setup_initial_pieces()
        self.update_valid_moves()
        self.draw_board()
        self.update_scores()

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python reversi.py [MODE]\nMode Options: CLASSIC, LARGE, MASSIVE, INFINITE\nExample: python reversi.py CLASSIC")
        sys.exit()

    root = tk.Tk()
    game = ReversiGame(root)
    root.mainloop()