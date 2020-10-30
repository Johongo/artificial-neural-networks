import numpy as np

# Board parameters
SIZE = 3
BOARD_SIZE = (SIZE, SIZE)

# Learning parameters
ALPHA = 0.1
GAMMA = 1.0
EPSILON = 1.0

# Game parameters
GAMES = 50000


class Game:
    def __init__(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2
        self.reset_board()

    def get_current_player(self):
        while True:
            yield 1
            yield -1

    def reset_board(self):
        self.board = np.zeros(BOARD_SIZE)
        self.game_over = False
        self.hash = None
        self.current_player = self.get_current_player()

    def update_board(self, position):
        self.board[position] = self.current_player.__next__()

    def get_positions(self):
        return np.argwhere(self.board == 0)

    def is_game_over(self):
        sums = max(list(self.board.sum(axis=0)) +
                   list(self.board.sum(axis=1)) +
                   [self.board.trace()] +
                   [np.fliplr(self.board).trace()],
                   key=abs)
        if abs(sums) == SIZE:
            return np.sign(sums)
        if self.get_positions().size == 0:
            return 0

    def get_board_hash(self):
        # board.tobytes() faster
        self.hash = str(self.board.flatten())
        return self.hash

    def print_board(self):
        for row in self.board:
            for element in row:
                if element > 0:
                    print('X', end='')
                elif element < 0:
                    print('O', end='')
                else:
                    print('-', end='')
            print()
        print()

    def play(self):
        for _ in range(GAMES):
            self.reset_board()

            game_status = self.is_game_over()
            while game_status is not None:
                positions = self.get_positions()
                position = self.player_1.choose_move(positions)
                self.player_1.move(self.board, position)
                self.update_board(position)

                game_status = self.is_game_over()
                if game_status is not None:
                    pass

                positions = self.get_positions()
                position = self.player_2.choose_move(positions)
                self.player_2.move(self.board, position)
                self.update_board(position)

                game_status = self.is_game_over()
                if game_status is not None:
                    pass



class Player:
    def __init__(self, marker):
        self.marker = marker
        self.history = {}
        self.policy = {}
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON

    def move(self, this.move):
        # record history

        pass

    def choose_move(self, moves):
        pass

    def add_to_history(self, board):
        self.history.append(board)

def main():
    player_1 = Player(1)
    player_2 = Player(-1)

    game = Game(player_1, player_2)
    game.update_board((1, 1))
    game.print_board()
    game.update_board((0, 2))
    game.print_board()
    game.update_board((2, 2))
    game.print_board()
    game.update_board(((2, 0)))
    game.print_board()
    game.update_board((0, 0))
    game.print_board()


if __name__ == "__main__":
    main()
