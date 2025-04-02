from Visualization.Board.board_consts import SQUARE_SIZE
import pygame

class ChessBoard:
    def __init__(self, board, screen, board_size=4, position=(0, 0)):
        self.board = board
        self.board_size = board_size  # Tamanho do tabuleiro (ex: 4x4)
        self.square_size = SQUARE_SIZE  # Tamanho de cada quadrado no tabuleiro
        self.screen = screen  # A superfície existente onde o tabuleiro será desenhado
        self.position = position  # Posição (x, y) onde o tabuleiro será desenhado

    def draw_square(self, x, y, color):
        pygame.draw.rect(self.screen, color, pygame.Rect(x, y, self.square_size, self.square_size))

    def draw_queen(self, x, y):
        queen_radius = self.square_size // 3  # Ajuste o tamanho da rainha
        queen_center = (x + self.square_size // 2, y + self.square_size // 2)
        pygame.draw.circle(self.screen, (255, 0, 0), queen_center, queen_radius)

    def render(self):
        # Desenha o tabuleiro
        for col in range(self.board_size):
            for row in range(self.board_size):
                # Desenha os quadrados do tabuleiro
                if (row + col) % 2 == 0:
                    color = (238, 238, 210)  # Quadrado claro
                else:
                    color = (118, 150, 86)  # Quadrado escuro
                # Desenha o quadrado com base na posição ajustada
                self.draw_square(self.position[0] + col * self.square_size, self.position[1] + row * self.square_size, color)
        
        # Desenha as rainhas
        for col, row in enumerate(self.board["chromosome"]):
            # A posição do vetor indica a linha da rainha na coluna correspondente
            if row is not None:
                self.draw_queen(self.position[0] + col * self.square_size, self.position[1] + row * self.square_size)
