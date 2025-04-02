import pygame
from Visualization.Board.board_consts import SQUARE_SIZE
from Visualization.Board.board import ChessBoard

# Defina o tamanho da tela
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400

# Inicializar o Pygame
pygame.init()

# Defina a superfície de tela
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Teste do Tabuleiro de Xadrez")

# Definir o estado do tabuleiro (1D)
# Cada índice do vetor representa a coluna, e o valor é a linha onde está a rainha
# Exemplo: [1, 3, 0, 2] significa:
# - Rainha na coluna 0, linha 1
# - Rainha na coluna 1, linha 3
# - Rainha na coluna 2, linha 0
# - Rainha na coluna 3, linha 2
board_state = [1, 3, 0, 2]  # Tabuleiro de exemplo com rainhas nas posições (0,1), (1,3), (2,0), (3,2)

# Instanciar o tabuleiro de xadrez
chess_board = ChessBoard(board_state, screen, board_size=4)

# Loop de execução do Pygame
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Limpar a tela (preencher com branco)
    screen.fill((255, 255, 255))

    # Renderizar o tabuleiro
    chess_board.render()

    # Atualizar a tela
    pygame.display.flip()

# Finalizar o Pygame
pygame.quit()
