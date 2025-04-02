from Visualization.FamilyScreen.family_screen import FamilyScreen
from Visualization.Board.board_consts import SQUARE_SIZE
from Visualization.Board.board import ChessBoard
from pygame.locals import QUIT
import pygame

pygame.init()
screen_width = 1000  # Largura da tela
screen_height = 1000  # Altura da tela
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Chess Family Screen")

# Exemplo de tabuleiro com as posições das rainhas no novo formato [linha_coluna]
# Tabuleiro 1: Rainhas nas posições (0,1), (1,3), (2,0), (3,2)
board1 = [1, 3, 0, 2]  # Tabuleiro de exemplo com rainhas nas posições (0,1), (1,3), (2,0), (3,2)
# Tabuleiro 2: Rainhas nas posições (0,0), (1,2), (2,3), (3,1)
board2 = [0, 2, 3, 1]  # Outro tabuleiro de exemplo com rainhas nas posições (0,0), (1,2), (2,3), (3,1)
# Tabuleiro 3: Rainhas nas posições (0,3), (1,1), (2,0), (3,2)
board3 = [3, 1, 0, 2]  # Tabuleiro inferior de exemplo com rainhas nas posições (0,3), (1,1), (2,0), (3,2)

# Criando a tela FamilyScreen e passando os tabuleiros
family_screen = FamilyScreen(board1, board2, board3, screen)

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    screen.fill((255, 255, 255))  # Preenche a tela com a cor branca
    family_screen.render()  # Renderiza todos os tabuleiros
    pygame.display.update()  # Atualiza a tela

pygame.quit()
