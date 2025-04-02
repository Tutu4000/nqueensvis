from Visualization.GenericScreen.generic_screen import GenericScreen
from Visualization.FamilyScreen.family_screen import FamilyScreen
from pygame.locals import QUIT
import pygame

screen_width = 1000
screen_height = 1000
music_file = './Visualization/assets/background_music.mp3'  # Caminho para sua música de fundo

# Criando a instância da tela com a imagem e música
generic_screen = GenericScreen(screen_width, screen_height, music_file)

# Exemplo de tabuleiros com a nova lógica (dicionários com 'chromosome' e outros dados)
board00 = {
    "chromosome": [1, 1, 1, 1], 
    "generation": -1,
    "parents": None,
    "crossover_point": None,
    "mutation": None
}
board0 = {
    "chromosome": [0, 0, 0, 0],
    "generation": -1,
    "parents": None,
    "crossover_point": None,
    "mutation": None
}
board1 = {
    "chromosome": [1, 3, 0, 2],  # Tabuleiro de exemplo com rainhas nas posições (0,1), (1,3), (2,0), (3,2)
    "generation": 0,
    "parents": [board0, board00],
    "crossover_point": None,
    "mutation": None
}

board2 = {
    "chromosome": [0, 2, 3, 1],  # Outro tabuleiro de exemplo com rainhas nas posições (0,0), (1,2), (2,3), (3,1)
    "generation": 0,
    "parents": None,
    "crossover_point": None,
    "mutation": None
}
board3 = {
    "chromosome": [3, 1, 0, 2],  # Tabuleiro inferior de exemplo com rainhas nas posições (0,3), (1,1), (2,0), (3,2)
    "generation": 1,
    "parents": [board1, board2],  # Pais do board3 são o board1 e board2
    "crossover_point": None,  # Ponto de crossover (opcional, ajustado conforme necessidade)
    "mutation": None  # Mutação (opcional, ajustado conforme necessidade)
}

# Criando a tela FamilyScreen e passando os tabuleiros
family_screen = FamilyScreen(board3, generic_screen.screen)

running = True
# Dentro do loop principal
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # Checando os cliques nos botões
    family_screen.check_button_click()

    # Renderiza a tela
    generic_screen.render()
    family_screen.render()
    
    pygame.display.update()  # Atualiza a tela

pygame.quit()
