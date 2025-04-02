import pygame
from pygame.locals import QUIT

class GenericScreen:
    def __init__(self, screen_width, screen_height, music_file):
        # Inicializa a tela do Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Generic Screen")
        
        # Carregar e tocar a música de fundo
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.set_volume(0.5)  # Ajusta o volume da música
        pygame.mixer.music.play(-1)  # A música toca em loop infinito

    def render(self):
        self.screen.fill((48, 46, 43))

    def handle_events(self):
        # Gerenciar os eventos de fechamento da janela
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()

    def update(self):
        # Atualizar a tela
        pygame.display.update()

    def run(self):
        # Loop principal da tela
        running = True
        while running:
            self.handle_events()  # Lidar com eventos (ex: sair)
            self.render()  # Desenha a imagem de fundo
            self.update()  # Atualiza a tela
            pygame.time.Clock().tick(60)  # Limita a taxa de quadros (FPS)
