from Visualization.Board.board import ChessBoard
from Visualization.Board.board_consts import SQUARE_SIZE
import time
import pygame
class FamilyScreen:
    def __init__(self, board, screen):
        self.screen = screen
        self.current_board = board  # Adicionando um board para renderizar
        GAP = 125  # Espaçamento entre os tabuleiros
        BUTTON_WIDTH = 100
        BUTTON_HEIGHT = 40
        BUTTON_COLOR = (69, 133, 136)
        BUTTON_HOVER_COLOR = (131, 165, 152)
        
        # Definindo as posições para os tabuleiros:
        ypos = 400
        self.board1 = ChessBoard(self.current_board["parents"][0], screen, board_size=4, position=(50, ypos))
        board1_width = self.board1.board_size * SQUARE_SIZE  # Largura do tabuleiro 1
        
        # Definir o segundo tabuleiro com o espaçamento adequado
        self.board2_xpos = board1_width + GAP  # Corrige o espaçamento entre os tabuleiros
        self.board2 = ChessBoard(self.current_board["parents"][1], screen, board_size=4, position=(self.board2_xpos, ypos))
        
        # Definir o terceiro tabuleiro com o espaçamento adequado
        self.board3_xpos = self.board2_xpos + self.board2.board_size * SQUARE_SIZE + GAP  # Corrige o espaçamento
        self.board3 = ChessBoard(self.current_board, screen, board_size=4, position=(self.board3_xpos, ypos))

        # Definindo os botões abaixo dos tabuleiros
        self.button1_rect = pygame.Rect(125, ypos + self.board1.board_size * SQUARE_SIZE + 10, BUTTON_WIDTH, BUTTON_HEIGHT)
        self.button2_rect = pygame.Rect(self.board2_xpos + self.board2.board_size * SQUARE_SIZE / 2 - BUTTON_WIDTH / 2, ypos + self.board2.board_size * SQUARE_SIZE + 10, BUTTON_WIDTH, BUTTON_HEIGHT)

        self.button_color = BUTTON_COLOR
        self.button_hover_color = BUTTON_HOVER_COLOR

    def render(self):
        # Renderiza o texto no topo da tela
        self.render_text(f"Geração: {self.current_board['generation']}", (self.screen.get_width() // 2, 30))

        # Renderiza os tabuleiros
        self.board1.render()
        self.board2.render()
        self.board3.render()

        # Renderiza os botões
        self.render_button(self.button1_rect, "Detalhes")
        self.render_button(self.button2_rect, "Detalhes")
        
        # Renderiza os textos acima dos tabuleiros
        self.render_text("PAI 1", (self.board1.position[0] + self.board1.board_size * SQUARE_SIZE / 2, self.board1.position[1] - 30))
        self.render_text("PAI 2", (self.board2.position[0] + self.board2.board_size * SQUARE_SIZE / 2, self.board2.position[1] - 30))
        self.render_text("FILHO", (self.board3.position[0] + self.board3.board_size * SQUARE_SIZE / 2, self.board3.position[1] - 30))

    def render_text(self, text, position):
        font = pygame.font.Font(None, 36)  # Fonte maior para o título
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=position)
        self.screen.blit(text_surface, text_rect)

    def render_button(self, rect, text):
        # Verificando o estado do mouse (hover)
        mouse_pos = pygame.mouse.get_pos()
        if rect.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, self.button_hover_color, rect)
        else:
            pygame.draw.rect(self.screen, self.button_color, rect)

        # Adiciona o texto no botão
        font = pygame.font.Font(None, 30)
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def check_button_click(self):
        # Verifica se algum botão foi pressionado
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()
        if mouse_pressed[0]:  # Se o botão esquerdo do mouse foi pressionado
            if self.button1_rect.collidepoint(mouse_pos):
                self.handle_button1_click()
                # NAO TIRAR O TIMER
                time.sleep(1)
            elif self.button2_rect.collidepoint(mouse_pos):
                self.handle_button2_click()
                # NAO TIRAR O TIMER
                time.sleep(1)

    def handle_button1_click(self):
        parents = self.board1.board["parents"]
        if  parents != None:
            self.current_board = self.board1.board
            self.switch_rendering(parents)

    def handle_button2_click(self):
        parents = self.board2.board["parents"]
        if  parents != None:
            self.current_board = self.board2.board
            self.switch_rendering(parents)
        
    def switch_rendering(self, parents):
            self.board1 = ChessBoard(parents[0], self.screen, board_size=4, position=(50, 400))
            self.board2 = ChessBoard(parents[1], self.screen, board_size=4, position=(self.board2_xpos, 400))
            self.board3 = ChessBoard(self.current_board, self.screen, board_size=4, position=(self.board3_xpos, 400))
            self.render()
