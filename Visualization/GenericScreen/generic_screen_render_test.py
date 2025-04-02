from Visualization.GenericScreen.generic_screen import GenericScreen
import pygame
screen_width = 800
screen_height = 800
background_image = './Visualization/assets/background_image.png'    # Caminho para sua imagem de fundo
music_file = './Visualization/assets/background_music.mp3'          # Caminho para sua música de fundo

# Criando a instância da tela com a imagem e música
screen = GenericScreen(screen_width, screen_height, background_image, music_file)

# Inicia o loop da tela
screen.run()
