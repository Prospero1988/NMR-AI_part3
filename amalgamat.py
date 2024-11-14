import pygame
import sys
import random

# Inicjalizacja Pygame
pygame.init()

# Kolory
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Ustawienia okna
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("DVD Screensaver")

# Załadowanie obrazu DVD
dvd_image = pygame.image.load("dvd_logo.png")  # Można pobrać obrazek 'dvd_logo.png'
dvd_rect = dvd_image.get_rect()

# Skalowanie obrazu, jeśli jest zbyt duży
dvd_rect.width = 100
dvd_rect.height = 50
dvd_image = pygame.transform.scale(dvd_image, (dvd_rect.width, dvd_rect.height))

# Pozycja i prędkość
x, y = random.randint(0, screen_width - dvd_rect.width), random.randint(0, screen_height - dvd_rect.height)
speed_x, speed_y = 3, 3

# Obiekt do kontroli FPS
clock = pygame.time.Clock()

# Główna pętla programu
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Ruch logo
    x += speed_x
    y += speed_y

    # Odbicie od ścian
    if x <= 0 or x + dvd_rect.width >= screen_width:
        speed_x = -speed_x
    if y <= 0 or y + dvd_rect.height >= screen_height:
        speed_y = -speed_y

    # Czyszczenie ekranu i rysowanie nowej klatki
    screen.fill(WHITE)
    screen.blit(dvd_image, (x, y))
    pygame.display.flip()

    # Kontrola FPS - ustawienie na 60 dla płynności
    clock.tick(60)
