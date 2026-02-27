import pygame
import time

audio_file = "alert.mp3"  # change to your file

pygame.mixer.init()
pygame.mixer.music.load(audio_file)
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    time.sleep(0.5)