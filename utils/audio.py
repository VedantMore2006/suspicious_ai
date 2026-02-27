import pygame
import config


class AudioManager:
    def __init__(self):
        self.enabled = config.ENABLE_BEEP
        self.initialized = False
        self.alarm_playing = False

        if self.enabled:
            try:
                pygame.mixer.init()
                self.sound = pygame.mixer.Sound(config.ALERT_SOUND_PATH)
                self.sound.set_volume(config.AUDIO_VOLUME)
                self.initialized = True
            except Exception as e:
                print(f"[Audio Error] {e}")
                self.enabled = False

    def start_alarm(self):
        if not self.enabled or not self.initialized:
            return

        if not self.alarm_playing:
            self.sound.play(loops=-1)  # infinite loop
            self.alarm_playing = True

    def stop_alarm(self):
        if not self.enabled or not self.initialized:
            return

        if self.alarm_playing:
            self.sound.stop()
            self.alarm_playing = False