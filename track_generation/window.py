from event_loop import EventLoop

import pygame
import json
import sys


class Window():
    CONFIG_PATH = "config.json"
    
    def __init__(self):
        self.config = None
        self._load_config(self.CONFIG_PATH)
        self.screen = pygame.display.set_mode([self.config['WIDTH'], self.config['HEIGHT']])
        
        # create event loop
        self.event_loop = EventLoop(self, self.screen, self.config)        
    
    def _load_config(self, config_path: str) -> None:
        try:
            with open(config_path) as f:
                self.config = json.load(f)
        except FileNotFoundError as e:
            print(e)
            print("[ERROR] Failed to find config.")
            self._exit()
        except json.JSONDecodeError as e:
            print(e)
            print("[ERROR] Could not decode config. Is it in the correct format (JSON)?")
            self._exit()
        
    def _exit(self):
        pygame.quit()
        sys.exit(0)
        
if __name__ == '__main__':
    pygame.init()
    window = Window()
        