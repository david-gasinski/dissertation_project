from __future__ import annotations
from typing import TYPE_CHECKING

from track_gen.track import Track

import pygame
import math

if TYPE_CHECKING:
    from .window import Window

class EventLoop():
    
    def __init__(self, window: Window, screen: pygame.Surface , config: dict) -> None:
        self.screen = screen
        self.config = config
        self.window = window
        
        self.running = True
        
        self.render_track = True
        self.render_bezier = False
        
        # create a new track
        x_offset = self.config['width'] / 4
        y_offset = self.config['height'] / 4
        track_bounds = (
            [x_offset, self.config['width'] - x_offset],
            [y_offset, self.config['height'] - y_offset]
        )
        self.track = Track(100, track_bounds[0], track_bounds[1], 2021)
        self.track.generate_track()
        
        # start rendering
        self.render()
        
    def render(self):
        while self.running:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
            # render here        
            self.screen.fill((255,255,255))
            
            if self.render_track:
                self.track.render(self.screen)
                
            if self.render_bezier:
                pass    
                 
            # to here
            pygame.display.flip()
        self.window._exit()