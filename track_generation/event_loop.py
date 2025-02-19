from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from track_gen.track import Track
from track_gen.bezier import Bezier

import pygame
import numpy as np



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
        x_offset = self.config['WIDTH'] / 4
        y_offset = self.config['HEIGHT'] / 4
        track_bounds = (
            [x_offset, self.config['WIDTH'] - x_offset],
            [y_offset, self.config['HEIGHT'] - y_offset]
        )
        
        
        self.tracks = []
        self._num_tracks = 100
        self._rng = np.random.default_rng()
        
        for i in range(0, self._num_tracks):
            self.tracks.append(
                Track(self.config["NUM_POINTS"], track_bounds[0], track_bounds[1], self.config,  self._rng.integers(low=0, high=43918403, size=1))
            )
            self.tracks[i].generate_track()
            
        self.track = Track(2000, track_bounds[0], track_bounds[1], self.config, 80)
        self.track.generate_track()
        
        self.bezier = Bezier(1, 0.01)
        
        weights_x = [300, 350 ,400]
        weights_y = [400, 300, 400]
        
        self.bezier_coords = self.bezier.generate_bezier(self.bezier.QUADRATIC,weights_x, weights_y)
        
        # start rendering
        self.render()
        
    def render(self):
        track_index = 0
        
        while self.running:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
            # render here        
            self.screen.fill((255,255,255))
            
            if self.render_track:
                
                # render track onto screen
                # blit the screen onto the surface
                # save
                
                self.tracks[track_index].render(self.screen)
                pygame.image.save(self.screen, Path(f"{self.config['TRACK_IMG_FOLDER']}/{self.tracks[track_index].seed}.png"))
                
                track_index += 1 
                if track_index >= self._num_tracks:
                    self.running = False            
                                    
            if self.render_bezier:
               self.bezier.render_bezier(self.bezier_coords, self.screen)
             
            # to here
            pygame.display.flip()
        self.window._exit()