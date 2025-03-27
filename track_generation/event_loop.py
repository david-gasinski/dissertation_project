from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from track_gen.generators import convex_hull_generator
from genetic.algorithm import GeneticAlgorithm


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
                    
        self.tracks = []
        self._num_tracks = 100
        self._rng = np.random.default_rng()
        
        track_generator = convex_hull_generator.ConvexHullGenerator()
        
        #for i in range(self._num_tracks):
        #    seed = self._rng.integers(0, 43918403, size=1)
        #    self.tracks.append(track_generator.generate_track(i, self.config["concave_hull"]))
        
        # create new algorithm class
        self.genetic_alg = GeneticAlgorithm(track_generator, self.config["concave_hull"], 2, 0.05, self._num_tracks, 30)
        self.tracks = self.genetic_alg.start_generations()
        
        self.render()
        
        
    def render(self):
        track_index = 0
        
        while self.running:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
            # render here        
            self.screen.fill((0,0,0))
            
            self.tracks[track_index].render(self.screen)
            pygame.image.save(self.screen, Path(f"{self.config['TRACK_IMG_FOLDER']}/{self.tracks[track_index]._fitness}.{track_index}.{self.tracks[track_index].seed}.png"))
            
            track_index += 1
            
            if track_index >= self._num_tracks:
                self.running = False
                                       
            pygame.display.flip()
        self.window._exit()