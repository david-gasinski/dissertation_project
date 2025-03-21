from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from track_gen.track import Track as convex_hull_track_old
from track_gen.alternative_track import Track as control_point_track
from track_gen.bezier import Bezier

from track_gen.tracks import convex_hull_track
from track_gen.generators import convex_hull_generator

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
        
        self.alternative_track_gen = False
        
        # create a new track
        x_offset = self.config['WIDTH'] / 4
        y_offset = self.config['HEIGHT'] / 4
    
        self.tracks = []
        self._num_tracks = 100
        self._rng = np.random.default_rng()
        
        if not self.alternative_track_gen:
            config = self.config['CONVEX']
            
            track_bounds = (
                [x_offset, x_offset*3],
                [y_offset, y_offset*3]
            )
            
            for i in range(0, self._num_tracks):
                self.tracks.append(
                    convex_hull_track_old(config["NUM_POINTS"], track_bounds[0], track_bounds[1], config,  self._rng.integers(low=0, high=43918403, size=1))
                )
                self.tracks[i].generate_track()
        
        elif self.alternative_track_gen:
            
            config = self.config['CONTROL_POINT']
            
            track_bounds = (
                [-x_offset, x_offset],
                [-y_offset, y_offset]
            )
        
            for i in range(0, self._num_tracks):
                self.tracks.append(
                    control_point_track(config["NUM_POINTS"], track_bounds[0], track_bounds[1], config, self._rng.integers(low=0, high=43918403, size=1))
                )
                self.tracks[i].generate_track();
        
        """REMOVE"""
        #self.track = convex_hull_track(2000, track_bounds[0], track_bounds[1], self.config, 80)
        #self.track.generate_track()
        
        #self.bezier = Bezier(1, 0.01)
        
        #weights_x = [300, 350 ,400]
        #weights_y = [400, 300, 400]
        
        #self.bezier_coords = self.bezier.generate_bezier(self.bezier.QUADRATIC,weights_x, weights_y)
        
        # start rendering
        
        # create a new track generator
        track_generator = convex_hull_generator.ConvexHullGenerator()
        
        # generate many seeds until exception
        
        #for i in range(0, 1000):
        self.test_track = track_generator.generate_track(908372, self.config["concave_hull"])
        
        self.render()
        
    def render(self):
        track_index = 0
        
        while self.running:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
            # render here        
            self.screen.fill((0,0,0))
            
            if self.render_track:
                self.test_track.render(self.screen)
                
                # render track onto screen
                # blit the screen onto the surface
                # save
                
                #self.tracks[track_index].render(self.screen)
                #pygame.image.save(self.screen, Path(f"{self.config['TRACK_IMG_FOLDER']}/{self.tracks[track_index].seed}.png"))
                
                #track_index += 1 
                #if track_index >= self._num_tracks:
                #    self.running = False            
                                    
            #if self.render_bezier:
            #   self.bezier.render_bezier(self.bezier_coords, self.screen)
             
            # to here
            pygame.display.flip()
        self.window._exit()