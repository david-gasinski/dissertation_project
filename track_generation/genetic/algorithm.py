import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from track_gen.abstract import abstract_track_generator    



class GeneticAlgorithm():
    """
        Main class for genetic algorithm implementation.
        
        This architecture allows the use of callback functions for analysing fitness.
    """
    
    def __init__(self, mutation_rate: float, crossover_rate: float, population_size: int = 100, generations: int = 100) -> None:
        self.population_size = population_size
        self.generations = generations
        
        # for performane reasons, two arrays are used
        # python list to store the track object
        # numpy array used for fitness calculations, storing index of tracks
        self.population = []
        self.fitness = np.ndarray(shape=(population_size, 2))
        
        self.seed_generator = np.random.default_rng()

    def initialise_population(self, generator: abstract_track_generator.TrackGenerator) -> None:
        # build population of size pop_size and calculate the fitness of each
        for track in range(self.population_size):
            seed = self.seed_generator.integers(low=0, high=43918403, size=1)

            self.population.append(generator.generate_track(seed))
            self.fitness[track, 0] = track # store index 
            
    def calculate_fitness(self) -> None:
        for track in range(self.population_size):
            
            # calculate the fitness of the population
            fitness = self.population[track].fitness()
            self.fitness[track, 1] = fitness

    def crossover(self) -> None:
        return   
    
    def mutation(self) -> None:
        return         

            
        