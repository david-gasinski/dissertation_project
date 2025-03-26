import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from track_gen.abstract import abstract_track_generator 
    from track_gen.abstract import abstract_track



class GeneticAlgorithm():
    """
        Main class for genetic algorithm implementation.
        
        This architecture allows the use of callback functions for analysing fitness.
    """
    
    def __init__(self, generator: abstract_track_generator.TrackGenerator, mutation_rate: float, crossover_rate: float, population_size: int = 100, generations: int = 100) -> None:
        self.population_size = population_size
        self.generations = generations
        self.generator = generator

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # for performane reasons, two arrays are used
        # python list to store the track object
        # numpy array used for fitness calculations, storing index of tracks
        self.population = []
        self.fitness = np.ndarray(shape=(population_size, 3)) # index start_seed fitness
        
        self.seed_generator = np.random.default_rng()

    def initialise_population(self, generator: abstract_track_generator.TrackGenerator) -> None:
        # build population of size pop_size and calculate the fitness of each
        for track in range(self.population_size):
            seed = self.seed_generator.integers(low=0, high=43918403, size=1)

            self.population.append(generator.generate_track(seed))
            
            # store index, seed 
            self.fitness[track, 0] = track 
            self.fitness[track, 1] = seed
            
    def calculate_fitness(self) -> None:
        for track in range(self.population_size):
            
            # calculate the fitness of the population
            fitness = self.population[track].fitness()
            self.fitness[track, 2] = fitness

    def start_generations(self) -> list[abstract_track.Track]:
        generations = 0
        
        while generations < self.generations:
            if generations == 0: # first generation
                self.initialise_population(self.generator)
            
            self.calculate_fitness()
            
            # sort the array by fitness
            self.fitness[:,:, 2].argsort()
                
            # do tournament selection
                
    
    def tournament_selection(self, k: int) -> np.ndarray:
        if not self.generations % k == 0:
            return np.asarray([])
        
        tournaments = self.generations / k
        
        # pick 2 individuals
        population_np = np.ndarray(size=(tournaments, 3))
        population = []


        # list from 0 to generations
        selection = self.seed_generator.shuffle(
            np.linspace()
        )

        selection =  np.random.sample(low=0, high=self.generations, size=(tournaments , 2))
        
        self.seed_generator.shuffle
        
        for index in range(0, tournaments):
            individuals = [self.fitness[selection[index]], self.fitness[selection[index]]]
        
        # compare the two individuals and return offsping
    
        return
    


    def crossover(self) -> None:
        return   
    
    def mutation(self) -> None:
        return         

            
        