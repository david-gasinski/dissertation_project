import numpy as np
from typing import Callable
from typing import Union

from track_gen.abstract import abstract_track_generator 
from track_gen.abstract import abstract_track
import timeit





class GeneticAlgorithm():
    """
        Main class for genetic algorithm implementation.
        
        This architecture allows the use of callback functions for analysing fitness.
    """
    
    def __init__(self, generator: abstract_track_generator.TrackGenerator, config: dict, tournament_size_k: int, mutation_rate: float, population_size: int = 100, generations: int = 100) -> None:
        self.population_size = population_size
        self.generations = generations
        self.generator = generator
        self.config = config

        self.mutation_rate = mutation_rate
        self.tournament_size_k = tournament_size_k

        # for performane reasons, two arrays are used
        # python list to store the track object
        # numpy array used for fitness calculations, storing index of tracks
        self.population = []
        self.fitness = np.ndarray(shape=(population_size, 3)) # index start_seed fitness
        
        self.seed_generator = np.random.default_rng()

    def initialise_population(self) -> None:
        # build population of size pop_size and calculate the fitness of each
        for track in range(self.population_size):
            seed = self.seed_generator.integers(low=0, high=43918403, size=1)

            self.population.append(self.generator.generate_track(seed, self.config))
                        
            # store index, seed 
            self.fitness[track, 0] = track 
            self.fitness[track, 1] = seed
            
    def calculate_fitness(self) -> None:
        for track in range(self.population_size):
            # calculate the fitness of the population

            fitness = self.population[track].fitness()
            self.fitness[track, 2] = fitness

    def start_generations(self, cb: Callable = None, pass_config: bool = False) -> list[abstract_track.Track]:
        """
            Starts the evolutionary algorithm
            Once complete, an optional callback can be passed to run after completion with the following arguments

                cb(self.population)
            
            If cb is None or returns None, self.population will be returned, else the result of the callback will be returned
            
        """
        generations = 0
        
        while generations < self.generations:
            start = timeit.default_timer()
            if generations == 0: # first generation
                self.initialise_population()
            
            
            # sort the array by fitness
            self.fitness[:, 2].argsort()
                        
            # do tournament selection
            parents = self.tournament_selection(self.tournament_size_k)
            
            # generate the offspring
            offspring = self.generator.crossover(parents, self.config)
            
            # perform mutations based on mutation rate
            mutation_count = int(len(offspring) * self.mutation_rate)
            mutation_selection = np.linspace(0, mutation_count - 1 , mutation_count, dtype=np.int_)
            
            for mutate in mutation_selection:
                individual = offspring[mutate]
                offspring[mutate] = self.generator.mutate(individual, self.config)
                
            # create a new population and slice to only get self.population_size 
            parents.extend(offspring)
            self.population = parents[:self.population_size]
                
            # generate new a new array to keep track of fitness
            self.fitness = np.ndarray(shape=(self.population_size, 3))
            for track in range(self.population_size):
                # store index, seed 
                self.fitness[track, 0] = track 
                self.fitness[track, 1] = self.population[track].seed
            
            # calculate new fitness     
            self.calculate_fitness()
            print(f"Generation {generations}. Average fitness is {np.average(self.fitness[:, 2])}. Time to run {timeit.default_timer() - start}")
            
            generations += 1
        
        # as a final pass, calculate fitness and return population 
        self.calculate_fitness()
        # sort the array by fitness
        self.fitness[:, 2].argsort()
        
        cb_res = None
        if not cb is None:            
            cb_res = cb(self.population, self.config) if pass_config else cb(self.population)
        
        print(self.population)

        return [self.population[int(x[0])] for x in self.fitness] if cb_res is None else cb_res
    
    
    def tournament_selection(self, k: int) -> Union[list[abstract_track.Track], np.ndarray]:
        """
            Tournament selection with a tournament size of k
        """
        
        if not self.population_size % k == 0:
            return np.asarray([])
        
        tournaments = int(self.population_size / k) # cast to int as already checked modulus
        
        # pick 2 individuals
        population = []
        
        # list from 0 to generations
        selection = np.linspace(0, self.population_size - 1 , self.population_size, dtype=np.int_)
        #selection = self.seed_generator.shuffle(
        #    np.linspace(0, 48,2)
        #)

        self.seed_generator.shuffle(selection)

        for index in range(0, tournaments , 1):
            
            # get the fitness of both individua;s
            individuals = [
                self.fitness[selection[index]], 
                self.fitness[selection[self.population_size - index - 1]]
            ]
            
            # compare fitness
            winner = individuals[0] if individuals[0][2] > individuals[1][2] else individuals[1]

            # add the winner to population arrays
            population.append(self.population[int(winner[0])])
        return population
    

            
        