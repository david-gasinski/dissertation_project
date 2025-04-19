import numpy as np
import os
from typing import Callable
from typing import Union

from track_gen.abstract import abstract_track_generator
from track_gen.abstract import abstract_track

from track_gen import utils
import timeit



class GeneticAlgorithm:
    """
    Main class for genetic algorithm implementation.

    This architecture allows the use of callback functions for analysing fitness.

    Crossover types:
        single-point
        uniform
    """

    def __init__(
        self,
        generator: abstract_track_generator.TrackGenerator,
        tournament_size_k: int,
        mutation_rate: float,
        population_size: int = 100,
        generations: int = 100,
        crossover_type: str = "single-point",
        save_generation: int = 0,
        default_dir: str = 'tracks/'
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.generator = generator
        self.save_generation = save_generation
        self.default_dir = default_dir

        self.mutation_rate = mutation_rate
        self.tournament_size_k = tournament_size_k  

        if crossover_type == "single-point":
            self.crossover = self.generator.crossover
        elif crossover_type == "uniform":
            self.crossover = self.generator.uniform_crossover

        self.average_fitness = np.ndarray(shape=(generations))

        self.runtime = np.ndarray(shape=(generations))  # performance tracking

        # for performane reasons, two arrays are used
        # python list to store the track object
        # numpy array used for fitness calculations, storing index of tracks
        self.population = []
        self.fitness = np.ndarray(
            shape=(population_size, 3)
        )  # index start_seed fitness

        self.seed_generator = np.random.default_rng()

    def initialise_population(self) -> None:
        # build population of size pop_size and calculate the fitness of each
        for track in range(self.population_size):
            seed = self.seed_generator.integers(low=0, high=43918403, size=1)

            self.population.append(self.generator.generate_track(seed))

            # store index, seed
            self.fitness[track, 0] = track
            self.fitness[track, 1] = seed

    def calculate_fitness(self) -> None:
        for track in range(self.population_size):
            # calculate the fitness of the population

            fitness = self.generator.fitness(self.population[track])
            self.fitness[track, 2] = fitness

    def start_generations(
        self, cb: Callable = None, pass_config: bool = False
    ) -> list[abstract_track.Track]:
        """
        Starts the evolutionary algorithm
        Once complete, an optional callback can be passed to run after completion with the following arguments

            cb(self.population)

        If cb is None or returns None, self.population will be returned, else the result of the callback will be returned

        """
        generations = 0

        while generations < self.generations:
            start = timeit.default_timer()
            if generations == 0:  # first generation
                self.initialise_population()

            self.fitness[:, 2].argsort()

            # do tournament selection
            parents = self.tournament_selection(self.tournament_size_k)

            # generate the offspring
            offspring = self.crossover(parents)

            # perform mutations based on mutation rate
            mutation_count = int(len(offspring) * self.mutation_rate)
            mutation_selection = np.linspace(
                0, mutation_count - 1, mutation_count, dtype=np.int_
            )

            for mutate in mutation_selection:
                self.generator.mutate(offspring[mutate])

            # create a new population and slice to only get self.population_size
            parents.extend(offspring)
            self.population = parents[: self.population_size]

            # generate new a new array to keep track of fitness
            self.fitness = np.ndarray(shape=(self.population_size, 3))
            for track in range(self.population_size):
                # store index, seed
                self.fitness[track, 0] = track
                self.fitness[track, 1] = self.population[track].seed

            # calculate and sort fitness
            self.calculate_fitness()

            # save fitness
            self.average_fitness[generations] = np.average(self.fitness[:, 2])

            # print(f"Generation {generations}. Average fitness is {self.average_fitness[generations]}. Time to run {timeit.default_timer() - start}")
            self.runtime[generations] = timeit.default_timer() - start
            
            if self.save_generation:
                # get the top tracks
                highest_fitnes = self.fitness[:self.save_generation]
                
                # create the folder structure
                try:
                    os.makedirs(os.path.join(self.default_dir, f'{generations}/'))
                except Exception as e:
                    print("Couldnt make dir, idek why")
                
                for track in highest_fitnes:
                    track_obj = self.population[track[0]]
                    
                    utils.save_track( # save track
                        track_obj,
                        os.path.join(self.default_dir, f"{generations}/{track.fitness()} {track.seed} {track[0]}.json"),
                        os.path.join(self.default_dir, f"{generations}/{track.fitness()} {track.seed} {track[0]}.png")                        
                    )
                
            generations += 1

        # as a final pass, calculate fitness and return population
        self.calculate_fitness()
        # sort the array by fitness
        self.fitness[:, 2].argsort()

        cb_res = None
        if not cb is None:
            cb_res = (
                cb(self.population, self.config) if pass_config else cb(self.population)
            )

        return (
            [self.population[int(x[0])] for x in self.fitness]
            if cb_res is None
            else cb_res
        )

    def tournament_selection(
        self, k: int
    ) -> Union[list[abstract_track.Track], np.ndarray]:
        """
        Tournament selection with a tournament size of k
        """

        if not self.population_size % k == 0:
            return np.asarray([])

        tournaments = int(
            self.population_size / k
        )  # cast to int as already checked modulus

        # pick 2 individuals
        population = []

        # list from 0 to generations
        selection = np.linspace(
            0, self.population_size - 1, self.population_size, dtype=np.int_
        )

        self.seed_generator.shuffle(selection)

        for index in range(0, tournaments, 1):

            # get the fitness of both individua;s
            individuals = [
                self.fitness[selection[index]],
                self.fitness[selection[self.population_size - index - 1]],
            ]

            # compare fitness
            winner = (
                individuals[0]
                if individuals[0][2] > individuals[1][2]
                else individuals[1]
            )

            # add the winner to population arrays
            population.append(self.population[int(winner[0])])
        return population
