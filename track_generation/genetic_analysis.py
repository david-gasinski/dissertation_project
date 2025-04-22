from track_gen.generators import track_generator
from genetic import algorithm
import matplotlib.pyplot as plt

import json
import codecs
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from common.notifications import NotifcationClient

with open("config.json") as f:
    config = json.load(f)['concave_hull']

max_generations = config["ea_config"]["max_generations"]
min_generations = config["ea_config"]["min_generations"]
generation_step_size = config["ea_config"]["generation_step_size"]


def save_track(track, data_path, img_path):
    json.dump(track.serialize(), codecs.open(data_path, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)
    
    # plot the track 
    plt.plot(track.TRACK_COORDS[:, 0], track.TRACK_COORDS[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Fitness: {track.fitness()} [{track.seed}]")
    plt.savefig(img_path)
    plt.clf()


track_gen = track_generator.TrackGenerator(config)
noti = NotifcationClient(os.getenv("PUSHOVER_APP_KEY"), os.getenv("PUSHOVER_USER_KEY"))

generation_path = config["ea_config"]["generation_path"]

for i in range(min_generations, max_generations, generation_step_size):
    
    # create the folder structure
    try:
        os.makedirs(os.path.join(generation_path.format(i), 'tracks/'))
    except Exception as e:
        print("Couldnt make dir, idek why")
    
    genetic = algorithm.GeneticAlgorithm(
        track_gen,
        config["ea_config"]["tournament_size"],
        config["ea_config"]["mutation_rate"],
        config["ea_config"]["population"],
        i,
        crossover_type=config["ea_config"]["crossover"],
        save_generation=50,
        default_dir=generation_path.format(i) 
    )
    
    tracks = genetic.start_generations()
    
    # save and serialize each track
    for idx in range(len(tracks)):
        track = tracks[idx]
        save_track(
            track, 
            os.path.join(generation_path.format(i), f"tracks/{track.fitness()} {track.seed} {idx}.json"),
            os.path.join(generation_path.format(i), f"tracks/{track.fitness()} {track.seed} {idx}.png")
        )
    
    # plot and save the average fitness vs generations
    generations = np.linspace(1, i + 1, i)

    plt.plot(generations, genetic.average_fitness)
    plt.xlabel("Number of Generations")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness vs Generations")
    plt.savefig(os.path.join(generation_path.format(i), "convergence.png"))
    plt.clf()

    print(f"Finished {i} generations in {np.sum(genetic.runtime)}s")
    
    # send notification
    noti.send_notifcation("Finished first generation", f"Finished {generation_path.format(i)}", os.path.join(generation_path.format(i), "convergence.png"))
    
    # plot runtimes
    plt.plot(generations, genetic.runtime)
    plt.xlabel("Number of Generations")
    plt.ylabel("Runtime (s)")
    plt.title("Generations vs Runtime (s)")
    plt.savefig(os.path.join(generation_path.format(i), "run_time.png"))
    plt.clf()
    
    