from collections import defaultdict
import matplotlib.pyplot as plt
from aco_tsp.model import AcoTspModel, TSPGraph


def plot_tsp(graph, path, title):
    """Visualizes the TSP path."""
    positions = graph.pos  # This returns a dictionary {city: (x, y)}
    
    x = [positions[i][0] for i in path] + [positions[path[0]][0]]
    y = [positions[i][1] for i in path] + [positions[path[0]][1]]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o-', label="Path", color="blue")
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    for i, city in enumerate(path):
        plt.text(positions[city][0], positions[city][1], str(city), fontsize=12, color='red')

    plt.show()

def main():
    # tsp_graph = TSPGraph.from_random(num_cities=20, seed=1)
    tsp_graph = TSPGraph.from_tsp_file("aco_tsp/data/kroA100.tsp")
    model_params = {
        "num_agents": tsp_graph.num_cities,
        "tsp_graph": tsp_graph,
    }
    number_of_episodes = 50

    results = defaultdict(list)

    best_path = None
    best_distance = float("inf")

    model = AcoTspModel(**model_params)

    # Plot initial random path based on actual city IDs
    initial_path = tsp_graph.cities  # This should be a list of city IDs
    plot_tsp(tsp_graph, initial_path, title="Initial Path (Random)")

    for e in range(number_of_episodes):
        model.step()
        results["best_distance"].append(model.best_distance)
        results["best_path"].append(model.best_path)
        print(
            f"Episode={e + 1}; Min. distance={model.best_distance:.2f}; pheromone_1_8={model.grid.G[17][15]['pheromone']:.4f}"
        )
        if model.best_distance < best_distance:
            best_distance = model.best_distance
            best_path = model.best_path
            print(f"New best distance:  distance={best_distance:.2f}")

    # Plot final best path after optimization
    if best_path is not None:
        plot_tsp(tsp_graph, best_path, title="Best Path (Optimized)")

    print(f"Best distance: {best_distance:.2f}")
    print(f"Best path: {best_path}")

    # Plot the best distance per episode
    _, ax = plt.subplots()
    ax.plot(results["best_distance"])
    ax.set(xlabel="Episode", ylabel="Best distance", title="Best distance per episode")
    plt.show()


if __name__ == "__main__":
    main()
