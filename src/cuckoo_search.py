import numpy as np
from src.train import train_and_evaluate  # we will add this
from src.config import RANDOM_SEED

np.random.seed(RANDOM_SEED)

# Search bounds
BOUNDS = {
    "lr": (1e-5, 3e-5),
    "encoder_scale": (0.2, 0.6),
    "dropout": (0.2, 0.4),
    "threshold": (0.5, 0.7),
}

POP_SIZE = 10
PA = 0.25       # discovery probability
ITERATIONS = 10


def levy_flight(beta=1.5):
    sigma = (
        np.math.gamma(1 + beta)
        * np.sin(np.pi * beta / 2)
        / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    return u / abs(v) ** (1 / beta)


def random_solution():
    return {
        k: np.random.uniform(*v)
        for k, v in BOUNDS.items()
    }


def cuckoo_search():
    nests = [random_solution() for _ in range(POP_SIZE)]
    fitness = []
    for i, n in enumerate(nests):
        print(f"[INFO] Evaluating initial nest {i+1}/{POP_SIZE}: {n}")
        score = train_and_evaluate(**n)
        print(f"[INFO] Fitness: {score}")
        fitness.append(score)

    best_idx = np.argmax(fitness)
    best_nest = nests[best_idx]

    for _ in range(ITERATIONS):
        for i in range(POP_SIZE):
            new_nest = nests[i].copy()

            for k in new_nest:
                step = levy_flight()
                new_nest[k] += step * (BOUNDS[k][1] - BOUNDS[k][0])
                new_nest[k] = np.clip(new_nest[k], *BOUNDS[k])

            new_fitness = train_and_evaluate(**new_nest)

            if new_fitness > fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness

        # Abandon worst nests
        abandon = np.random.rand(POP_SIZE) < PA
        for i in range(POP_SIZE):
            if abandon[i]:
                nests[i] = random_solution()
                fitness[i] = train_and_evaluate(**nests[i])

        best_idx = np.argmax(fitness)
        best_nest = nests[best_idx]

        print("Best so far:", best_nest, "Fitness:", fitness[best_idx])

    return best_nest
if __name__ == "__main__":
    print("[INFO] Starting Cuckoo Search optimization...")
    best = cuckoo_search()
    print("\n[SUCCESS] Best hyperparameters found:")
    for k, v in best.items():
        print(f"{k}: {v}")
