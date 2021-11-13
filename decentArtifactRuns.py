import numpy as np

NUM_TRIALS = 1000
P_HEART = 0.1
P_FEATHER = 0.1
P_HOURGLASS = 0.02
P_HAT = 0.1 * 0.2
P_GOBLET = 0.1 * 0.05
P_LIST = [P_HEART, P_FEATHER, P_HOURGLASS, P_HAT, P_GOBLET]
P_OTHER = 1 - np.sum(np.array(P_LIST))
PROBABILITIES = P_LIST + [P_OTHER]
THRESHOLD = 4


def draw_artifact():
    return np.random.multinomial(1, PROBABILITIES)


def threshold_reached(seen):
    num_desired_outcomes = len(seen) - 1
    return sum(seen[:num_desired_outcomes]) == THRESHOLD


def single_run():
    seen = np.zeros(len(PROBABILITIES))
    num_draws = 0
    while not threshold_reached(seen):
        num_draws += 1
        draw = draw_artifact()
        seen = np.logical_or(seen, draw)
    return num_draws


def run_simulation():
    results = []
    for i in range(NUM_TRIALS):
        results.append(single_run())
    results = np.array(results)
    return results


if __name__ == "__main__":
    num_trials = run_simulation()
    print("Mean: ", num_trials.mean())
    print("1% quantile: ", np.quantile(num_trials, 0.01))
    print("25% quantile: ", np.quantile(num_trials, 0.25))
    print("Median: ", np.median(num_trials))
    print("75% quantile: ", np.quantile(num_trials, 0.75))
    print("99% quantile: ", np.quantile(num_trials, 0.99))
