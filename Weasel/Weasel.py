import random
import string
import matplotlib.pyplot as plt

# === Constants ===
# Prompt user for a custom phrase
user_input = input("Enter a target phrase (uppercase letters and spaces only): ").upper()

# Sanitize input
allowed_chars = set(string.ascii_uppercase + " ")
if any(c not in allowed_chars for c in user_input):
    raise ValueError("Target must contain only uppercase letters (A-Z) and spaces.")

TARGET = user_input
CHARS = string.ascii_uppercase + " "
MUTATION_RATE = 0.05
POPULATION_SIZE = 100

# === Fitness Function ===
def compute_fitness(candidate):
    return sum(1 for a, b in zip(candidate, TARGET) if a == b)

# === Mutation Function ===
def mutate(parent):
    return "".join(
        random.choice(CHARS) if random.random() < MUTATION_RATE else c
        for c in parent
    )

# === Random String Generator ===
def random_string(length):
    return ''.join(random.choice(CHARS) for _ in range(length))

# === Main Evolution Function (with static plot) ===
def run_weasel_static_plot():
    best = random_string(len(TARGET))
    best_score = compute_fitness(best)
    generation = 0
    generations = []
    scores = []

    with open("weasel_output.txt", "w") as log_file:
        log_file.write("Evolution Log\n")
        log_file.write(f"Target: {TARGET}\n\n")

        while best_score < len(TARGET):
            generation += 1
            offspring = [mutate(best) for _ in range(POPULATION_SIZE)]
            scored_offspring = [(child, compute_fitness(child)) for child in offspring]
            new_best, new_score = max(scored_offspring, key=lambda x: x[1])

            if new_score > best_score:
                best = new_best
                best_score = new_score

            # Logging and printing
            line_txt = f"Gen {generation:3d} | Score: {best_score:2d} | {best}"
            print(line_txt)
            log_file.write(line_txt + "\n")
            generations.append(generation)
            scores.append(best_score)

        log_file.write("\nTarget phrase evolved successfully!\n")
        print("\nTarget phrase evolved successfully!")

    # === Plot the final fitness curve ===
    plt.figure(figsize=(8, 5))
    plt.plot(generations, scores, marker='o', linestyle='-', color='navy')
    plt.title("Fitness Score Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.ylim(0, len(TARGET) + 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitness_plot.png")
    plt.show()

# === Run the program ===
if __name__ == "__main__":
    run_weasel_static_plot()
