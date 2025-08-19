import random
import string
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# --- Simple Weasel parameters (match your class code) ---
LETTERS = string.ascii_uppercase + " "
MUTATION_RATE = 0.05
POP_SIZE = 100
REPORT_EVERY = 5  # print progress every 5 generations


# ----------------- Core helpers (minimal) -----------------

def sanitize(s: str) -> str:
    # Uppercase and keep only A–Z and space
    return "".join(ch for ch in s.upper() if ch in LETTERS)

def random_phrase(n: int) -> str:
    return "".join(random.choice(LETTERS) for _ in range(n))

def fitness(phrase: str, target: str) -> int:
    return sum(p == t for p, t in zip(phrase, target))

def mutate_locked(parent: str, target: str, rate: float) -> str:
    """Mutate only positions that are currently wrong (classic Weasel).
    This guarantees monotonic progress and reliable finish.
    """
    out = []
    for c, t in zip(parent, target):
        if c == t:
            out.append(c)
        else:
            out.append(random.choice(LETTERS) if random.random() < rate else c)
    return "".join(out)


# --------------------------- GUI ---------------------------

class App:
    def __init__(self, root):
        self.root = root
        root.title("Weasel — Minimal GUI")

        frame = ttk.Frame(root, padding=10)
        frame.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        ttk.Label(frame, text="Target phrase:").grid(row=0, column=0, sticky="w")
        self.target_var = tk.StringVar(value="METHINKS IT IS LIKE A WEASEL")
        ttk.Entry(frame, textvariable=self.target_var, width=40).grid(row=0, column=1, sticky="ew")
        frame.columnconfigure(1, weight=1)

        self.start_btn = ttk.Button(frame, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=2, padx=(6, 0))

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(frame, textvariable=self.status_var).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 2))

        self.out = tk.Text(frame, height=16, width=80, wrap="none")
        self.out.grid(row=2, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(2, weight=1)

        yscroll = ttk.Scrollbar(frame, orient="vertical", command=self.out.yview)
        yscroll.grid(row=2, column=3, sticky="ns")
        self.out["yscrollcommand"] = yscroll.set

        self.worker = None

    def log(self, s: str):
        self.out.insert("end", s + "\n")
        self.out.see("end")

    def start(self):
        if self.worker and self.worker.is_alive():
            return

        target = sanitize(self.target_var.get()).strip()
        if not target:
            messagebox.showerror("Input error", "Please enter a phrase using letters and spaces.")
            return

        # Lock UI and clear output
        self.start_btn.configure(state="disabled")
        self.status_var.set("Running…")
        self.out.delete("1.0", "end")

        # Launch worker thread
        self.worker = threading.Thread(target=self.run_weasel, args=(target,), daemon=True)
        self.worker.start()
        self.root.after(100, self.poll)

    def poll(self):
        if self.worker and self.worker.is_alive():
            self.root.after(100, self.poll)
        else:
            self.start_btn.configure(state="normal")
            if "Running" in self.status_var.get():
                self.status_var.set("Done.")

    # ----------------- Minimal evolutionary loop -----------------
    def run_weasel(self, target: str):
        # Initialize
        population = [random_phrase(len(target)) for _ in range(POP_SIZE)]
        generation = 0
        reached = False

        while True:
            generation += 1
            population.sort(key=lambda s: fitness(s, target), reverse=True)
            best = population[0]
            fit = fitness(best, target)

            # Print every 5 generations (and gen 1)
            if generation == 1 or generation % REPORT_EVERY == 0:
                self.root.after(0, self.log, f"Gen {generation:4d}: {best}  (fitness {fit}/{len(target)})")

            # Success check: always print the exact final generation + phrase
            if best == target:
                reached = True
                self.root.after(0, self.log, f"Gen {generation:4d}: {best}  (fitness {fit}/{len(target)})  ✅")
                break

            # Next generation from best parent (classic Weasel locked mutation)
            parent = best
            population = [mutate_locked(parent, target, MUTATION_RATE) for _ in range(POP_SIZE)]

        # Final summary
        if reached:
            self.root.after(0, self.log, f"\nReached target in {generation} generations.")
            self.root.after(0, self.log, f"Final : '{best}'")
            self.root.after(0, self.log, f"Target: '{target}'")
            self.root.after(0, self.status_var.set, f"Success. Generations: {generation}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.minsize(720, 430)
    root.mainloop()
