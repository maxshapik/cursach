import random
import csv
import json
import matplotlib.pyplot as plt

class Worker:
    def __init__(self, id, salary, contribution, category):
        self.id = id
        self.salary = salary
        self.contribution = contribution
        self.category = category

    def to_dict(self):
        return {
            "id": self.id,
            "salary": self.salary,
            "contribution": self.contribution,
            "category": self.category,
        }

    @staticmethod
    def from_dict(data):
        return Worker(
            data["id"],
            data["salary"],
            data["contribution"],
            data["category"]
        )

def generate_workers(n):
    workers = []
    for i in range(n):
        salary = random.randint(1000, 5000)
        contribution = random.uniform(0.5, 1.0)
        category = random.randint(1, 3)
        workers.append(Worker(i + 1, salary, contribution, category))
    return workers

def greedy_algorithm(workers, budget, k):
    workers.sort(key=lambda w: w.contribution / w.salary, reverse=True)
    team = []
    total_salary = 0
    for worker in workers:
        if len(team) < k and total_salary + worker.salary <= budget:
            team.append(worker)
            total_salary += worker.salary
    return team

def ant_colony_algorithm(workers, budget, k, iterations=50, alpha=1, beta=2, rho=0.1, Q=100):
    n = len(workers)
    pheromones = [1.0] * n
    best_team = []
    best_score = 0
    for _ in range(iterations):
        for _ in range(10):
            team = []
            total_salary = 0
            selected = [False] * n
            contributions = [w.contribution for w in workers]
            salaries = [w.salary for w in workers]
            p_max = max(contributions) if max(contributions) != 0 else 1
            s_max = max(salaries) if max(salaries) != 0 else 1
            heuristics = [(w.contribution / p_max) / (w.salary / s_max) for w in workers]

            for _ in range(k):
                probabilities = []
                for i in range(n):
                    if not selected[i] and total_salary + workers[i].salary <= budget:
                        prob = (pheromones[i] ** alpha) * (heuristics[i] ** beta)
                        probabilities.append(prob)
                    else:
                        probabilities.append(0.0)
                total = sum(probabilities)
                if total == 0:
                    break
                probabilities = [p / total for p in probabilities]
                chosen = random.choices(range(n), weights=probabilities)[0]
                team.append(workers[chosen])
                total_salary += workers[chosen].salary
                selected[chosen] = True

            score = sum(w.contribution for w in team)
            if score > best_score:
                best_score = score
                best_team = team

        for i in range(n):
            pheromones[i] *= (1 - rho)
        for w in best_team:
            pheromones[workers.index(w)] += Q / (1 + best_score)

    return best_team

def run_batch_experiments(n_min, n_max, n_step, k, budget_factor, algorithms):
    results = []
    for n in range(n_min, n_max + 1, n_step):
        workers = generate_workers(n)
        budget = sum(w.salary for w in workers) * budget_factor
        for alg_name, alg_func in algorithms.items():
            team = alg_func(workers[:], budget, k)
            total_contribution = sum(w.contribution for w in team)
            results.append({"n": n, "algorithm": alg_name, "contribution": total_contribution})
    return results

def plot_results(results):
    plt.clf()
    algorithms = set(r["algorithm"] for r in results)
    for alg in algorithms:
        x = [r["n"] for r in results if r["algorithm"] == alg]
        y = [r["contribution"] for r in results if r["algorithm"] == alg]
        plt.plot(x, y, label=alg)
    plt.xlabel("Number of Workers")
    plt.ylabel("Total Contribution")
    plt.title("Comparison of Algorithms")
    plt.legend()
    plt.grid(True)
    plt.show()

def save_results_to_csv(results, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "algorithm", "contribution"])
        writer.writeheader()
        writer.writerows(results)

def save_results(workers, filename):
    with open(filename, "w") as f:
        json.dump([w.to_dict() for w in workers], f, indent=2)

def load_workers(filename):
    try:
        with open(filename, "r") as f:
            return [Worker.from_dict(d) for d in json.load(f)]
    except FileNotFoundError:
        print("Файл не знайдено.")
        return []

def edit_workers(workers):
    while True:
        print("\nПоточні працівники:")
        for w in workers:
            print(f"{w.id}. Зарплата: {w.salary}, Внесок: {w.contribution:.2f}, Категорія: {w.category}")
        choice = input("Редагувати працівника (номер) або натисніть Enter для завершення: ")
        if not choice:
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(workers):
                salary = int(input("Нова зарплата: "))
                contribution = float(input("Новий внесок: "))
                category = int(input("Нова категорія: "))
                workers[idx].salary = salary
                workers[idx].contribution = contribution
                workers[idx].category = category
            else:
                print("Недійсний номер працівника.")
        except ValueError:
            print("Невірний ввід. Спробуйте ще раз.")

def get_experiment_config():
    return {
        "n_min": 10,
        "n_max": 100,
        "n_step": 10,
        "k": 5,
        "budget_factor": 0.4,
        "algorithms": {
            "Greedy": greedy_algorithm,
            "Ant Colony": ant_colony_algorithm,
        },
    }

def run_configured_experiments():
    config = get_experiment_config()
    results = run_batch_experiments(
        config["n_min"], config["n_max"], config["n_step"],
        config["k"], config["budget_factor"], config["algorithms"]
    )
    plot_results(results)
    save_results_to_csv(results, "experiment_results.csv")

def main():
    workers = []
    while True:
        print("\nМеню:")
        print("1. Згенерувати працівників")
        print("2. Завантажити працівників з файлу")
        print("3. Редагувати працівників")
        print("4. Запустити алгоритми")
        print("5. Зберегти працівників у файл")
        print("6. Запустити експерименти")
        print("0. Вийти")
        choice = input("Ваш вибір: ")

        if choice == "1":
            n = int(input("Кількість працівників: "))
            workers = generate_workers(n)
        elif choice == "2":
            filename = input("Ім'я файлу: ")
            workers = load_workers(filename)
        elif choice == "3":
            edit_workers(workers)
        elif choice == "4":
            budget = float(input("Бюджет: "))
            k = int(input("Кількість працівників у команді: "))
            greedy_team = greedy_algorithm(workers[:], budget, k)
            ant_team = ant_colony_algorithm(workers[:], budget, k)
            print("\nGreedy Team:")
            for w in greedy_team:
                print(vars(w))
            print("\nAnt Colony Team:")
            for w in ant_team:
                print(vars(w))
        elif choice == "5":
            filename = input("Ім'я файлу: ")
            save_results(workers, filename)
        elif choice == "6":
            run_configured_experiments()
        elif choice == "0":
            break
        else:
            print("Невірний вибір. Спробуйте ще раз.")

if __name__ == "__main__":
    main()
