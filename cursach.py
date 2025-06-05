import random
import csv
import json
import matplotlib.pyplot as plt
import time

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
        category = random.randint(1, min(10, int(n/2)))
        workers.append(Worker(i + 1, salary, contribution, category))
    return workers

def manual_input_workers():
    workers = []
    print("Введення даних індивідуальної задачі:")
    n = int(input("Кількість працівників: "))
    for i in range(n):
        print(f"\nПрацівник #{i + 1}:")
        salary = int(input("Зарплата: "))
        contribution = float(input("Внесок: "))
        category = int(input("Категорія: "))
        workers.append(Worker(i + 1, salary, contribution, category))
    return workers

def greedy_algorithm(workers, S, h, l):
    for w in workers:
        w.efficiency = w.contribution / w.salary
    workers.sort(key=lambda w: w.efficiency, reverse=True)

    selected = []
    used_categories = set()
    total_salary = 0

    for w in workers:
        if len(selected) >= h:
            break
        if w.category in used_categories:
            continue
        if total_salary + w.salary > S:
            continue

        selected.append(w)
        used_categories.add(w.category)
        total_salary += w.salary

    if len(selected) < l:
        print("Рішення не задовольняє мінімальну кількість учасників")
        return []

    return selected

def ant_colony_algorithm(workers, S, h, l, num_ants=10, num_iterations=100, alpha=0.5, beta=1, rho=0.1):
    n = len(workers)
    pheromones = [1.0] * n

    salaries = [w.salary for w in workers]
    contributions = [w.contribution for w in workers]
    s_min = min(salaries)
    p_max = max(contributions)
    s_max = max(salaries)
    heuristics = [(w.contribution / p_max) / (w.salary / s_max) for w in workers]
    upper_bound = p_max * min(h, S // s_min)

    best_team = []
    best_score = 0

    for _ in range(num_iterations):
        teams = []
        scores = []

        for _ in range(num_ants):
            selected = []
            used_categories = set()
            total_salary = 0

            while True:
                candidates = [i for i in range(n)
                              if workers[i].category not in used_categories and
                                 workers[i] not in selected and
                                 total_salary + workers[i].salary <= S and
                                 len(selected) < h]

                if not candidates:
                    break

                probs = [((pheromones[i] ** alpha) * (heuristics[i] ** beta)) for i in candidates]
                total = sum(probs)
                if total == 0:
                    break
                probs = [p / total for p in probs]

                chosen_index = random.choices(range(len(candidates)), weights=probs)[0]
                chosen = candidates[chosen_index]

                selected.append(workers[chosen])
                used_categories.add(workers[chosen].category)
                total_salary += workers[chosen].salary

            if len(selected) >= l:
                teams.append(selected)
                scores.append(sum(w.contribution for w in selected))

        for i in range(n):
            pheromones[i] *= (1 - rho)

        for team, score in zip(teams, scores):
            delta_tau = score / upper_bound
            for w in team:
                pheromones[workers.index(w)] += delta_tau

        if scores:
            max_score = max(scores)
            if max_score > best_score:
                best_score = max_score
                best_team = teams[scores.index(max_score)]

    return best_team

def experiment_iteration_max():
    m = 10
    h = 7
    l = 3
    n = 15
    iterations = []
    scores = []
    workers = generate_workers(n)
    S = sum(w.salary for w in workers) * random.uniform(0.5, 0.7)
    with open("iteration_max_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration_count", "best_contribution"])
        for i in range(150):
            team = ant_colony_algorithm(workers, S, h, l, m, i + 1)
            score = sum(w.contribution for w in team)
            writer.writerow([i + 1, score])
            iterations.append(i + 1)
            scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, scores, marker='o', linestyle='-', color='blue')
    plt.xlabel("Кількість ітерацій")
    plt.ylabel("Найкращий внесок")
    plt.title("Вплив кількості ітерацій на результат")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def experiment_alpha_beta():
    alpha_values = [0.1, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3]
    beta = 1
    rho = 0.1
    m = 10
    h = 7
    l = 3
    n = 30
    K = 30
    results = []
    workers = generate_workers(n)
    S = sum(w.salary for w in workers) * random.uniform(0.5, 0.7)
    with open("alpha_beta_experiment.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "average_contribution"])
        for alpha in alpha_values:
            scores = []
            for _ in range(K):
                team = ant_colony_algorithm(workers, S, h, l, m, 50, alpha=alpha, beta=beta, rho=rho)
                scores.append(sum(w.contribution for w in team))
            avg_score = sum(scores) / K
            writer.writerow([alpha, avg_score])
            results.append((alpha, avg_score))

    alphas, scores = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, scores, marker='o', linestyle='-', color='blue')
    plt.xlabel("Alpha")
    plt.ylabel("Середній вклад")
    plt.title("Вплив співввідношення параметрів α та β на якість розв'язку")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def experiment_problem_size():
    m = 10
    h = 7
    l = 3
    K = 30
    sizes = []
    greedy_scores_all, greedy_times_all = [], []
    aco_scores_all, aco_times_all = [], []
    with open("size_experiment.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "greedy_avg_contribution", "greedy_avg_time", "aco_avg_contribution", "aco_avg_time"])
        for n in range(10, 101, 10):
            greedy_scores, greedy_times = [], []
            aco_scores, aco_times = [], []
            workers = generate_workers(n)
            S = sum(w.salary for w in workers) * random.uniform(0.5, 0.7)
            for _ in range(K):     
                start = time.time()
                team_greedy = greedy_algorithm(workers[:], S, h, l)
                greedy_times.append(time.time() - start)
                greedy_scores.append(sum(w.contribution for w in team_greedy))

                start = time.time()
                team_aco = ant_colony_algorithm(workers[:], S, h, l, m)
                aco_times.append(time.time() - start)
                aco_scores.append(sum(w.contribution for w in team_aco))

            avg_greedy_score = sum(greedy_scores) / K
            avg_greedy_time = sum(greedy_times) / K
            avg_aco_score = sum(aco_scores) / K
            avg_aco_time = sum(aco_times) / K

            writer.writerow([n, avg_greedy_score, avg_greedy_time, avg_aco_score, avg_aco_time])

            sizes.append(n)
            greedy_scores_all.append(avg_greedy_score)
            greedy_times_all.append(avg_greedy_time)
            aco_scores_all.append(avg_aco_score)
            aco_times_all.append(avg_aco_time)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, greedy_scores_all, label="Жадібний алгоритм", marker='o')
    plt.plot(sizes, aco_scores_all, label="Алгоритм мурашиної колонії", marker='o')
    plt.title("Залежність внеску від розміру задачі")
    plt.xlabel("Кількість працівників")
    plt.ylabel("Середній внесок")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sizes, greedy_times_all, label="Жадібний алгоритм", marker='o')
    plt.plot(sizes, aco_times_all, label="Алгоритм мурашиної колонії", marker='o')
    plt.title("Залежність часу виконання від розміру задачі")
    plt.xlabel("Кількість працівників")
    plt.ylabel("Середній час, с")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()   

def save_workers(workers, filename):
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
        print("7. Ввести працівників вручну")
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
            S = float(input("Бюджет: "))
            h = int(input("Макс. кількість працівників: "))
            l = int(input("Мін. кількість працівників: "))
            greedy_team = greedy_algorithm(workers[:], S, h, l)
            ant_team = ant_colony_algorithm(workers[:], S, h, l)
            print("\nGreedy Team:")
            for w in greedy_team:
                print(vars(w))
            print(f"Сума внесків (Greedy): {sum(w.contribution for w in greedy_team):.2f}")
            print("\nAnt Colony Team:")
            for w in ant_team:
                print(vars(w))
            print(f"Сума внесків (Ant Colony): {sum(w.contribution for w in ant_team):.2f}")
        elif choice == "5":
            filename = input("Ім'я файлу: ")
            save_workers(workers, filename)
        elif choice == "6":
            print("\nОберіть експеримент:")
            print("1. Визначення параметра умови завершення роботи алгоритму мурашиної колонії")
            print("2. Дослідження впливу співвідношення параметрів α та β на ефективність роботи алгоритму")
            print("3. Дослідження впливу кількості працівників на час роботи жадібного алгоритму та алгоритму мурашиної колонії")
            exp_choice = input("Ваш вибір: ")
            if exp_choice == "1":
                experiment_iteration_max()
            elif exp_choice == "2":
                experiment_alpha_beta()
            elif exp_choice == "3":
                experiment_problem_size()
            else:
                print("Невірний вибір експерименту.")
        elif choice == "7":
            workers = manual_input_workers()
        elif choice == "0":
            break
        else:
            print("Невірний вибір. Спробуйте ще раз.")

if __name__ == "__main__":
    main()