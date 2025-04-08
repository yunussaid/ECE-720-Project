import matplotlib.pyplot as plt
import csv

# Load heuristic results (replace with the correct path to your heuristic results CSV)
heuristic_trials = []
heuristic_success_count = []

with open('heuristic_results.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        heuristic_trials.append(int(row[0]))
        heuristic_success_count.append(int(row[1]))

# Load the saved results from the CSV file
trials = []
success_count = []

# Load PPO results
with open('PPO_0003_results.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        trials.append(int(row[0]))
        success_count.append(int(row[1]))

# Plot both the heuristic and PPO success rates
plt.plot(trials, success_count, label='PPO_0003 Success Rate', color='blue')
plt.plot(heuristic_trials, heuristic_success_count, label='Heuristic Success Rate', color='green')
plt.xlabel('Trial Number')
plt.ylabel('Cumulative Success Count')
plt.title('Success Rates Over 1000 Trials')
plt.legend()
plt.show()