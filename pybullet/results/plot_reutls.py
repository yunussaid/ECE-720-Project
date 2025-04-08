import sys
import matplotlib.pyplot as plt
import csv


trials = list(range(1, 1001))

def get_success_count(filename):
    success_count = []

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            success_count.append(int(row[1]))
    
    return success_count

plt.xlabel('Episode Number')
plt.ylabel('Cumulative Success Count')
plt.title('Performance Over 1000 Episodes')
plt.legend()
plt.show()

sys.exit()

# Plot #1
plt.plot(trials, get_success_count('heuristic_1_1n.csv'), label='heuristic_1')
plt.plot(trials, get_success_count('heuristic_2_1n.csv'), label='heuristic_2')
plt.plot(trials, get_success_count('heuristic_2_0n.csv'), label='heuristic_2_noiseless')

# Plot #2
plt.plot(trials, get_success_count('heuristic_2_1n.csv'), label='heuristic_2')
plt.plot(trials, get_success_count('PPO_0003_1n.csv'), label='PPO_0003_1n')

# Plot #3
plt.plot(trials, get_success_count('heuristic_2_1n.csv'), label='heuristic_2')
plt.plot(trials, get_success_count('PPO_0003_1n.csv'), label='PPO_0003_1n')
plt.plot(trials, get_success_count('PPO_00003_1n.csv'), label='PPO_00003_1n')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load heuristic results (replace with the correct path to your heuristic results CSV)
heuristic_trials = []
heuristic_success_count = []

with open('heuristic_results.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        heuristic_trials.append(int(row[0]))
        heuristic_success_count.append(int(row[1]))

# Load the saved results from the CSV files
trials = []
success_count_ppo_0003 = []

# Load PPO_0003 results
with open('PPO_0003_results.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        trials.append(int(row[0]))
        success_count_ppo_0003.append(int(row[1]))

# Load PPO_00003 results
success_count_ppo_00003 = []

with open('PPO_00003_results.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        success_count_ppo_00003.append(int(row[1]))  # Assuming the second column has success count

# Plot both the heuristic and PPO success rates
plt.plot(trials, success_count_ppo_00003, label='PPO_00003 Success Rate', color='red')
plt.plot(trials, success_count_ppo_0003, label='PPO_0003 Success Rate', color='blue')
plt.plot(heuristic_trials, heuristic_success_count, label='Heuristic Success Rate', color='green')
plt.xlabel('Trial Number')
plt.ylabel('Cumulative Success Count')
plt.title('Success Rates Over 1000 Trials')
plt.legend()
plt.show()