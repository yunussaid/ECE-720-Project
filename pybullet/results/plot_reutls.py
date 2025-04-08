import matplotlib.pyplot as plt
import csv

# Load the saved results from the CSV file
trials = []
success_count = []

with open('heuristic_results.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        trials.append(int(row[0]))
        success_count.append(int(row[1]))

# Plot the results
plt.plot(trials, success_count)
plt.xlabel('Trial Number')
plt.ylabel('Cumulative Success Count')
plt.title('Heuristic Controller Success Over 1000 Trials')
plt.show()