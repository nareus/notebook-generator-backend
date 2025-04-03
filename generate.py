import matplotlib.pyplot as plt

# Usability scores and their frequencies
scores = [4, 4, 4, 4, 5, 5, 5]  

# Calculate percentage frequencies
total = len(scores)
bins = range(1, 7)  # Bins from 1 to 5 (inclusive) plus 6 as the upper limit

plt.figure(figsize=(8, 6))
plt.hist(scores, bins=bins, weights=[100/total]*total, edgecolor='black', align='left', rwidth=0.8)

plt.xlabel('Effectiveness Score')
plt.ylabel('Percentage')
plt.xticks(range(1, 6))
plt.yticks(range(0, 101, 10))
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()