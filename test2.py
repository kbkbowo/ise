import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['A', 'B', 'C', 'D']
values = [5, 7, 8, 4]
errors = [0.8, 0.6, 1.2, 0.9]  # Error values

# Bar plot with error bars
plt.bar(categories, values, yerr=errors, capsize=5, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot with Error Bars')

# Show plot
plt.savefig('test2.png')
