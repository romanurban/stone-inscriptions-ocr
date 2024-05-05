import matplotlib.pyplot as plt

# Data for plotting
categories = ['Pēc Attīrīšanas', 'Pirms Attīrīšanas']
values = [2229, 7032]

# Create a horizontal bar plot
plt.figure(figsize=(10, 4))
plt.barh(categories, values, color=['green', 'blue'])
plt.xlabel('Skaits')
plt.title('Datu kopa pirms un pēc attīrīšanas')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Annotate the percentage reduction
percentage_reduction = 68.3
plt.annotate(f'{percentage_reduction}% samazinājums',
             xy=(values[0], 0), xytext=(values[0]/2, -0.1),
             horizontalalignment='center', verticalalignment='top')

plt.tight_layout()
plt.show()
