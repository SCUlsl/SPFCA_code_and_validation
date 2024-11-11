import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the newly uploaded CSV file into a DataFrame
file_path_new = './static_results/n(500,3000,50)m(0.1,2,0.1)new_times.csv'
df_new = pd.read_csv(file_path_new)

# Display the first few rows of the DataFrame to understand its structure
df_new.head()

# Drop the unnamed column
df_new.drop(columns=['Unnamed: 0'], inplace=True)

# Generate a heatmap of the DataFrame with specified axis ranges, no titles, and a high-contrast color map
plt.figure(figsize=(10, 6))
sns.heatmap(df_new, cmap='coolwarm', annot=False, linewidths=0.5, vmin=0.68, vmax=0.8, center=0.7)

# Set the x and y axis ticks to the specified ranges and include the last label for both axes
plt.gca().set_xticks([i for i in range(0, len(df_new.columns), len(df_new.columns) // 5)] + [len(df_new.columns) - 1])
plt.gca().set_xticklabels([int(df_new.columns[i].split('segment')[1]) if i < len(df_new.columns) - 1 else 3000 for i in range(0, len(df_new.columns), len(df_new.columns) // 5)] + [3000], color='blue')
plt.gca().set_yticks([i for i in range(0, len(df_new.index), len(df_new.index) // 5)] + [len(df_new.index) - 1])
plt.gca().set_yticklabels([0.0,0.3,0.6,0.9,1.2,1.5,1.8,2.0],color='blue')

# Draw a grid with 5 vertical and 5 horizontal lines, making them more uniform
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Remove the axis labels and the title
plt.xlabel('')
plt.ylabel('')
plt.title('')

plt.show()