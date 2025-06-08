import matplotlib.pyplot as plt
import mplcursors

import numpy as np

result_array = np.load('results/xyp_gpt-4.1_r3_p64.npy')

x_vals = result_array[:, :, 0].flatten()
y_vals = result_array[:, :, 1].flatten()
p_vals = result_array[:, :, 2].flatten()

plt.figure(figsize=(12, 8))
scatter = plt.scatter(x_vals, y_vals, c=p_vals, cmap='turbo', marker="s")

# Add interactive cursor to display x, y, and probability on hover
# cursor = mplcursors.cursor(scatter, hover=True)
# @cursor.connect("add")
# def on_add(sel):
#     i = sel.index
#     sel.annotation.set_text(
#         f"x={x_vals[i]:.0f}\ny={y_vals[i]:.0f}\nprob={p_vals[i]:.3f}"
#     )
plt.clim(0, 1)
plt.xlim(0, 1024) 
plt.ylim(0, 512)
plt.colorbar(scatter, label="'yes' probability")
plt.title("Probability of 'yes' as a function of flash location")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().invert_yaxis()
plt.axis('equal')
plt.show()

