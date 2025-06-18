import matplotlib.pyplot as plt
import numpy as np

# Dummy feature importance values
features = ["SqFt", "Bedrooms", "Bathrooms", "Garage", "Location"]
importance = np.random.rand(len(features))

plt.figure(figsize=(8,5))
plt.barh(features, importance, color="skyblue")
plt.xlabel("Importance Score")
plt.title("Feature Importance")

# Save the plot
plt.savefig("feature_importance_plot.png")
plt.close()
