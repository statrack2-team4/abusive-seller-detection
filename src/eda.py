import matplotlib.pyplot as plt
import seaborn as sns

def plot_review_density(df):
    sns.histplot(df["review_density"], bins=30)
    plt.title("Review Density Distribution")
    plt.show()
