# import matplotlib.pyplot as plt
# import numpy as np

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Data
    percentages = [15, 25, 50, 75, 100]


    ### Raw
    # Dice
    benchmark = [0.656, 0.761, 0.878, 0.869, 0.860]
    seqseg = [0.745, 0.808, 0.860, 0.859, 0.889]
    # Centerline
    # benchmark = [0.708, 0.849, 0.879, 0.888, 0.829]
    # seqseg = [0.936, 0.908, 0.951, 0.914, 0.933]


    ### Largest

    # Dice
    # benchmark = [0.513, 0.860, 0.844, 0.876, 0.860]
    # seqseg = [0.745, 0.808, 0.860, 0.858, 0.889]
    # Centerline
    # benchmark = [0.433, 0.849, 0.846, 0.888, 0.825]
    # seqseg = [0.936, 0.908, 0.951, 0.914, 0.963]

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(percentages, benchmark, color='blue', label='Benchmark', marker='o')
    plt.scatter(percentages, seqseg, color='green', label='Seqseg', marker='s')

    # Labeling
    plt.xlabel('% of dataset')
    plt.ylabel('Dice Score')
    plt.title('Dice Score vs % of Dataset')
    plt.xticks(percentages)
    # y starts at 0
    plt.ylim(0, 1)
    plt.xlim(0, 100)
    # Add lines connecting the points
    plt.plot(percentages, benchmark, color='blue', linestyle='--', alpha=0.5)
    plt.plot(percentages, seqseg, color='green', linestyle='--', alpha=0.5)
    # Make texts larger
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.show()
