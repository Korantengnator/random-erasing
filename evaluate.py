import json
import matplotlib.pyplot as plt
import os


def load_results(experiment_name):
    """Load the saved training history for a given experiment."""
    path = f'results/{experiment_name}_history.json'
    with open(path, 'r') as f:
        return json.load(f)


def plot_training_curves(baseline, re_results):
    """
    Creates two side-by-side plots:
    - Left:  Training accuracy over epochs (both runs)
    - Right: Test error over epochs (both runs)
    These are the figures you'll put in your report.
    """

    epochs = [entry['epoch'] for entry in baseline]

    # Pull out the values we want to plot
    baseline_train_acc = [entry['train_acc'] for entry in baseline]
    re_train_acc       = [entry['train_acc'] for entry in re_results]

    baseline_test_error = [entry['test_error'] for entry in baseline]
    re_test_error       = [entry['test_error'] for entry in re_results]

    # Create a figure with 2 side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Training Accuracy ---
    ax1.plot(epochs, baseline_train_acc, label='Baseline', color='steelblue', linewidth=2)
    ax1.plot(epochs, re_train_acc, label='Random Erasing', color='darkorange', linewidth=2)
    ax1.set_title('Training Accuracy over Epochs', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Test Error ---
    ax2.plot(epochs, baseline_test_error, label='Baseline', color='steelblue', linewidth=2)
    ax2.plot(epochs, re_test_error, label='Random Erasing', color='darkorange', linewidth=2)
    ax2.set_title('Test Error over Epochs', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Error (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
    print("Plot saved to results/training_curves.png")
    plt.show()


def print_summary_table(baseline, re_results):
    """Prints a clean comparison table — useful for your report."""

    b_final = baseline[-1]
    r_final = re_results[-1]
    improvement = b_final['test_error'] - r_final['test_error']

    print("\n" + "="*55)
    print(f"{'RESULTS SUMMARY':^55}")
    print("="*55)
    print(f"{'Metric':<30} {'Baseline':>10} {'With RE':>10}")
    print("-"*55)
    print(f"{'Final Train Accuracy':<30} {b_final['train_acc']:>9.2f}% {r_final['train_acc']:>9.2f}%")
    print(f"{'Final Test Accuracy':<30} {b_final['test_acc']:>9.2f}% {r_final['test_acc']:>9.2f}%")
    print(f"{'Final Test Error':<30} {b_final['test_error']:>9.2f}% {r_final['test_error']:>9.2f}%")
    print("-"*55)
    print(f"{'Improvement (Error Reduction)':<30} {improvement:>9.2f}%")
    print("="*55)


if __name__ == "__main__":
    print("Loading results...")
    baseline   = load_results("baseline")
    re_results = load_results("with_RE")

    print_summary_table(baseline, re_results)
    plot_training_curves(baseline, re_results)