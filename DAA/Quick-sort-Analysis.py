import random
import time

# Utility function to swap elements
def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

# Partition for deterministic pivot (last element)
def partition(arr, low, high, randomized=False):
    # Randomly select pivot if needed
    if randomized:
        pivot_index = random.randint(low, high)
        swap(arr, pivot_index, high)
    
    pivot = arr[high]  # pivot is the last element
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            swap(arr, i, j)
    swap(arr, i + 1, high)  # Put pivot in the correct place
    return i + 1

# QuickSort function (uses partition and handles both deterministic and randomized)
def quicksort(arr, low, high, randomized=False, count=0):
    if low < high:
        pi = partition(arr, low, high, randomized)
        count += (high - low)  # Count comparisons in partition step
        count = quicksort(arr, low, pi - 1, randomized, count)
        count = quicksort(arr, pi + 1, high, randomized, count)
    return count

# Function to run the analysis
def run_analysis(arr):
    print("Initial array:", arr)

    # Analysis for deterministic quicksort
    arr_deterministic = arr.copy()
    start_time = time.time()
    comp_deterministic = quicksort(arr_deterministic, 0, len(arr_deterministic) - 1, randomized=False)
    print(f"Deterministic QuickSort: {time.time() - start_time:.6f}s with {comp_deterministic} comparisons.")
    print("Sorted array (Deterministic):", arr_deterministic)

    # Analysis for randomized quicksort
    arr_randomized = arr.copy()
    start_time = time.time()
    comp_randomized = quicksort(arr_randomized, 0, len(arr_randomized) - 1, randomized=True)
    print(f"Randomized QuickSort: {time.time() - start_time:.6f}s with {comp_randomized} comparisons.")
    print("Sorted array (Randomized):", arr_randomized)

# Main function to input array size and elements
def get_input():
    while True:
        try:
            n = int(input("Enter the number of elements: "))
            arr = list(map(int, input(f"Enter {n} integers separated by space: ").split()))
            
            # Ensure correct number of elements are entered
            if len(arr) != n:
                raise ValueError(f"Expected {n} elements, but got {len(arr)}.")

            # Check if all entries are integers (if there were any errors in input conversion)
            if any(type(x) is not int for x in arr):
                raise ValueError("All elements must be integers.")

            # Return the list if all checks pass
            return arr
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

# Main entry point
if __name__ == "__main__":
    arr = get_input()  # Get valid input from user
    run_analysis(arr)  # Run analysis on the array
