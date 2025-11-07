# Recursive Fibonacci Series with Step Count and Complexity Info

print("\nPhase 2: Recursive Fibonacci Series Generation")
num = int(input("Enter how many Fibonacci numbers are required: "))

# Step counter dictionary for mutable tracking
step_counter = {"count": 0}

def recursive_fib(n, counter):
    counter["count"] += 1
    if n <= 1:
        return n
    return recursive_fib(n - 1, counter) + recursive_fib(n - 2, counter)

# Generate full Fibonacci series
series = []
for i in range(num):
    series.append(recursive_fib(i, step_counter))

# Output
print("Fibonacci Series:", series)
print("Step Count (Recursive Calls):", step_counter["count"])
print("Time Complexity: O(2^n)")
print("Space Complexity: O(n)")
