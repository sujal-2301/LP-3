# Iterative Fibonacci with Step Count and Complexity Info

print("Phase 1: Iterative Fibonacci Series Generation")
number = int(input("Enter how many Fibonacci numbers are required: "))

# Step count initialization
count = 0

# Base series initialization
series = [0, 1]
count += 1

a = 0
count += 1

b = 1
count += 1

print("="*30)
print("Displaying the Fibonacci Series for:", number)

for i in range(2, number):
    c = a + b
    count += 1

    series.append(c)
    count += 1

    a = b
    count += 1

    b = c
    count += 1

print("Fibonacci Series:", series[:number])
print("Step Count Formula: 4n + 3")
print("Actual Step Count:", count)

# Time and space complexity
print("Time Complexity: O(n)")
print("Space Complexity: O(n)")
