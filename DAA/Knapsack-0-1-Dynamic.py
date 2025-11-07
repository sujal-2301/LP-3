def knapsack_dp(weights, values, capacity, n):
    # Create a DP table with n+1 rows and capacity+1 columns, initialized to 0
    dp = []
    for _ in range(n + 1):
        dp.append([0] * (capacity + 1))

    # Fill the DP table using bottom-up approach
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            # If current item can be included in the knapsack, 
            # we decide between including it or excluding it
            dp[i][w] = max(
                values[i - 1] + dp[i - 1][w - weights[i - 1]] if weights[i - 1] <= w else 0,
                dp[i - 1][w]
            )
    
    # The value in dp[n][capacity] is the maximum value that can be carried
    return dp[n][capacity]

# Get input from the user for number of items, weights, values, and knapsack capacity
n = int(input("Enter the number of items: "))
weights = [int(input(f"Enter weight of item {i+1}: ")) for i in range(n)]
values = [int(input(f"Enter value of item {i+1}: ")) for i in range(n)]
capacity = int(input("Enter knapsack capacity: "))

# Solve the knapsack problem using dynamic programming
max_value = knapsack_dp(weights, values, capacity, n)

# Output the result with a clear message
print(f"\nMaximum value the knapsack can carry: {max_value}")

#sample input : 
# P = {1, 2, 5, 6}
# W = {2, 3, 4, 5}
# output = 8
