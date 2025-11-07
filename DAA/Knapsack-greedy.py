# Title: Solve the Fractional Knapsack Problem Using a Greedy Method

# Function to solve the fractional knapsack problem
def fractional_knapsack(values, weights, capacity):
    n = len(values)
    
    # Create a list to store (value-to-weight ratio, value, weight, item number)
    items = []
    for i in range(n):
        ratio = values[i] / weights[i]
        items.append((ratio, values[i], weights[i], i + 1))

    # Sort items by ratio in descending order (greedy approach)
    items.sort(reverse=True)

    total_profit = 0.0
    total_weight = 0.0
    selected_items = []

    for item in items:
        ratio, value, weight, item_no = item

        if total_weight + weight <= capacity:
            # Take the whole item
            total_weight += weight
            total_profit += value
            selected_items.append((item_no, 1))  # Full item taken
        else:
            # Take the fraction of the item that fits
            remaining = capacity - total_weight
            fraction = remaining / weight

            total_weight += weight * fraction
            total_profit += value * fraction
            selected_items.append((item_no, round(fraction, 2)))  # Partial item taken
            break  # Knapsack is full

    # Print result
    print("\nSelected items (Item Number, Fraction Taken):", selected_items)
    print("Total Weight in Knapsack:", round(total_weight, 2))
    print("Total Profit Earned:", round(total_profit, 2))


# --- Main Program Starts Here ---

# Input number of items
n = int(input("Enter the number of items: "))

# Input values (profits)
values = []
print("Enter the values (profits) of the items:")
for i in range(n):
    val = float(input(f"Value of item {i + 1}: "))
    values.append(val)

# Input weights
weights = []
print("Enter the weights of the items:")
for i in range(n):
    wt = float(input(f"Weight of item {i + 1}: "))
    weights.append(wt)

# Input knapsack capacity
capacity = float(input("Enter the capacity of the knapsack: "))

# Solve the problem
fractional_knapsack(values, weights, capacity)
