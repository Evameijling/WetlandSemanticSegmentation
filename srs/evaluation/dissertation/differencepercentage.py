def compute_percentage_differences(list1, list2):
    """
    Computes the percentage difference between corresponding elements of two lists.

    Args:
        list1 (list): First list of numbers.
        list2 (list): Second list of numbers.

    Returns:
        list: A list of percentage differences, formatted with "+" or "-" for increase or decrease.
    """
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    differences = []
    for a, b in zip(list1, list2):
        if a == 0:
            differences.append("N/A")  # Handle division by zero or undefined changes
        else:
            diff_percentage = ((b - a) / a) * 100
            sign = "+" if diff_percentage > 0 else "-" if diff_percentage < 0 else ""
            differences.append(f"{sign}{abs(diff_percentage):.2f}%")
    return differences

######## train ######
# # Example usage 1%
# list1 = [0.8289, 0.6760, 0.5583, 0.6760, 0.6723, 0.7047, 0.4768]
# list2 = [0.8482, 0.7126, 0.5916, 0.7126, 0.7021, 0.7815, 0.4384]

# # Example usage 10%
# list1 = [0.8459, 0.7055, 0.5834, 0.7055, 0.7074, 0.7399, 0.4433]
# list2 = [0.8563, 0.7228, 0.6042, 0.7228, 0.7191, 0.7479, 0.4286]

# # Example usage 30%
# list1 = [0.8505, 0.7320, 0.6122, 0.7320, 0.7262, 0.7715, 0.4250]
# list2 = [0.8588, 0.7484, 0.6319, 0.7484, 0.7414, 0.7864, 0.4098]

# # Example usage 50%
# list1 = [0.8530, 0.7344, 0.6161, 0.7344, 0.7324, 0.7625, 0.4219]
# list2 = [0.8597, 0.7533, 0.6369, 0.7533, 0.7467, 0.7874, 0.4092]

# # Example usage 70%
# list1 = [0.8551, 0.7321, 0.6140, 0.7321, 0.7245, 0.7694, 0.4239]
# list2 = [0.8633, 0.7533, 0.6369, 0.7533, 0.7483, 0.7835, 0.4106]

# # Example usage 100%
# list1 = [0.8522, 0.7287, 0.6096, 0.7287, 0.7226, 0.7687, 0.4295]
# list2 = [0.8565, 0.7383, 0.6198, 0.7383, 0.7322, 0.7748, 0.4233]

######## validation ######
# # Example usage 1%
# list1 = [0.6873, 0.4436, 0.3343, 0.4436, 0.5373, 0.4747, 0.6911]
# list2 = [0.7175, 0.4620, 0.3517, 0.4620, 0.5839, 0.4607, 0.6883]

# # Example usage 10%
# list1 = [0.7613, 0.5148, 0.3975, 0.5148, 0.5905, 0.5322, 0.6298]
# list2 = [0.7667, 0.5141, 0.3986, 0.5141, 0.5923, 0.5229, 0.6297]

# # Example usage 30%
# list1 = [0.7833, 0.5818, 0.4564, 0.5818, 0.6347, 0.5760, 0.5810]
# list2 = [0.7897, 0.5685, 0.4461, 0.5685, 0.6508, 0.5544, 0.5865]

# # Example usage 50%
# list1 = [0.7969, 0.6037, 0.4776, 0.6037, 0.6387, 0.6091, 0.5691]
# list2 = [0.7859, 0.5927, 0.4661, 0.5927, 0.6435, 0.5857, 0.5740]

# # Example usage 70%
# list1 = [0.7849, 0.5954, 0.4678, 0.5954, 0.6382, 0.5947, 0.5735]
# list2 = [0.7888, 0.5961, 0.4692, 0.5961, 0.6443, 0.5933, 0.5700]

# Example usage 100%
list1 = [0.7870, 0.5940, 0.4670, 0.5940, 0.6412, 0.5909, 0.5712]
list2 = [0.7956, 0.5963, 0.4721, 0.5963, 0.6423, 0.5959, 0.5647]

differences = compute_percentage_differences(list1, list2)

# Print results
for i, diff in enumerate(differences):
    print(f"Index {i}: {diff}")

# Format results with a check for non-numeric entries
formatted_results = " & ".join(
    [f"{float(diff.strip('%')):+.2f}\\%" if diff != "N/A" else "N/A" for diff in differences]
) + " \\\\"

# Print the formatted string
print(formatted_results)
