import csv

def calculate_averages(filename):
    # Initialize a dictionary to store sums for each category
    sums = {}
    count = 0

    # Read the CSV file
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        
        # Loop through each row in the CSV file
        for row in reader:
            for key, value in row.items():
                # If the key is not in the sums dictionary, initialize it to 0
                if key not in sums:
                    sums[key] = 0
                sums[key] += float(value)
            
            count += 1

    # Calculate averages
    averages = {key: value / count for key, value in sums.items()}

    return averages

# Usage:
filename = 'liver-eval-patch_unet-Task03'
averages = calculate_averages(filename)
for key, value in averages.items():
    print(f"{key}: {value}")
