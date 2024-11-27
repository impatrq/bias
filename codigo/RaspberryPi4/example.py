import numpy as np

# Example JSON dictionary
json_dict = {
    "ch0": [1, 2, 3, 4],
    "ch1": [5, 6, 7, 8],
    "ch2": [9, 10, 11, 12]
}

# Extract the values from the dictionary and stack them as rows
json_dict = np.stack([json_dict[key] for key in sorted(json_dict.keys())])

print("NumPy Array Shape:", json_dict.shape)
print("NumPy Array:")
print(json_dict)
