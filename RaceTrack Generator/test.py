import os
a = []

for i in range(0,100):
    file_path = f'maps/Track{i}_map.png'

    if os.path.isfile(file_path):
        a.append(f"Track{i}")
    else:
        print(f"File not found: {i}")
        

# Example usage


print(a)