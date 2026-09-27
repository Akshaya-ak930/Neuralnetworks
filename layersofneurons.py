inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

outputs = []

for i in range(3):              # loop through neurons
    total = 0

    for j in range(4):          # loop through inputs/weights
        total = total + inputs[j] * weights[i][j]

    total = total + biases[i]

    outputs.append(total)

print(outputs)
