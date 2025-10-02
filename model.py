import numpy as np
import pandas as pd
import pickle
import os

dataset = pd.read_csv("data.csv")
x_train = dataset["Hours"]
y_train = dataset["Scores"]


def cost_function(x_train, y_train, m, c):
    n = len(x_train)
    cost_sum = 0
    for i in range(n):
        cost = (y_train[i] - (m * x_train[i] + c)) ** 2
        cost_sum += cost

    return (cost_sum * (1 / n)) ** (1 / 2)


def gradient_function(x_train, y_train, m, c):
    n = len(x_train)
    m_gradient = 0
    c_gradient = 0

    for i in range(n):
        f = m * x_train[i] + c

        m_gradient += (f - y_train[i]) * x_train[i]
        c_gradient += f - y_train[i]

    m_gradient /= n
    c_gradient /= n

    return m_gradient, c_gradient


def gradient_descent(x_train, y_train, lr, it):
    m = 0
    c = 0

    for i in range(it):
        m_gradient, c_gradient = gradient_function(x_train, y_train, m, c)

        m -= lr * m_gradient
        c -= lr * c_gradient
        if i % 1000 == 0:
            print(f"Training Progress: Iteration {i} | Cost: {cost_function(x_train, y_train, m, c):.4f}")

    return m, c


lr = 0.01
it = 10000

print("This is a simple implementation of Linear Regression from scratch.")

if os.path.exists("model.pkl"):
    print("Pre-trained model found! Loading existing model...")
    model = None
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        print("Model loaded successfully!\n")
        print("Linear Regression Model - Study Hours vs Test Scores")
        print("Enter study hours to get predicted scores (type 'q' to quit)\n")

    while True:
        x = input("Enter study hours: ")
        if x.lower() == "q":
            print("Thanks for using the Linear Regression Model!")
            break
        try:
            x = float(x)

        except ValueError:
            print("Invalid input!Please enter a numeric value (e.g., 5.5,4 etc) or 'q' to quit.")
            continue
        predicted = model["m"] * x + model["c"]

        print(f"Predicted Score: {predicted:.2f}\n")

else:
    print("No pre-trained model found. Training new model from scratch...")
    print("Starting Linear Regression Training\n")
    final_m, final_c = gradient_descent(x_train, y_train, lr, it)

    model = {"m": final_m, "c": final_c}

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
        print("Model training completed and saved successfully!")

    if os.path.exists("model.pkl"):
        print("Loading the newly trained model...")
        model = None
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
            print("Model loaded successfully!\n")
            print("Linear Regression Model - Study Hours vs Test Scores")
            print("Enter study hours to get predicted scores (type 'q' to quit)\n")

        while True:
            x = input("Enter study hours: ")
            if x.lower() == "q":
                print("Thanks for using the Linear Regression Model!")
                break
            try:
                x = float(x)

            except ValueError:
                print("Invalid input!Please enter a numeric value (e.g., 5.5,4 etc) or 'q' to quit.")
                continue
            predicted = model["m"] * x + model["c"]

            print(f"Predicted Score: {predicted:.2f}\n")

    else:
        print("Failed to save the trained model! Please check file permissions.")
