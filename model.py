# Import necessary libraries
import pandas as pd
import pickle
import os

# Load the dataset from CSV file
dataset = pd.read_csv("data.csv")
x_train = dataset["Hours"]  # Independent variable (study hours)
y_train = dataset["Scores"]  # Dependent variable (test scores)


def cost_function(x_train, y_train, m, c):
    # Calculate the Root Mean Square Error (RMSE) cost function
    n = len(x_train)  # Number of data points
    cost_sum = 0
    # Calculate squared differences for each data point
    for i in range(n):
        a = m * x_train[i] + c  # Predicted value using linear equation
        cost = (y_train[i] - a) ** 2  # Squared difference between actual and predicted
        cost_sum += cost

    return (cost_sum * (1 / n)) ** (1 / 2)  # Return RMSE


def gradient_function(x_train, y_train, m, c):
    # Calculate gradients for slope (m) and intercept (c)
    n = len(x_train)
    m_gradient = 0  # Gradient for slope
    c_gradient = 0  # Gradient for intercept

    # Calculate gradients using partial derivatives
    for i in range(n):
        f = m * x_train[i] + c  # Current prediction

        m_gradient += (f - y_train[i]) * x_train[
            i
        ]  # Partial derivative with respect to m
        c_gradient += f - y_train[i]  # Partial derivative with respect to c

    m_gradient /= n  # Average the gradients
    c_gradient /= n

    return m_gradient, c_gradient


def gradient_descent(x_train, y_train, lr, it):
    # Train the linear regression model using gradient descent algorithm
    m = 0  # Initialize slope (weight)
    c = 0  # Initialize intercept (bias)

    # Training loop
    for i in range(it):
        m_gradient, c_gradient = gradient_function(x_train, y_train, m, c)

        m -= lr * m_gradient  # Update slope
        c -= lr * c_gradient  # Update intercept
        # Print progress every 1000 iterations
        if i % 1000 == 0:
            print(
                f"Training Progress: Iteration {i} | Cost: {cost_function(x_train, y_train, m, c):.4f}"
            )

    return m, c


# Hyperparameters
lr = 0.01  # Learning rate (step size for parameter updates)
it = 10000  # Number of iterations for training

print("This is a simple implementation of Linear Regression from scratch.")

# Check if a pre-trained model exists
if os.path.exists("model.pkl"):
    print("Pre-trained model found! Loading existing model...")
    model = None
    # Load the saved model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        print("Model loaded successfully!\n")
        print("Linear Regression Model - Study Hours vs Test Scores")
        print("Enter study hours to get predicted scores (type 'q' to quit)\n")

    # Interactive prediction loop
    while True:
        x = input("Enter study hours: ")
        if x.lower() == "q":
            print("Thanks for using the Linear Regression Model!")
            break
        # Validate user input
        try:
            x = float(x)

        except ValueError:
            print(
                "Invalid input!Please enter a numeric value (e.g., 5.5,4 etc) or 'q' to quit."
            )
            continue
        predicted = model["m"] * x + model["c"]  # Make prediction using linear equation

        print(f"Predicted Score: {predicted:.2f}\n")

else:
    # No pre-trained model found, train from scratch
    print("No pre-trained model found. Training new model from scratch...")
    print("Starting Linear Regression Training\n")
    final_m, final_c = gradient_descent(x_train, y_train, lr, it)

    model = {"m": final_m, "c": final_c}  # Save the trained parameters

    # Serialize and save the model to disk
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
        print("Model training completed and saved successfully!")

    # Load the newly trained model for immediate use
    if os.path.exists("model.pkl"):
        print("Loading the newly trained model...")
        model = None
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
            print("Model loaded successfully!\n")
            print("Linear Regression Model - Study Hours vs Test Scores")
            print("Enter study hours to get predicted scores (type 'q' to quit)\n")

        # Interactive prediction loop for newly trained model
        while True:
            x = input("Enter study hours: ")
            if x.lower() == "q":
                print("Thanks for using the Linear Regression Model!")
                break
            # Validate user input
            try:
                x = float(x)

            except ValueError:
                print(
                    "Invalid input!Please enter a numeric value (e.g., 5.5,4 etc) or 'q' to quit."
                )
                continue
            predicted = (
                model["m"] * x + model["c"]
            )  # Make prediction using linear equation

            print(f"Predicted Score: {predicted:.2f}\n")

    else:
        print("Failed to save the trained model! Please check file permissions.")
