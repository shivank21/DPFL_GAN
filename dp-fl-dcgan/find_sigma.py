import math

def calculate_sigma(epsilon, delta):
    sigma = (2 * math.log(delta / 1.25)) / epsilon
    return sigma

# Example usage:
epsilon = float(input("Enter the value of epsilon: "))
delta = float(input("Enter the value of delta: "))

sigma = calculate_sigma(epsilon, delta)
print("Sigma (σ) is:", sigma)
