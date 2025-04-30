#Logistic Regression
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):  # Fixed __init__
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, score):
        return 1 / (1 + (2.718281828459045 ** (-score)))

    def fit(self, features, targets):
        sample_count = len(features)
        self.weights = [0] * len(features[0])
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(sample_count):
                linear_output = sum(self.weights[j] * features[i][j] for j in range(len(features[0]))) + self.bias
                prediction = self.sigmoid(linear_output)

                error = prediction - targets[i]
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * error * features[i][j]
                self.bias -= self.learning_rate * error

    def predict_proba(self, features):
        probabilities = []
        for row in features:
            linear_output = sum(self.weights[j] * row[j] for j in range(len(row))) + self.bias
            probabilities.append(self.sigmoid(linear_output))
        return probabilities

    def predict(self, features, threshold=0.5):
        probabilities = self.predict_proba(features)
        return [1 if prob >= threshold else 0 for prob in probabilities]

    def evaluate(self, true_labels, predicted_labels):
        tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p == 1)
        tn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p == 0)
        fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)

        accuracy = (tp + tn) / len(true_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


# Sample dataset: AND logic gate
training_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
training_labels = [0, 0, 0, 1]

test_inputs = [
    [1, 1],
    [0, 0],
    [1, 0]
]
test_labels = [1, 0, 0]

model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(training_inputs, training_labels)
predictions = model.predict(test_inputs)

print("Predictions:", predictions)
print("\nPerformance Metrics:")
model.evaluate(test_labels, predictions)
