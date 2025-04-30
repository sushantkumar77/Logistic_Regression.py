#LINEAR REGRESSION
class LinearRegression:
    def _init_(self):
        self.intercept = 0
        self.slope = 0

    def calculate_mean(self, values):
        return sum(values) / len(values)

    def calculate_variance(self, values, mean_val):
        return sum((val - mean_val) ** 2 for val in values)

    def calculate_covariance(self, features, targets, mean_feature, mean_target):
        return sum((features[i] - mean_feature) * (targets[i] - mean_target) for i in range(len(features)))

    def fit(self, features, targets):
        mean_feature = self.calculate_mean(features)
        mean_target = self.calculate_mean(targets)
        self.slope = self.calculate_covariance(features, targets, mean_feature, mean_target) / self.calculate_variance(features, mean_feature)
        self.intercept = mean_target - self.slope * mean_feature

    def predict(self, inputs):
        return [self.intercept + self.slope * value for value in inputs]

    def mean_squared_error(self, actual, predicted):
        return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)

    def root_mean_squared_error(self, actual, predicted):
        return self.mean_squared_error(actual, predicted) ** 0.5

    def mean_absolute_error(self, actual, predicted):
        return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)

    def r2_score(self, actual, predicted):
        mean_actual = self.calculate_mean(actual)
        total_sum_squares = sum((a - mean_actual) ** 2 for a in actual)
        residual_sum_squares = sum((a - p) ** 2 for a, p in zip(actual, predicted))
        return 1 - (residual_sum_squares / total_sum_squares)

    def evaluate(self, actual, predicted):
        print("Mean Squared Error (MSE):", self.mean_squared_error(actual, predicted))
        print("Root Mean Squared Error (RMSE):", self.root_mean_squared_error(actual, predicted))
        print("Mean Absolute Error (MAE):", self.mean_absolute_error(actual, predicted))
        print("R² Score:", self.r2_score(actual, predicted))


training_features = [1, 2, 3, 4, 5]
training_targets = [3, 5, 7, 9, 11]

test_features = [6, 7]
test_targets = [13, 15]

model = LinearRegression()
model.fit(training_features, training_targets)
predictions = model.predict(test_features)

print("Model Coefficients:")
print("Intercept:", model.intercept)
print("Slope:", model.slope)
print("Predictions:", predictions)

print("\nPerformance Metrics:")
model.evaluate(test_targets, predictions)


