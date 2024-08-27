from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class Commodity:
    def __init__(self, csv_name):
        self.name = csv_name
        dataset = pd.read_csv(csv_name)
        self.X = dataset.iloc[:, :-1].values
        self.Y = dataset.iloc[:, 3].values
        
        depth = random.randrange(7, 18)
        self.regressor = DecisionTreeRegressor(max_depth=depth)
        self.regressor.fit(self.X, self.Y)
        
        # Predictions
        self.y_pred = self.regressor.predict(self.X)
        
        # Calculate Accuracy Metrics
        self.r2_score = r2_score(self.Y, self.y_pred)
        self.mae = mean_absolute_error(self.Y, self.y_pred)
        self.mse = mean_squared_error(self.Y, self.y_pred)
        self.rmse = np.sqrt(self.mse)
        
        print(f"R² Score for {self.name}: {self.r2_score}")
        print(f"Mean Absolute Error (MAE) for {self.name}: {self.mae}")
        print(f"Mean Squared Error (MSE) for {self.name}: {self.mse}")
        print(f"Root Mean Squared Error (RMSE) for {self.name}: {self.rmse}")

    def getPredictedValue(self, value):
        if value[1] >= 2019:
            fsa = np.array(value).reshape(1, 3)
            return self.regressor.predict(fsa)[0]
        else:
            c = self.X[:, 0:2]
            x = []
            for i in c:
                x.append(i.tolist())
            fsa = [value[0], value[1]]
            ind = 0
            for i in range(0, len(x)):
                if x[i] == fsa:
                    ind = i
                    break
            return self.Y[i]
