
class ClassifierWrapper:
    def __init__(self, model, features, class_labels):
        self.model = model
        self.features = features
        self.class_labels = class_labels

    def predict(self, x_observation: list) -> str:
        result = self.model.predict([x_observation])
        return self.class_labels[int(result[0])]

    def prediction_needs(self, verbosity=True):
        if verbosity : return f"You need to provide the values of {self.features} to get a prediction."
        else : return self.features