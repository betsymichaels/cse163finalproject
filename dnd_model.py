class DnDModel:
    """
    This class defines an object that keeps track of a machine leanring
    model trained to predict dnd character information and atributes
    of its methods and training
    """
    def __init__(self, model, label_type, test_acc=None, train_acc=None):
        """
        Takes a machine learning model and constructs and object that keeps
        track of that model and the data used the train it

        lable_type = the type of lable the model predicts for
        test_acc = the accuracy with wich the model could predict lables
                   from it's testing data
        train_acc = the accuracy with wich the model could predict lables
                   from it's training data

        if test_acc is not specified, it defaults to None
        if train_acc is not specified, it defaults to None
        """
        self._model = model
        self._label_type = label_type
        self._test_acc = test_acc
        self._train_acc = train_acc

    def get_model(self):
        """
        Returns the machine learning model stored in this class
        """
        return self._model

    def get_label_type(self):
        """
        Returns the name of the column this model was trained to predict for
        """
        return self._label_type

    def get_test_acc(self):
        """
        Returns the initial test accuracy of this model
        """
        return self._test_acc

    def get_train_acc(self):
        """
        Returns the initial training accuracy of this model
        """
        return self._train_acc
