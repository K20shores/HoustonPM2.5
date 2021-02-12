from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from enum import Flag, auto
import pickle

# https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class ModelTypes(Flag):
    LinearRegression = auto()
    MLPRegressor = auto()
    Lasso = auto()
    Ridge = auto()
    ElasticNet = auto()
    RandomForestRegressor = auto()
    ExtraTreesRegressor = auto()
    DecisionTreeRegressor = auto()
    AdaBoostRegressor = auto()
    XGBRegressor = auto()

class ModelCollection:
    def __init__(self, model_flags = None):
        self.evaluations = {
            ModelTypes.LinearRegression : self._do_linear_regression,
            ModelTypes.MLPRegressor : self._do_neural_regression,
            ModelTypes.Lasso : self._do_lasso,
            ModelTypes.Ridge : self._do_ridge,
            ModelTypes.ElasticNet : self._do_elasticnet,
            ModelTypes.RandomForestRegressor : self._do_random_forest_regressor,
            ModelTypes.ExtraTreesRegressor : self._do_extra_tree_regressor,
            ModelTypes.DecisionTreeRegressor : self._do_decision_tree_regressor,
            ModelTypes.AdaBoostRegressor : self._do_ada_boostregressor,
            ModelTypes.XGBRegressor : self._do_xgboost
        }
        self.flags = model_flags
        # R^2
        self.r2 = {}
        # root mean squared error
        self.rmse = {}
        # predicitons
        self.pred = {}
        self.models = {}
        self.predicted_variable = None
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.kwargs = {
            ModelTypes.AdaBoostRegressor : {"base_estimator":DecisionTreeRegressor(), "learning_rate":.1},
            ModelTypes.XGBRegressor : {"n_estimators":100},
            ModelTypes.MLPRegressor : {"hidden_layer_sizes":(50, 50,)}
        }
    
    def set_data(self, predicted_variable, x, y, test_size=.2, shuffle=True):
        self.predicted_variable = predicted_variable
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle)
    
    def evaluate(self, silent=True, leading_whitespace = '\t'):
        for model_type in ModelTypes:
            if self.flags is None:
                self.__evaluate_model(model_type)
                if not silent:
                    print(f"{leading_whitespace}{model_type.name} : R^2 {self.r2[model_type]:.2f}")
                    print(f"{leading_whitespace}{model_type.name} : RMSE {self.rmse[model_type]:.2f}")
            elif model_type & self.flags :
                self.__evaluate_model(model_type)
                if not silent:
                    print(f"{leading_whitespace}{model_type.name} : R^2 {self.r2[model_type]:.2f}")
                    print(f"{leading_whitespace}{model_type.name} : RMSE {self.rmse[model_type]:.2f}")
    
    
    def get_result(self, model_type):
        if self.flags is not None and not (model_type & self.flags):
            raise ValueError(f"Model {model_type} disabled by class flags")
        if model_type not in self.pred:
            self.__evaluate_model(model_type)
        return {'r2':self.r2[model_type], 'rmse':self.rmse[model_type]}
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__ = pickle.loads(f.read())
    
    def set_kwarg(self, model_type, **kwargs):
        self.kwargs[model_type] = kwargs
    
    def __evaluate_model(self, model_type):
        self.evaluations[model_type](**self._get_kwargs(model_type))
        y_pred = self.models[model_type].predict(self.x_test)
        self.pred[model_type] = y_pred
        self.r2[model_type] = self.models[model_type].score(self.x_test, self.y_test)
        self.rmse[model_type] = mean_squared_error(self.y_test, y_pred)
    
    def _do_linear_regression(self, **kwargs):
        self.models[ModelTypes.LinearRegression] = LinearRegression(**kwargs).fit(self.x_train, self.y_train)
    
    @ignore_warnings(category=ConvergenceWarning)
    def _do_neural_regression(self, **kwargs):
        self.models[ModelTypes.MLPRegressor] = MLPRegressor(**kwargs).fit(self.x_train, self.y_train)
    
    def _do_lasso(self, **kwargs):
        self.models[ModelTypes.Lasso] = Lasso(**kwargs).fit(self.x_train, self.y_train)
    
    def _do_elasticnet(self, **kwargs):
        self.models[ModelTypes.ElasticNet] = ElasticNet(**kwargs).fit(self.x_train, self.y_train)

    def _do_random_forest_regressor(self, **kwargs):
        self.models[ModelTypes.RandomForestRegressor] = RandomForestRegressor(**kwargs).fit(self.x_train, self.y_train)
    
    def _do_extra_tree_regressor(self, **kwargs):
        self.models[ModelTypes.ExtraTreesRegressor] = ExtraTreesRegressor(**kwargs).fit(self.x_train, self.y_train)
    
    def _do_decision_tree_regressor(self, **kwargs):
        self.models[ModelTypes.DecisionTreeRegressor] = DecisionTreeRegressor(**kwargs).fit(self.x_train, self.y_train)

    def _do_ada_boostregressor(self, **kwargs):
        self.models[ModelTypes.AdaBoostRegressor] = AdaBoostRegressor(**kwargs).fit(self.x_train, self.y_train)
    
    def _do_xgboost(self, **kwargs):
        self.models[ModelTypes.XGBRegressor] = XGBRegressor(**kwargs).fit(self.x_train, self.y_train)
    
    def _do_ridge(self, **kwargs):
        self.models[ModelTypes.Ridge] = Ridge(**kwargs).fit(self.x_train, self.y_train)
    
    def _get_kwargs(self, model_type):
        return self.kwargs[model_type] if model_type in self.kwargs else {}