from catboost import CatBoostRegressor
import pandas as pd
from numpy import ndarray


class Transformer:
    keywords_dict = {}
    genres = None
    directors = None
    film_locations = None
    bound = 0

    def fit(self, X, bound):
        y = X['awards']
        X = X.drop(columns='awards')
        keywords_lst = [word for words in list(X['keywords']) for word in words]
        dict_words = dict(zip(keywords_lst, [keywords_lst.count(word) for word in keywords_lst]))

        self.keywords_dict = dict(sorted(dict_words.items(), key=lambda item: item[1])[::-1])
        self.genres = list(X['genres'])
        self.directors = list(X['directors'])
        self.film_locations = list(X['filming_locations'])
        self.bound = bound

        return X, y

    def transform(self, X):
        X[[genre for genres in self.genres for genre in genres]] = 0
        X[[director for directors in self.directors for director in directors]] = 0
        X[[location for locations in self.film_locations for location in locations]] = 0

        for key, value in self.keywords_dict.items():
            if value < self.bound:
                break
            else:
                X[key] = 0

        for value in set(word for words in list(X['keywords']) for word in words):
            if value in self.keywords_dict.keys() and self.keywords_dict[value] >= self.bound:
                X.loc[[value in row['keywords'] for i, row in X.iterrows()], value] = 1

        for feature in ['genres', 'directors', 'filming_locations']:
            X[[word for words in list(X[feature]) for word in words]] = 0
            for value in set(word for words in list(X[feature]) for word in words):
                X.loc[[value in row[feature] for i, row in X.iterrows()], value] = 1

            X = X.drop(columns=feature)

        X['actor_0_gender'] = X['actor_0_gender'].astype('category')
        X['actor_1_gender'] = X['actor_1_gender'].astype('category')
        X['actor_2_gender'] = X['actor_2_gender'].astype('category')

        X = X.drop(columns='keywords')

        return X

    def fit_transform(self, X, bound):
        X, y = self.fit(X, bound)

        return self.transform(X), y


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    transformer = Transformer()
    X_train, y_train = transformer.fit_transform(df_train, 58)
    X_test = transformer.transform(df_test)
    categorical_features = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender']

    regressor = CatBoostRegressor(**{'learning_rate': 0.04626808450407065, 'max_depth': 9, 'n_estimators': 455},
                                  cat_features=categorical_features, verbose=False, train_dir='/tmp/catboost_info')
    regressor.fit(X_train, y_train)

    return regressor.predict(X_test)
