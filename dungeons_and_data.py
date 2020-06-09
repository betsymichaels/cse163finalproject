import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class DungeonsAndData:
    """

    """
    def __init__(self, data):
        """

        """
        self._data = data

    def _filter_level(self, min_level=1, max_level=20):
        """

        """
        low_level = self._data['level'] >= min_level
        high_level = self._data['level'] <= max_level
        return self._data[low_level & high_level]

    def predict_from_stats(self, label_type, min_level=1, max_level=20):
        """

        """
        data = self._filter_level(min_level, max_level)

        data = data.loc[:, [label_type, 'HP', 'AC', 'Str', 'Dex',
                        'Con', 'Int', 'Wis', 'Cha']]
        data = data.dropna()

        features = data.loc[:, data.columns != label_type]
        labels = data[label_type]

        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3)

        model = DecisionTreeClassifier(min_samples_leaf=3)
        model.fit(features_train, labels_train)

        pred_test = model.predict(features_test)
        test_acc = accuracy_score(labels_test, pred_test)
        return test_acc

    def _char_ratio(self, name, letters, spec_case=None,
                    cons=False, vow=False):
        """

        """
        if spec_case is None:
            spec_case = {}

        name = re.sub(r'\W+|\d+', '', name)
        count = 0
        total_valid = 0

        vowels = {'a', 'e', 'i', 'o', 'u'}

        for c in name:
            valid_char = True
            not_vow = vow and c not in vowels
            not_cons = cons and c in vowels

            if not_vow or not_cons or c not in spec_case:
                valid_char = False

            if c in letters:
                count += 1
            if valid_char:
                total_valid += 1

        if total_valid == 0:
            return 0
        return count / total_valid

    def _generate_name_data(self, data):
        """

        """
        data['vow ratio'] = data['name'].apply(lambda s:
                                               self._char_ratio(s.lower(),
                                                                set('aeiou'),
                                                                spec_case='y'))
        data['front vow'] = data['name'].apply(lambda s:
                                               self._char_ratio(s.lower(),
                                                                set('ei'),
                                                                vow=True))
        data['broad vow'] = data['name'].apply(lambda s:
                                               self._char_ratio(s.lower(),
                                                                set('eo'),
                                                                vow=True))
        data['dental cons'] = data['name'].apply(lambda s:
                                                 self._char_ratio(s.lower(),
                                                                  set('tdn' +
                                                                  'szrlx'),
                                                                  cons=True))
        data['labial cons'] = data['name'].apply(lambda s:
                                                 self._char_ratio(s.lower(),
                                                                  set('pbvfm'),
                                                                  cons=True))
        data['velar cons'] = data['name'].apply(lambda s:
                                                self._char_ratio(s.lower(),
                                                                 set('kgcqx'),
                                                                 cons=True))
        data['nasal cons'] = data['name'].apply(lambda s:
                                                self._char_ratio(s.lower(),
                                                                 set('mn'),
                                                                 cons=True))
        data['plosives'] = data['name'].apply(lambda s:
                                              self._char_ratio(s.lower(),
                                                               set('tdkgp' +
                                                               'bcqx'),
                                                               cons=True))
        data['frictives'] = data['name'].apply(lambda s:
                                               self._char_ratio(s.lower(),
                                                                set('fvszx'),
                                                                cons=True))
        data['hard cons'] = data['name'].apply(lambda s:
                                               self._char_ratio(s.lower(),
                                                                set('tdkgcqx'),
                                                                cons=True))
        data['unique cons'] = data['name'].apply(lambda s:
                                                 self._char_ratio(s.lower(),
                                                                  set('rljh'),
                                                                  cons=True))
        data['voiced'] = data['name'].apply(lambda s:
                                            self._char_ratio(s.lower(),
                                                             set('dgzbvj'),
                                                             spec_case=set('' +
                                                             'xmnrlwy'),
                                                             cons=True))

    def predict_from_name(self, label_type):
        """

        """
        name_data = self._data[['name', label_type]]
        name_data = name_data.dropna()

        self._generate_name_data(name_data)
        name_data = name_data.loc[:, name_data.columns != 'name']

        features = name_data.loc[:, name_data.columns != label_type]
        labels = name_data[label_type]

        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.2)

        model = DecisionTreeClassifier()
        model.fit(features_train, labels_train)

        pred_test = model.predict(features_test)
        test_acc = accuracy_score(labels_test, pred_test)
        return test_acc

    def label_accuracy_from_stats(self, model, class_name, col='class',
                                  min_level=1, max_level=20):
        """

        """
        data = self._filter_level(min_level, max_level)

        data = data.loc[:, [col, 'AC', 'HP', 'Str', 'Dex',
                            'Con', 'Int', 'Wis', 'Cha']]
        data = data.dropna()

        of_class = data[col] == class_name
        df_class = data[of_class]

        labels = df_class[col]
        features = df_class.loc[:, df_class.columns != col]

        class_pred = model.predict(features)
        class_acc = accuracy_score(labels, class_pred)

        return class_acc

    def label_from_stat_block(self, model, stat_block):
        """

        """
        stat_row = dict()
        for stat in stat_block:
            stat_row[stat] = {0: stat_block[stat]}

        data = pd.DataFrame.from_dict(stat_row)
        prediction = model.predict(data)
        return prediction

    def label_from_name(self, model, name):
        name_dict = {'name': {0: name}}

        data = pd.DataFrame.from_dict(name_dict)
        self._generate_name_data(data)

        features = data.loc[:, data.columns != 'name']

        prediction = model.predict(features)
        return prediction

    def mean_stats_per_classifier(self, class_type, min_level=1, max_level=20):
        """

        """
        data = self._filter_level(min_level, max_level)
        data = data.groupby(class_type).mean()
        data = data.loc[:, 'HP':'Cha']
        for column in data:
            data[column] = data[column].apply(lambda x: round(x))
        return data
