import re
import pandas as pd

from dnd_model import DnDModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class DungeonsAndData:
    """
    This class defines an object representing dungeons and dragons
    character data and impelments methods with which to use,
    manipulate, and analyze that data
    """
    def __init__(self, data):
        """
        Takes a DataFrame containing Dungeons and Dragons character
        information and constructs a DungeonsAndData object which
        keeps track of that DataFrame
        """
        self._data = data

    def get_data(self):
        """
        Returns the DataFrame of dnd character information
        """
        return self._data

    def _drop_low_frequency(self, data, col, min_freq):
        """
        Takes a column name and a minimum frequency (min_freq) and
        DataFrame (data) and returns a mask of that data with all rows
        for which the data for that column appeard less that the minimum frequency
        removed
        """
        low = self._data[col].map(self._data[col].value_counts()) <= min_freq
        filtered = self._data[~low]
        return filtered

    def _filter_level(self, min_level=1, max_level=20):
        """
        Takes a minimum number (min_level) and a maximum
        number (max_level) and returns a DataFrame of all
        rows in the DataFrame this class represents for
        which the level of that row falls between
        min_level (inclusive) and max_level(inclusive)

        if no min_level is provided, it defaults to 1
        if no max_level is provided, it defaults to 20
        """
        low_level = self._data['level'] >= min_level
        high_level = self._data['level'] <= max_level
        return self._data[low_level & high_level]

    def predict_from_stats(self, label_type, min_level=1, max_level=20,
                           min_frequency=0):
        """
        Takes the name of a column in the DataFrame (lable_type) and
        createsa machine learning model that uses character stats
        (Hp, Ac, strenght, dexterity, constitution, intelligence, wisdom,
        and charisma) to predict the coresponding data in lable_type and
        returns a DnDModel that stores the machine learning model along with
        other information about it

        min_frequency is the minimum amount of occurences a label has to have
        to be included in the test/training data
        only anylizes data with levels from min_level(inclusive) - max_level
        (inclusive)

        if no min_level is provided, it defaults to 1
        if no max_level is provided, it defaults to 20
        if no min_frequency is provided, it defaults to 0
        """
        data = self._filter_level(min_level, max_level)

        if min_frequency > 0:
            data = self._drop_low_frequency(data, label_type, min_frequency)

        data = data.loc[:, [label_type, 'HP', 'AC', 'Str', 'Dex',
                        'Con', 'Int', 'Wis', 'Cha']]
        data = data.dropna()

        features = data.loc[:, data.columns != label_type]
        labels = data[label_type]

        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3)

        model = DecisionTreeClassifier(min_samples_leaf=3, max_depth=10)
        model.fit(features_train, labels_train)

        pred_test = model.predict(features_test)
        test_acc = accuracy_score(labels_test, pred_test)

        stat_model = DnDModel(model, label_type, test_acc)
        return stat_model

    def _char_ratio(self, name, letters, spec_case=None,
                    cons=False, vow=False):
        """
        Takes a string (string) and a set of letters (letters)
        and returns the ratio of letters in letters to
        other valid characters that appear in the string

        a character is considered invalid if vow is true and
        it is not a vowel, cons is true and it is not a consonant,
        or it is listed as a special case

        if spec_case is not defined, it defaults to None
        if cons is not defined, it defaults to False
        if vow is not defined, if defaults to False

        if there are no valid characters, returns 0
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
        Takes a DataFrame (data) and uses IPA classifications to generate
        numerical data by analyzing letters in the names stored in that
        DataFrame
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
                                                             spec_case=set(''
                                                             + 'xmnrlwy'),
                                                             cons=True))

    def predict_from_name(self, label_type, min_frequency=0):
        """
        Takes the name of a column in the DataFrame (lable_type) and
        creates a machine learning model that uses atributes of a character's
        name to predict the coresponding data in lable_type and returns a
        DnDModel from that algorithm

        min_frequency is the minimum amount of occurences a label has to
        have to be included in the test/training data
        if no min_frequency is provided, it defaults to 0
        """
        name_data = self._data[['name', label_type]].copy()
        name_data = name_data.dropna()

        if min_frequency > 0:
            name_data = self._drop_low_frequency(name_data, label_type,
                                                 min_frequency)

        self._generate_name_data(name_data)
        name_data = name_data.loc[:, name_data.columns != 'name']

        features = name_data.loc[:, name_data.columns != label_type]
        labels = name_data[label_type]

        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3)

        model = DecisionTreeClassifier()
        model.fit(features_train, labels_train)

        pred_test = model.predict(features_test)
        test_acc = accuracy_score(labels_test, pred_test)

        name_model = DnDModel(model, label_type, test_acc)
        return name_model

    def label_accuracy_from_stats(self, dnd_model, class_name, col,
                                  min_level=1, max_level=20):
        """
        Takes in a DnDModel (dnd_model), the name of a paricular classifier
        (class name) and the column it would be found it (col) and returns
        a the accuracy score with wich the model is able to predict
        members of that classifier

        only anylizes data with levels from min_level(inclusive) to
        max_level (inclusive)

        if no min_level is provided, it defaults to 1
        if no max_level is provided, it defaults to 20
        """
        data = self._filter_level(min_level, max_level)

        data = data.loc[:, [col, 'AC', 'HP', 'Str', 'Dex',
                            'Con', 'Int', 'Wis', 'Cha']]
        data = data.dropna()

        of_class = data[col] == class_name
        df_class = data[of_class]

        labels = df_class[col]
        features = df_class.loc[:, df_class.columns != col]

        class_pred = dnd_model.predict(features)
        class_acc = accuracy_score(labels, class_pred)

        return class_acc

    def label_from_stat_block(self, dnd_model, stat_block):
        """
        Takes a DnDModel, which contains an algorithm made to predict a
        particular type of classifier, and a list of stats for a dnd character
        in the order: 'AC', 'HP', 'Str', 'Dex', 'Con', 'Int', 'Wis', 'Cha'
        and predicts the clasifier of that type for a character with those
         stats
        """
        stat_row = dict()
        for stat in stat_block:
            stat_row[stat] = {0: stat_block[stat]}

        data = pd.DataFrame.from_dict(stat_row)
        prediction = dnd_model.predict(data)
        return prediction

    def label_from_name(self, dnd_model, name):
        """
        Takes a dnd_model, which contains an algorithm made to predict a
        particular type of classifier, and a character name and predicts
        the clasifier of that type for a character with that name
        """
        name_dict = {'name': {0: name}}

        data = pd.DataFrame.from_dict(name_dict)
        self._generate_name_data(data)

        features = data.loc[:, data.columns != 'name']

        prediction = dnd_model.predict(features)
        return prediction

    def percent_top_ten(self, column_name):
        """
        Takes in column name (column_name) and returns a DataFrame
        containing the top ten most common items in that column along
        with the percentage of all items in that column they make up
        """
        column = self._data[column_name]
        column = column.dropna()

        counts = column.value_counts()
        counts = counts.nlargest(10)

        counts = counts.to_frame('percent')
        counts[column_name] = counts.index
        counts['percent'] = counts['percent'] / len(column) * 100
        return counts

    def find_most_common_attribute(self, column_name):
        """
        Takes the name of a column (column_name) and returns
        the most common item to appear in that column
        """
        column = self._data[column_name]
        column = column.dropna()

        counts = column.value_counts()
        most_common = counts.nlargest(1)
        return most_common

    def find_most_common_build(self, atributes):
        """
        Takes a list of columns (attributes) and returns
        the most common combination of traits amount those columns
        """
        data = self._data.loc[:, atributes]
        data['all'] = data.apply(lambda s: s.str.cat(sep=' '), axis=1)

        counts = data['all'].value_counts()
        most_common = counts.nlargest(1)
        return most_common

    def top_3_distribution(self, race, race_column, atribute):
        """
        Takes a character race as a string (race), column representing data
        of character races(race_column), and the name of a different column
        (atribute) and returns a DataFrame of the three most common traits for
        that race within that column mapped to the percentage of members of
        that race that have that trait
        """
        of_race = self._data[race_column] == race
        just_race = self._data[of_race]
        count_attribute = just_race.groupby(atribute).count()
        top_three = count_attribute.nlargest(3)

        top_three[atribute] = top_three.index
        top_three['percent'] = top_three['percent'] / len(just_race) * 100
        return top_three
