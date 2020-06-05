"""

"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt


def problem2(data):
    """
    """
    class_race = data[['processedRace', 'justClass']]
    class_race['Total'] = 1
    mask = class_race['justClass'].str.contains('|', regex=False)
    multiclassed = class_race[mask]
    not_multiclassed = class_race[~mask]
    class_df = not_multiclassed[['justClass', 'Total']]
    race_df = class_race[['processedRace', 'Total']]
    class_df = class_df.dropna()
    multiclass_df = multiclassed[['justClass', 'Total']]
    race_df = race_df.dropna()
    grouped_race = race_df.groupby('processedRace')['Total'].sum()
    grouped_class = class_df.groupby('justClass')['Total'].sum()
    grouped_multiclass = multiclass_df.groupby('justClass')['Total'].sum()
    sorted_race = grouped_race.sort_values(ascending=False)
    sorted_class = grouped_class.sort_values(ascending=False)
    sorted_multi = grouped_multiclass.sort_values(ascending=False)
    not_multiclassed['class_race'] = (class_race['justClass'] +
                                      class_race['processedRace'])
    grouped_cr = not_multiclassed.groupby('class_race')['Total'].sum()
    sorted_class_race = grouped_cr.sort_values(ascending=False)
    print(sorted_race.nlargest(10, keep='first'))
    print(sorted_class.nlargest(10, keep='first'))
    print(sorted_class_race.nlargest(10, keep='first'))
    print(sorted_multi.nlargest(10, keep='first'))


def problem3(data):
    """
    """
    filter_down = data[['justClass', 'processedRace', 'processedAlignment',
                        'background']]
    filter_down['Total'] = 1
    filter_down.dropna()
    mask = filter_down['justClass'].str.contains('|', regex=False)
    multiclassed = filter_down[mask]
    not_multiclassed = filter_down[~mask]
    class_df = not_multiclassed[['justClass', 'Total']]
    race_df = not_multiclassed[['processedRace', 'Total']]
    backgrnd_df = not_multiclassed[['background', 'Total']]
    align_df = not_multiclassed[['processedAlignment', 'Total']]
    not_multiclassed['combination'] = (not_multiclassed['justClass'] +
                                       not_multiclassed['processedRace'] +
                                       not_multiclassed['background'] +
                                       not_multiclassed['processedAlignment'])
    multiclassed['combination'] = (multiclassed['justClass'] +
                                   multiclassed['processedRace'] +
                                   multiclassed['background'] +
                                   multiclassed['processedAlignment'])
    grouped_combo = not_multiclassed.groupby('combination')['Total'].sum()
    grouped_comboclass = multiclassed.groupby('combination')['Total'].sum()
    grouped_race = race_df.groupby('processedRace')['Total'].sum()
    grouped_class = class_df.groupby('justClass')['Total'].sum()
    grouped_backgrnd = backgrnd_df.groupby('background')['Total'].sum()
    grouped_align = align_df.groupby('processedAlignment')['Total'].sum()
    sorted_race = grouped_race.sort_values(ascending=False)
    sorted_class = grouped_class.sort_values(ascending=False)
    sorted_backgrnd = grouped_backgrnd.sort_values(ascending=False)
    sorted_align = grouped_align.sort_values(ascending=False)
    sorted_combo = grouped_combo.sort_values(ascending=False)
    sorted_comboclass = grouped_comboclass.sort_values(ascending=False)
    print('Most Common Character Build:', sorted_combo.nlargest(1,
                                                                keep='first'))
    print('Most Common Multiclass Character Build:',
          sorted_comboclass.nlargest(1, keep='first'))
    print('Most Common Race:', sorted_race.nlargest(1, keep='first'))
    print('Most Common Class:', sorted_class.nlargest(1, keep='first'))
    print('Most Common Background:', sorted_backgrnd.nlargest(1, keep='first'))
    print('Most Common Alignment:', sorted_align.nlargest(1, keep='first'))


def problem4(data):
    """
    """
    data = data[['justClass', 'background', 'processedAlignment',
                 'processedRace']]
    data = data.dropna()
    features = data[['justClass', 'background', 'processedAlignment']]
    features = pd.get_dummies(features)
    labels = data['processedRace']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    label_predictions = model.predict(features_test)
    print('Accuracy:', accuracy_score(labels_test, label_predictions))


def problem3v2(data):
    """
    """
    filter_down = data[['justClass', 'processedRace',
                        'processedAlignment']]
    filter_down['Total'] = 1
    filter_down.dropna()
    mask = filter_down['justClass'].str.contains('|', regex=False)
    multiclassed = filter_down[mask]
    not_multiclassed = filter_down[~mask]
    class_df = not_multiclassed[['justClass', 'Total']]
    race_df = not_multiclassed[['processedRace', 'Total']]
    align_df = not_multiclassed[['processedAlignment', 'Total']]
    multiclass_df = multiclassed[['justClass', 'Total']]
    not_multiclassed['combination'] = (not_multiclassed['justClass'] +
                                       not_multiclassed['processedRace'] +
                                       not_multiclassed['processedAlignment'])
    multiclassed['combination'] = (multiclassed['justClass'] +
                                   multiclassed['processedRace'] +
                                   multiclassed['processedAlignment'])
    grouped_combo = not_multiclassed.groupby('combination')['Total'].sum()
    grouped_comboclass = multiclassed.groupby('combination')['Total'].sum()
    grouped_race = race_df.groupby('processedRace')['Total'].sum()
    grouped_class = class_df.groupby('justClass')['Total'].sum()
    grouped_align = align_df.groupby('processedAlignment')['Total'].sum()
    grouped_multi = multiclass_df.groupby('justClass')['Total'].sum()
    sorted_race = grouped_race.sort_values(ascending=False)
    sorted_class = grouped_class.sort_values(ascending=False)
    sorted_align = grouped_align.sort_values(ascending=False)
    sorted_multi = grouped_multi.sort_values(ascending=False)
    sorted_combo = grouped_combo.sort_values(ascending=False)
    sorted_comboclass = grouped_comboclass.sort_values(ascending=False)
    print('Most Common Character Build:', sorted_combo.nlargest(1,
                                                                keep='first'))
    print('Most Common Multiclass Character Build:',
          sorted_comboclass.nlargest(1, keep='first'))
    print('Most Common Race:', sorted_race.nlargest(1, keep='first'))
    print('Most Common Class:', sorted_class.nlargest(1, keep='first'))
    print('Most Common Multiclass Combination:',
          sorted_multi.nlargest(1, keep='first'))
    print('Most Common Alignment:', sorted_align.nlargest(1, keep='first'))


def problem4v2(data):
    """
    """
    data = data[['justClass', 'processedAlignment',
                 'processedRace']]
    data = data.dropna()
    features = data[['justClass', 'processedAlignment']]
    features = pd.get_dummies(features)
    labels = data['processedRace']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    label_predictions = model.predict(features_test)
    print('Accuracy:', accuracy_score(labels_test, label_predictions))


def problem3v3(data):
    """
    """
    filter_down = data[['processedRace',
                        'processedAlignment']]
    filter_down['Total'] = 1
    filter_down.dropna()
    filter_down['combination'] = (filter_down['processedRace'] +
                                  filter_down['processedAlignment'])
    grouped_combo = filter_down.groupby('combination')['Total'].sum()
    sorted_combo = grouped_combo.sort_values(ascending=False)
    print('Most Common Character Build:', sorted_combo.nlargest(1,
                                                                keep='first'))


def problem4v3(data):
    """
    """
    data = data[['justClass',
                 'processedRace']]
    data = data.dropna()
    features = data[['justClass']]
    features = pd.get_dummies(features)
    labels = data['processedRace']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    label_predictions = model.predict(features_test)
    print('Accuracy:', accuracy_score(labels_test, label_predictions))


def main():
    file = '/Users/elisabethclithero/Downloads/test/uniqueTable.tsv'
    data = pd.read_table(file)
    problem2(data)
    problem3(data)
    problem4(data)
    problem3v2(data)
    problem4v2(data)
    problem3v3(data)
    problem4v3(data)


if __name__ == '__main__':
    main()
