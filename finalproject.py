"""

"""

import pandas as pd
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
# import altair as alt


def problem2(data):
    """
    Takes in the app data and prints the top 10 most common races,
    the top 10 most common classes, and the top 10 most common
    race+class combinations.
    """
    class_race = data[['processedRace', 'justClass']].copy()
    class_race['Total'] = 1
    mask = class_race['justClass'].str.contains('|', regex=False)
    multiclassed = class_race[mask].copy()
    not_multiclassed = class_race[~mask].copy()
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
    Takes in app data and prints the most common character build by:
    race, class with multiclasses removed, background, and alignment.
    Also prints the same as above,  but with multiclasses instead of
    single classes.
    Also prints most common race, class, background, and alignment.
    """
    filter_down = data[['justClass', 'processedRace', 'processedAlignment',
                        'background']].copy()
    filter_down['Total'] = 1
    filter_down.dropna()
    mask = filter_down['justClass'].str.contains('|', regex=False)
    multiclassed = filter_down[mask].copy()
    not_multiclassed = filter_down[~mask].copy()
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
    This is the machine learning model that doesn't really work
    """
    data = data[['justClass', 'background', 'processedAlignment',
                 'processedRace']].copy()
    data = data.dropna()
    mask = data['justClass'].str.contains('|', regex=False)
    not_multiclassed = data[~mask].copy()
    features = not_multiclassed[['justClass', 'background',
                                 'processedAlignment']]
    features = pd.get_dummies(features)
    labels = not_multiclassed['processedRace']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    label_predictions = model.predict(features_test)
    label_predictions2 = model.predict(features_train)
    print('Test Accuracy:', accuracy_score(labels_test, label_predictions))
    print('Training Accuracy:', accuracy_score(labels_train,
                                               label_predictions2))


def problem2survey(data):
    """
    Same as other problem 2, but uses survey data.
    """
    class_race = data[['race', 'class']].copy()
    class_race['Total'] = 1
    mask = class_race['class'].str.contains(';', regex=False)
    multiclassed = class_race[mask].copy()
    not_multiclassed = class_race[~mask].copy()
    class_df = not_multiclassed[['class', 'Total']]
    race_df = class_race[['race', 'Total']]
    class_df = class_df.dropna()
    multiclass_df = multiclassed[['class', 'Total']]
    race_df = race_df.dropna()
    grouped_race = race_df.groupby('race')['Total'].sum()
    grouped_class = class_df.groupby('class')['Total'].sum()
    grouped_multiclass = multiclass_df.groupby('class')['Total'].sum()
    sorted_race = grouped_race.sort_values(ascending=False)
    sorted_class = grouped_class.sort_values(ascending=False)
    sorted_multi = grouped_multiclass.sort_values(ascending=False)
    not_multiclassed['class_race'] = (class_race['class'] +
                                      class_race['race'])
    grouped_cr = not_multiclassed.groupby('class_race')['Total'].sum()
    sorted_class_race = grouped_cr.sort_values(ascending=False)
    print(sorted_race.nlargest(10, keep='first'))
    print(sorted_class.nlargest(10, keep='first'))
    print(sorted_class_race.nlargest(10, keep='first'))
    print(sorted_multi.nlargest(10, keep='first'))


def problem3survey(data):
    """
    Same as other problem 3, but with survey data.
    """
    filter_down = data[['class', 'race', 'alignment',
                        'background']].copy()
    filter_down['Total'] = 1
    filter_down.dropna()
    mask = filter_down['class'].str.contains(';', regex=False)
    multiclassed = filter_down[mask].copy()
    not_multiclassed = filter_down[~mask].copy()
    class_df = not_multiclassed[['class', 'Total']]
    race_df = not_multiclassed[['race', 'Total']]
    backgrnd_df = not_multiclassed[['background', 'Total']]
    align_df = not_multiclassed[['alignment', 'Total']]
    not_multiclassed['combination'] = (not_multiclassed['class'] +
                                       not_multiclassed['race'] +
                                       not_multiclassed['alignment'])
    multiclassed['combination'] = (multiclassed['class'] +
                                   multiclassed['race'] +
                                   multiclassed['alignment'])
    grouped_combo = not_multiclassed.groupby('combination')['Total'].sum()
    grouped_comboclass = multiclassed.groupby('combination')['Total'].sum()
    grouped_race = race_df.groupby('race')['Total'].sum()
    grouped_class = class_df.groupby('class')['Total'].sum()
    grouped_backgrnd = backgrnd_df.groupby('background')['Total'].sum()
    grouped_align = align_df.groupby('alignment')['Total'].sum()
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


def problem4survey(data):
    """
    Same as other problem 4, but with app data.
    """
    data = data[['class', 'alignment', 'race']].copy()
    data = data.dropna()
    mask = data['class'].str.contains(';', regex=False)
    multiclassed = data[mask].copy()
    features = multiclassed[['class', 'alignment']]
    features = pd.get_dummies(features)
    labels = multiclassed['race']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    label_predictions = model.predict(features_test)
    label_predictions2 = model.predict(features_train)
    print('Test Accuracy:', accuracy_score(labels_test, label_predictions))
    print('Training Accuracy:', accuracy_score(labels_train,
                                               label_predictions2))


def problem4actual(data, race):
    """
    Takes in app data and a race(as a string), prints the top
    3 classes, backgrounds, and alignments for that race.
    """
    df = data[['race', 'class', 'background', 'alignment']].copy()
    df['Total'] = 1
    select_race = df[df['race'] == race]
    class_df = select_race[['class', 'Total']]
    bckgrnd_df = select_race[['background', 'Total']]
    align_df = select_race[['alignment', 'Total']]
    grouped_class = class_df.groupby('class')['Total'].sum()
    grouped_backgrnd = bckgrnd_df.groupby('background')['Total'].sum()
    grouped_align = align_df.groupby('alignment')['Total'].sum()
    sorted_class = grouped_class.sort_values(ascending=False)
    sorted_backgrnd = grouped_backgrnd.sort_values(ascending=False)
    sorted_align = grouped_align.sort_values(ascending=False)
    print('Top 3', race, 'Classes:',
          sorted_class.nlargest(3, keep='first'))
    print('Top 3', race, 'Backgrounds:',
          sorted_backgrnd.nlargest(3, keep='first'))
    print('Top 3', race, 'Alignments:',
          sorted_align.nlargest(3, keep='first'))


def main():
    file1 = 'uniqueTable.tsv'
    file2 = 'survey_data_processed.csv'
    data1 = pd.read_table(file1)
    data2 = pd.read_csv(file2)
    data2 = data2.rename(columns={"What Is Your Character's Class? (If your " +
                                  "character multi-classed, select all that " +
                                  "apply)": "class", "What is your " +
                                  "character's race?": "race", "What is " +
                                  "your character's alignment?": "alignment",
                                  "What is your character's background? " +
                                  "(Please only write in other if you are " +
                                  "using a background from an official " +
                                  "D&D 5th edition supplementary product " +
                                  "besides the PHB, otherwise just pick " +
                                  "custom)": "background"})
    problem2(data1)
    problem3(data1)
    problem4(data1)
    problem2survey(data2)
    problem3survey(data2)
    problem4survey(data2)
    problem4actual(data2, 'Human')
    problem4actual(data2, 'Dwarf')
    problem4actual(data2, 'Dragonborn')


if __name__ == '__main__':
    main()
