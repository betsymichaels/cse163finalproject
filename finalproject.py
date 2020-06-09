"""

"""

import pandas as pd
import altair as alt


def percent_top_ten(data, column_name):
    """
    Takes in column name and prints the percentages of the
    top ten most common items in that column.
    """
    column = data[column_name]
    column = column.dropna()

    counts = column.value_counts()
    counts = counts.nlargest(10)
    counts = counts.to_frame('percent')
    counts[column_name] = counts.index
    counts['percent'] = counts['percent'] / len(column) * 100
    return counts
    # chart = alt.Chart(counts).mark_bar().encode()
    # chart.save('most_common_10.html')


def finding_most_common(df, column_name):
    """
    """
    column = df[column_name]
    column = column.dropna()

    counts = column.value_counts()
    most_common = counts.nlargest(1)
    return most_common


def character_build(data):
    """
    Takes in app data and prints the most common character build by:
    race, class with multiclasses removed, background, and alignment.
    Also prints the most common character build with multiclasses instead of
    single classes.
    Also prints most common race, class, background, and alignment.
    """
    filter_down = data[['justClass', 'processedRace', 'processedAlignment',
                        'background']].copy()
    filter_down.dropna()
    # removing multiclasses into their own dataframe
    mask = filter_down['justClass'].str.contains('|', regex=False)
    multiclassed = filter_down[mask].copy()
    not_multiclassed = filter_down[~mask].copy()
    # calling finding_most_common on the four traits
    common_class = finding_most_common(not_multiclassed, 'justClass')
    common_race = finding_most_common(not_multiclassed, 'processedRace')
    common_backgrnd = finding_most_common(not_multiclassed, 'background')
    common_align = finding_most_common(not_multiclassed, 'processedAlignment')
    # creating a column for combined character build, with and without
    # multiclassing
    not_multiclassed['combination'] = (not_multiclassed['justClass'] +
                                       not_multiclassed['processedRace'] +
                                       not_multiclassed['background'] +
                                       not_multiclassed['processedAlignment'])
    multiclassed['combination'] = (multiclassed['justClass'] +
                                   multiclassed['processedRace'] +
                                   multiclassed['background'] +
                                   multiclassed['processedAlignment'])
    common_build = finding_most_common(not_multiclassed, 'combination')
    common_multi = finding_most_common(multiclassed, 'combination')
    # printing results(this was for debugging, going to have to return
    # these dataframes in the final product)
    print('Most Common Character Build:', common_build)
    print('Most Common Multiclass Character Build:', common_multi)
    print('Most Common Race:', common_race)
    print('Most Common Class:', common_class)
    print('Most Common Background:', common_backgrnd)
    print('Most Common Alignment:', common_align)
    result = pd.concat([common_build + common_multi + common_race +
                        common_class, common_backgrnd + common_align],
                       ignore_index=True)
    return result


def top_3_distribution(data, race):
    """
    Takes in app data and a race(as a string), prints the top
    3 classes, backgrounds, and alignments for that race.
    """
    df = data[['processedRace', 'justClass', 'background',
               'processedAlignment']].copy()
    # creating a column to help sum up the number of same objects
    df['Total'] = 1
    # filtering down to just the race specified in the parameters
    select_race = df[df['processedRace'] == race]
    # finding the most common classes, backgrounds, and alignments
    # for that race
    class_df = select_race[['justClass', 'Total']]
    bckgrnd_df = select_race[['background', 'Total']]
    align_df = select_race[['processedAlignment', 'Total']]
    grouped_class = class_df.groupby('justClass')['Total'].sum()
    grouped_backgrnd = bckgrnd_df.groupby('background')['Total'].sum()
    grouped_align = align_df.groupby('processedAlignment')['Total'].sum()
    sorted_class = grouped_class.sort_values(ascending=False)
    sorted_backgrnd = grouped_backgrnd.sort_values(ascending=False)
    sorted_align = grouped_align.sort_values(ascending=False)
    # printing results(this was for debugging, going to have to return
    # these dataframes in the final product)
    print('Top 3', race, 'Classes:',
          sorted_class.nlargest(3, keep='first'))
    print('Top 3', race, 'Backgrounds:',
          sorted_backgrnd.nlargest(3, keep='first'))
    print('Top 3', race, 'Alignments:',
          sorted_align.nlargest(3, keep='first'))
    chart = alt.Chart(data)
    chart.mark_point().encode()


def most_common_10_survey(data):
    """
    Same as other most_common_10, but uses survey data.
    """
    class_race = data[['race', 'class']].copy()
    class_race = class_race.dropna()
    # creating a column to help sum up the number of same objects
    class_race['Total'] = 1
    # removing multiclasses into their own dataframe
    mask = class_race['class'].str.contains(';', regex=False)
    multiclassed = class_race[mask].copy()
    not_multiclassed = class_race[~mask].copy()
    # creating dataframes to group and sort by most to least
    # common race, class, and multiclass
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
    # creating a new column with combined class and race, to find
    # the top 10 race/class combinations
    not_multiclassed['class_race'] = (class_race['class'] +
                                      class_race['race'])
    grouped_cr = not_multiclassed.groupby('class_race')['Total'].sum()
    sorted_class_race = grouped_cr.sort_values(ascending=False)
    # printing results(this was for debugging, going to have to return
    # these dataframes in the final product)
    print(sorted_race.nlargest(10, keep='first'))
    print(sorted_class.nlargest(10, keep='first'))
    print(sorted_class_race.nlargest(10, keep='first'))
    print(sorted_multi.nlargest(10, keep='first'))
    chart = alt.Chart(data)
    chart.mark_point().encode()


def character_build_survey(data):
    """
    Same as other character_build function, but with survey data.
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
    # printing results(this was for debugging, going to have to return
    # these dataframes in the final product)
    print('Most Common Character Build:', sorted_combo.nlargest(1,
                                                                keep='first'))
    print('Most Common Multiclass Character Build:',
          sorted_comboclass.nlargest(1, keep='first'))
    print('Most Common Race:', sorted_race.nlargest(1, keep='first'))
    print('Most Common Class:', sorted_class.nlargest(1, keep='first'))
    print('Most Common Background:', sorted_backgrnd.nlargest(1, keep='first'))
    print('Most Common Alignment:', sorted_align.nlargest(1, keep='first'))
    chart = alt.Chart(data)
    chart.mark_point().encode()


def top_3_distribution_survey(data, race):
    """
    Takes in survey data and a race(as a string), prints the top
    3 classes, backgrounds, and alignments for that race.
    """
    df = data[['race', 'class', 'background', 'alignment']].copy()
    # creating a column to help sum up the number of same objects
    df['Total'] = 1
    # filtering down to just the race specified in the parameters
    select_race = df[df['race'] == race]
    # finding the most common classes, backgrounds, and alignments
    # for that race
    class_df = select_race[['class', 'Total']]
    bckgrnd_df = select_race[['background', 'Total']]
    align_df = select_race[['alignment', 'Total']]
    grouped_class = class_df.groupby('class')['Total'].sum()
    grouped_backgrnd = bckgrnd_df.groupby('background')['Total'].sum()
    grouped_align = align_df.groupby('alignment')['Total'].sum()
    sorted_class = grouped_class.sort_values(ascending=False)
    sorted_backgrnd = grouped_backgrnd.sort_values(ascending=False)
    sorted_align = grouped_align.sort_values(ascending=False)
    # printing results(this was for debugging, going to have to return
    # these dataframes in the final product)
    print('Top 3', race, 'Classes:',
          sorted_class.nlargest(3, keep='first'))
    print('Top 3', race, 'Backgrounds:',
          sorted_backgrnd.nlargest(3, keep='first'))
    print('Top 3', race, 'Alignments:',
          sorted_align.nlargest(3, keep='first'))
    chart = alt.Chart(data)
    chart.mark_point().encode()


def main():
    file1 = '/Users/elisabethclithero/Downloads/finalproject/app_' \
            'data_processed.txt'
    file2 = '/Users/elisabethclithero/Downloads/finalproject/survey_' \
            'data_processed.txt'
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    # renaming columns in the survey data
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
    # calling the functions
    print(percent_top_ten(data1, 'processedRace'))
    print()
    print(percent_top_ten(data2, 'race'))
    print()
    print(percent_top_ten(data1, 'justClass'))
    print()
    print(percent_top_ten(data2, 'class'))
    print()
    data1_copy = data1[['processedRace', 'justClass']].copy()
    data1_copy['class_race'] = (data1_copy['processedRace'] +
                                data1_copy['justClass'])
    print(percent_top_ten(data1_copy, 'class_race'))
    print()
    data2_copy = data2[['race', 'class']].copy()
    data2_copy['class_race'] = (data2_copy['race'] +
                                data2_copy['class'])
    print(percent_top_ten(data2_copy, 'class_race'))

    # chart = alt.Chart(percent1).mark_bar().encode(x='processedRace',
    #                                              y='percent')
    # chart = alt.Chart(percent2).mark_bar().encode(x='race',
    #                                              y='percent')
    # chart.save('most_common_10.html')
    print(character_build(data1))
    # top_3_distribution(data1, 'Human')
    # top_3_distribution(data1, 'Dwarf')
    # top_3_distribution(data1, 'Dragonborn')
    # character_build_survey(data2)
    # top_3_distribution_survey(data2, 'Human')
    # top_3_distribution_survey(data2, 'Dwarf')
    # top_3_distribution_survey(data2, 'Dragonborn')


if __name__ == '__main__':
    main()
