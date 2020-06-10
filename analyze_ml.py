from dungeons_and_data import DungeonsAndData
import pandas as pd
import numpy as np
import altair as alt


def machine_learning_model_info(data, source, col, lable_type, 
                                for_stats=True):
    """
    Takes a DungeonsAndData object, the name of the data set,
    the kind of lable being tested for, and a column to name and
    calls a machine learning,
    method to predict 15 lables in that column times,
    keeping track of accuracy scores,
    and returns a list containing the source name  and lable type,
    the average accuracy, the highest accuracy, and the
    lowest accuracy

    if for_stats is true, uses stats to predict the column
    if for_stats is false, usese name to predict the column
    if not defined, for_stats = true

    only predicts classes for levels 1-7 and lables
    require a minimum frequency of 20 to be included
    in tests
    """
    lvl = 20
    if col == 'class' or col == 'justClass':
        lvl = 7
    max_acc = 0
    min_acc = 1
    acc_scores = list()

    for i in range(15):
        if(for_stats):
            model = data.predict_from_stats(col, max_level=lvl, min_frequency=20)
        else:
            model = data.predict_from_name(col, max_level=lvl, min_frequency=20)
        acc_scores.append(model.get_test_acc())
        min_acc = min(model.get_test_acc(), min_acc)
        max_acc = max(model.get_test_acc(), max_acc)
    mean_acc = sum(acc_scores)
    model_info = [source, mean_acc, min_acc, max_acc, lable_type]
    return model_info


def main():
    app_data = pd.read_csv('app_data_processed.csv')
    dnd_app = DungeonsAndData(app_data)

    survey_data = pd.read_csv('survey_data_processed.csv')
    dnd_survey = DungeonsAndData(survey_data)

    stat_ml_info = list()
    stat_ml_info.append(machine_learning_model_info(dnd_app, 'app',
                                                   'justClass',
                                                   'class'))
    stat_ml_info.append(machine_learning_model_info(dnd_app, 'app',
                                                    'processedRace',
                                                    'race'))
    stat_ml_info.append(machine_learning_model_info(dnd_survey, 'survey',
                                                    'class', 'class'))
    stat_ml_info.append(machine_learning_model_info(dnd_survey, 'survey',
                                                    'race', 'race'))

    s_ml_data = pd.DataFrame(stat_ml_info, columns=['source', 'accuracy',
                                                       'min', 'max', 'type'])

    s_ml_chart = alt.Chart(s_ml_data).mark_bar().encode(
                                                        x='source:O',
                                                        y='accuracy:Q',
                                                        color='type:O',
                                                        column='source:O')
    min_line = alt.Chart(s_ml_data).mark_tick(color='black',
                                           thickness=2,
                                           size=20).encode(x='source:O',
                                                           y='min_acc:Q')
    max_line = alt.Chart(s_ml_data).mark_tick(color='black',
                                              thickness=2,
                                              size=20).encode(x='source:O',
                                                              y='max_acc:Q')
    stat_ml_info_chart = s_ min_line + max_line
    s_ml_chart.save('s_ml_chart.html')


    name_ml_info = list()
    name_ml_info.append(machine_learning_model_info(dnd_app, 'app',
                                                   'justClass', 'class',
                                                   False))
    name_ml_info.append(machine_learning_model_info(dnd_app, 'app',
                                                    'processedRace',
                                                    'race', False))
    name_ml_info.append(machine_learning_model_info(dnd_survey, 'survey',
                                                    'class', 'class',
                                                    False))
    name_ml_info.append(machine_learning_model_info(dnd_survey, 'survey',
                                                    'race', 'race', False))

    name_ml_data = pd.DataFrame(stat_ml_info, columns=['source', 'accuracy',
                                                       'min', 'max', 'type'])

    name_ml_chart = alt.Chart(n_ml_data).mark_bar().encode( x='source:O',
                                                               y='accuracy:Q',
                                                           color='type:O',
                                                           column='source:O')
    n_ml_chart.save('n_ml_chart.html')


if __name__ == '__main__':
    main()
