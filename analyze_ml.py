from dungeons_and_data import DungeonsAndData
import pandas as pd
import numpy as np
import altair as alt

def machine_learning_model_info(data, source, col, lable_type):
    """
    Takes a DungeonsAndData object, the name of the data set,
    the kind of lable being tested for, and a column to name and
    calls a machine learning,
    method to predict 15 lables in that column times,
    keeping track of accuracy scores,
    and returns a list containing the source name  and lable type,
    the average accuracy, the highest accuracy, and the
    lowest accuracy

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
        model = data.predict_from_stats(col, max_level=lvl, min_frequency=20)
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
    s_ml_chart.save('stat_ml_chart.html')



if __name__ == '__main__':
    main()
