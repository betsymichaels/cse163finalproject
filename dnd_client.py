from dungeons_and_data import DungeonsAndData
import pandas as pd
import altair as alt


def main():
    app_data = pd.read_csv('app_data_processed.csv')
    dnd_app = DungeonsAndData(app_data)

    survey_data = pd.read_csv('survey_data_processed.csv')
    dnd_survey = DungeonsAndData(survey_data)

    thing = dnd_survey.find_most_common_build(['race', 'class', 'background', 'alignment'])
    print(thing)

    dnd_app.mean_stats_per_classifier('justClass', max_level=7)

    chart = alt.Chart(source).mark_point().encode(x='Wis', y='Int',
            size='level')

    chart.save('chart.html')


if __name__ == '__main__':
    main()
