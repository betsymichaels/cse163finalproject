from dungeons_and_data import DungeonsAndData
import pandas as pd


def main():
    app_data = pd.read_csv('app_data_processed.csv')
    dnd_app = DungeonsAndData(app_data)

    survey_data = pd.read_csv('survey_data_processed.csv')
    dnd_survey = DungeonsAndData(survey_data)

    thing = dnd_survey.percent_top_ten('race')
    print(thing)

    dnd_app.mean_stats_per_classifier('justClass', max_level=7)
    

if __name__ == '__main__':
    main()