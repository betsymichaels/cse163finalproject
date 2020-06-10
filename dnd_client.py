from dungeons_and_data import DungeonsAndData
import pandas as pd


def main():
    app_data = pd.read_csv('app_data_processed.csv')
    dnd_app = DungeonsAndData(app_data)

    survey_data = pd.read_csv('survey_data_processed.csv')
    dnd_survey = DungeonsAndData(survey_data)


if __name__ == '__main__':
    main()
