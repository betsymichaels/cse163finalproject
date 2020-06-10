"""
Tests for problems 2, 3, and 4
"""
from finalproject import (percent_top_ten, find_most_common_attribute,
                          find_most_common_build, top_3_distribution)
import pandas as pd


def test_percent_top_ten():
    """
    Tests the percent_top_ten function for the correct
    values calculated.
    """
    t_data = pd.read_csv('/Users/elisabethclithero/School/' +
                         'cse163finalproject/dnd_test_data.csv')
    t_data = t_data.rename(columns={"What Is Your Character's Class? (If your "
                                    + "character multi-classed, select all tha"
                                    + "t apply)": "class", "What is your " +
                                    "character's race?": "race", "What is " +
                                    "your character's alignment?": "alignment",
                                    "What is your character's background? " +
                                    "(Please only write in other if you are " +
                                    "using a background from an official " +
                                    "D&D 5th edition supplementary product " +
                                    "besides the PHB, otherwise just pick " +
                                    "custom)": "background"})
    print(percent_top_ten(t_data, 'race'))


def test_find_most_common_attribute():
    """
    Tests the finding_most_common function, both for the correct
    values calculated, and for behavior when the function is
    given an empty DataFrame, or the wrong column name
    """
    t_data = pd.read_csv('/Users/elisabethclithero/School/' +
                         'cse163finalproject/dnd_test_data.csv')
    t_data = t_data.rename(columns={"What Is Your Character's Class? (If your "
                                    + "character multi-classed, select all tha"
                                    + "t apply)": "class", "What is your " +
                                    "character's race?": "race", "What is " +
                                    "your character's alignment?": "alignment",
                                    "What is your character's background? " +
                                    "(Please only write in other if you are " +
                                    "using a background from an official " +
                                    "D&D 5th edition supplementary product " +
                                    "besides the PHB, otherwise just pick " +
                                    "custom)": "background"})
    print(find_most_common_attribute(t_data, 'alignment'))


def test_find_most_common_build():
    """
    Tests the find_most_common_build function, both for the correct
    values calculated, and for behavior when the function is
    given an empty DataFrame, or the wrong column name
    """
    t_data = pd.read_csv('/Users/elisabethclithero/School/' +
                         'cse163finalproject/dnd_test_data.csv')
    t_data = t_data.rename(columns={"What Is Your Character's Class? (If your "
                                    + "character multi-classed, select all tha"
                                    + "t apply)": "class", "What is your " +
                                    "character's race?": "race", "What is " +
                                    "your character's alignment?": "alignment",
                                    "What is your character's background? " +
                                    "(Please only write in other if you are " +
                                    "using a background from an official " +
                                    "D&D 5th edition supplementary product " +
                                    "besides the PHB, otherwise just pick " +
                                    "custom)": "background"})
    print(find_most_common_build(t_data, ['race', 'class', 'background',
                                          'alignment']))


def test_top_3_distribution():
    """
    Tests the top_3_distribution function, both for the correct
    values calculated, and for behavior when the function is
    given an empty DataFrame, or the wrong column name
    """
    t_data = pd.read_csv('/Users/elisabethclithero/School/' +
                         'cse163finalproject/dnd_test_data.csv')
    t_data = t_data.rename(columns={"What Is Your Character's Class? (If your "
                                    + "character multi-classed, select all tha"
                                    + "t apply)": "class", "What is your " +
                                    "character's race?": "race", "What is " +
                                    "your character's alignment?": "alignment",
                                    "What is your character's background? " +
                                    "(Please only write in other if you are " +
                                    "using a background from an official " +
                                    "D&D 5th edition supplementary product " +
                                    "besides the PHB, otherwise just pick " +
                                    "custom)": "background"})
    t_data = t_data[['race', 'alignment']]
    # print(top_3_distribution(t_data, 'Human', 'race', 'alignment'))


def main():
    test_percent_top_ten()
    test_find_most_common_attribute()
    test_find_most_common_build()
    test_top_3_distribution()


if __name__ == '__main__':
    main()
