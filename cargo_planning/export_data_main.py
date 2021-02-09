import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from operator import itemgetter


def export_data(box_to_truck, truck_size_dict, dest_to_dict,load_pattern):
    """
    Converts the given excel sheet containing shipping boxes into a dictionary
    for the main program. The columns will need to be renamed before reading.
    The function assumes that two shipping boxes are the same if they share all
    of the following four attributes are the same : Stackability, Destination, Box-ID and Weight

    Parameters
    ----------
    file : path to the excel file that contains information regarding the boxes to be packed
    and available truck sizes.
    File needs to have two sheets
        - box_data   : information of all the boxes with following columns
                       ['Length', 'Width', 'Height', 'Quantity', 'Weight', 'Destination', 'Stackable']
        - truck_size : n x 3 rows with one row for each truck in [length, breadth, height] format

    Returns
    -------
    box_dict : A unique box F-code as keys and various box attributes as values.

    f_to_r_dict : Boxes are seen to be named using R-code in excel sheets but they are not unique.
    F-code is unique.

    f_to_r_dict : Returns the R-code of a box using its F-code

    dest_to_dict : Dictionary containing destination names and corresponding integer codes

    truck_size_dict : Dictionary with keys as truck names and values as the corresponding truck sizes
    """

    box_to_truck["Stackable"] = box_to_truck["Stackable"].map(
        {"Yes": 0, "No": 1})
    box_to_truck["Destination"] = box_to_truck["Destination"].apply(
        lambda x: x.strip())
    box_to_truck["Destination"] = box_to_truck["Destination"].map(dest_to_dict)

    # Counting the boxes and destinations for report generation stage
    box_per_dest = box_to_truck[['Destination', 'no_of_boxes']].groupby(
        'Destination', as_index=False).sum()
    box_per = {box_per_dest['Destination'][i]: box_per_dest['no_of_boxes'][i]
               for i in range(box_per_dest.shape[0])}
    # print(box_per)
    rev_dest_codes = {v:k for k,v in dest_to_dict.items()}
    # print(rev_dest_codes)
    with open("ui_input/write_pdf_1.txt", "w") as f:
        f.write(str(box_per) + "\n")
        f.write(str(rev_dest_codes))

    # Aggregation and sorting of boxes
    if load_pattern == 'Back' or load_pattern == 'Rear Loading':
        box_df_grouped = (
        box_to_truck.groupby(
            ["Stackable", "Destination", "Box ID", "Weight"], as_index=False
        )
        .agg(
            {"Length": "mean", "Width": "mean",
                "Height": "mean", "no_of_boxes": "sum"}
        )
        #.sort_values(
        #    by=["Stackable", "Destination", "Width",
        #        "Length", "Height", "Weight"],
        #    ascending=[True, True, False, False, False, False],
        .sort_values(
            by=["Stackable","Width","Height", "Destination",
                "Length", "Weight"],
            ascending=[True, False , False, False, False, False],
        )
    )
    else:
        box_df_grouped = (
        box_to_truck.groupby(
            ["Stackable", "Destination", "Box ID", "Weight"], as_index=False
        )
        .agg(
            {"Length": "mean", "Width": "mean",
                "Height": "mean", "no_of_boxes": "sum"}
        )
        .sort_values(
            by=["Stackable", "Destination", "Width",
                "Length", "Height", "Weight"],
            ascending=[True, False, False, False, False, False],
        )
    )

    # print(box_df_grouped)

    box_df_grouped.rename({"no_of_boxes": "Quantity"}, inplace=True, axis=1)
    box_df_grouped = box_df_grouped[
        [
            "Box ID",
            "Length",
            "Width",
            "Height",
            "Quantity",
            "Weight",
            "Destination",
            "Stackable",
        ]
    ]
    box_df_grouped.reset_index(inplace=True, drop=True)

    # F-code is a unique columns to distinguish individual rows
    box_df_grouped["F-code"] = [
        "F-" + str(x) for x in range(1, box_df_grouped.shape[0] + 1)
    ]
    box_dict = (
        box_df_grouped[
            [
                "Length",
                "Width",
                "Height",
                "Quantity",
                "Weight",
                "Destination",
                "Stackable",
                "F-code",
            ]
        ]
        .set_index("F-code")
        .T.to_dict("list")
    )
    f_to_r_dict = (
        box_df_grouped[["Box ID", "F-code"]
                       ].set_index("F-code").T.to_dict("list")
    )
    # print(dest_to_dict)
    return box_dict, f_to_r_dict, dest_to_dict, truck_size_dict


# a,b,c,d = export_data('../data/Container_2.xlsx')
