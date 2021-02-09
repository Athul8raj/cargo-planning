# -*- coding: utf-8 -*-
"""
Created on Aug 27 10:41:01 2020

@authors: Athulraj.Puthalath, Unnikrishna Puthumana
"""
from ast import literal_eval
from operator import itemgetter
from collections import Counter
import itertools
from time import perf_counter
import numpy as np
from export_data_main import export_data
from logger_base import logger, CustomException

class PackingHeuristic:
    """
    Core of the algorithm that packs the given boxes into given trucks.
    Takes in both box and truck related data and executes the packing functions.
    Constraints :
        - Order of destinations for unloading the boxes
        - Volume of box and truck
        - Aspect constraints : length, breadth and height
        - Intersection with truck walls and other boxes
        - Base area constraint for better box support
        - Stackability constraint for non-stackable boxes

    After packing, results are stored in files that are to be later accessed
    to process the request and retrieve final packing information.

    Parameters
    ----------
    truck_size_dict :   Dictionary with
                        key : truck name (TRUCK-1 for example)
                        value : [l, b, h] truck dimensions
    box_dict      : Dictionary with
                    key : original box name used in shipping
                    value : list with the following order
                            ['Length', 'Width', 'Height', 'Quantity', 'Weight',
                             'Destination', 'Stackable']
    dest_to_dist  : Dictionary with
                    key : Destination code assigned by the user (int)
                    value : Destination name
    base_area_thr : Boxes will not be packed if the support from the bottom
                    side is less than this value (in percentage)

    """

    def __init__(self, truck_size_dict, box_dict, dest_to_dist, base_area_thr=100,
                 load_pattern='Side'):
        self.truck_size_dict = truck_size_dict
        self.box_dict = box_dict
        self.dest_to_dist = dest_to_dist
        self.base_area_thr = base_area_thr
        self.load_pattern = load_pattern


    def create_box_list(self):
        """
        Takes in the box types dictionary as the input and returns a list of
        all the boxes.

        Returns
        -------
        all_boxes_dict: Dictionary with
                        keys   - destination code
                        values - array containing the box information
                                 in the order [length  width  height  box_type  sorting_parameter]
                        each row of the array represents a one individual box
                        (there may be duplicate rows quantity of a box type is > 1)

        Dictionary (all_box_ids) : Dictionary with
                                   key   - A system generated unique key for every box
                                   value - Shipping code of the box
        """

        _ = np.zeros((1, 7))
        all_boxes_dict = {
            list(self.dest_to_dist.values())[x]: []
            for x in range(len(self.dest_to_dist))
        }
        all_box_ids = {
            x: list(self.box_dict.keys())[x - 1]
            for x in range(1, len(self.box_dict) + 1)
        }
        i = 1  # start box type
        for box_vals in self.box_dict.values():
            n_boxes = int(box_vals[3])
            box_vals[3] = i  # including box type (i) in the array
            temp_box = np.array([box_vals]) * (np.ones((n_boxes, 7)))
            _ = np.concatenate([_, temp_box], axis=0)
            i += 1
        _ = _[1:].astype(np.int64)
        for key in all_boxes_dict.keys():
            all_boxes_dict[key] = _[_[:, 5] == key]
        return all_boxes_dict, all_box_ids

    def volume(self, dim):
        """
        Calculates box volume from box dimensions using V = length*width*height
        Parameters
        ----------
        dim : 1D array in the format of [l b h]

        Returns
        -------
        Volume of the box in the format Vx1E6
        If the input dimensions are in cms, volume returned will be in m3
        """
        return np.cumprod(dim / 100)[-1]

    def volume_pvts(self, corner_pts):
        """
        Calculates box volume from its 8 corner points
        Parameters
        ----------
        corner_pts : 2D np array of shape (8,3). every row represents one point
        with [x y z] co-ordinates format. The order of the points (rows in the array)
        need to follow the box drawing convention.

        Returns
        -------
        Volume of the box in the format Vx1E6
        If the input dimensions are in cms, volume returned will be in m3
        """
        # Generate l, b, h from global co-ordinates
        length = corner_pts[1][0] - corner_pts[0][0]
        breadth = corner_pts[2][1] - corner_pts[0][1]
        height = corner_pts[4][2] - corner_pts[0][2]
        return length * breadth * height / 1e6


    def create_box(self, local_origin, box_dims):
        """
        Packs a new box into the truck.
        Plaicng is done by occupying 8 corner points that are inside the truck volume.

        Parameters
        ----------
        local origin : list of X,Y,Z co-ordinates [x, y, z] of the point where
        corner-0 of the box is aligned with
        box_dims : list of box dimensions in the format [length, width, height]

        Returns
        -------
        List of co-ordinates of the 8 corner points of the newly place box
        List has 8 items with each items being a list with format [x, y, z]
        """
        # Global co-ordinates of the placed box = Local Origin + Box Dimensions
        c_x, c_y, c_z = local_origin[0], local_origin[1], local_origin[2]
        l, b, h = box_dims[0], box_dims[1], box_dims[2]
        corner_pts = [
            [c_x, c_y, c_z],
            [c_x + l, c_y, c_z],
            [c_x, c_y + b, c_z],
            [c_x + l, c_y + b, c_z],
            [c_x, c_y, c_z + h],
            [c_x + l, c_y, c_z + h],
            [c_x, c_y + b, c_z + h],
            [c_x + l, c_y + b, c_z + h],
        ]
        return corner_pts

    def rec_intersect_check(self, packed_items, temp_box_corners):
        """
        Checks whether any of the boxes in a dictionary of boxes intersects
        the temporariliy placed box or not.
        Intersection is checked by comparing the relative position of 1st and
        8th corner points of both the boxes.

        Parameters
        ----------
        packed_items :  Dictionary of packed boxes in the current truck.
                        Value of an items is a list.
                        First element of this list is a list of 8 corner points
                        of the corresponding packed box
        temp_box_corners:   List of 8 corner points of the temporariliy placed box.

        Returns
        -------
        True if the temporarily placed box is intersecting with at least one of the
        already packed boxes
        False otherwise
        """
        # For chcking intersection, check whether Corner-0 or Corner-7 of the
        # temporary box is inside an already packed box
        for pk_key in packed_items.keys():
            if (packed_items[pk_key][0][7][0] > temp_box_corners[0][0]
                and packed_items[pk_key][0][7][1] > temp_box_corners[0][1]
                and packed_items[pk_key][0][7][2] > temp_box_corners[0][2]
                and packed_items[pk_key][0][0][0] < temp_box_corners[7][0]
                and packed_items[pk_key][0][0][1] < temp_box_corners[7][1]
                and packed_items[pk_key][0][0][2] < temp_box_corners[7][2]
            ):
                return True
        return False


    def xy_intersect(self, temp_box_corners, non_stackables):
        """
        Checks whether a given box is placed above any of the non-stackable boxes.

        Parameters
        ----------
        temp_box_corners    : List of 8 corner points of the box to be placed
        non_stackable_boxes : Dictionary of non-stackable boxes
                            Every items is a non-stackable box.
                            Value is a list with first element being a list of 8 corner points
                            of the correspnding non-stackable box

        Returns
        -------
        True if the placed box is above a non-stackable box fully or partly
        False otherwise
        """
        # Find all the non-stackable boxes : Equate the top surface z-cordinate of
        # non-stackable boxes with the bottom surface z-cordinate of temp box
        non_stackables_2 = {
            u: non_stackables[u]
            for u in non_stackables.keys()
            if non_stackables[u][0][4][2] == temp_box_corners[0][2]
        }

        l1 = np.abs(temp_box_corners[1][0] - temp_box_corners[0][0])
        b1 = np.abs(temp_box_corners[2][1] - temp_box_corners[0][1])
        center_1 = np.mean(temp_box_corners, axis=0)

        # Check XY intersection for all the non-stackable boxes iterated above
        # Follows the same logic of checking volumetric intersection in the
        # previous function, but in 2D space.
        for ns_box_key in non_stackables_2.keys():
            center_2 = np.mean(non_stackables_2[ns_box_key][0], axis=0)
            c_to_c = np.abs(center_1 - center_2)

            l2 = np.abs(
                non_stackables_2[ns_box_key][0][1][0]
                - non_stackables_2[ns_box_key][0][0][0]
            )
            b2 = np.abs(
                non_stackables_2[ns_box_key][0][2][1]
                - non_stackables_2[ns_box_key][0][0][1]
            )
            if np.greater((l1 + l2) / 2, c_to_c[0]) & np.greater(
                (b1 + b2) / 2, c_to_c[1]
            ):
                return True
        return False


    def base_area(self, temp_box_corners, packed_boxes):
        """
        Calculates fraction of the base area of a box supported by other boxes
        below it in percentage. With the assumption that the boxes are of uniform
        density, base area estimation will give a sense of stability

        Parameters
        ----------
        temporary_box_corners : Box whose supported base area (%) is to be calculated
        packed_boxes : Dictionary of all the packed boxes (in standard form of packed_items)

        Returns
        -------
        Percentage of the total base area of a box supported by other boxes below it.
        """
        # Return 100% when the box is on the truck floor
        if temp_box_corners[0][2] == 0:
            return 100.0

        # Find all the packed boxes that are packed on the same Z-level
        # Equate top surface z-cord of packed boxes with the bottom surface
        # z-cordinate of the temporary box
        packed_same_level = {
            u: packed_boxes[u]
            for u in packed_boxes.keys()
            if packed_boxes[u][0][4][2] == temp_box_corners[0][2]
        }
        tot_base_area = 0
        x1, x2 = temp_box_corners[0][0], temp_box_corners[3][0]
        y1, y2 = temp_box_corners[0][1], temp_box_corners[3][1]

        for box_key in packed_same_level.keys():
            x3, x4 = (
                packed_same_level[box_key][0][0][0],
                packed_same_level[box_key][0][3][0],
            )
            y3, y4 = (
                packed_same_level[box_key][0][0][1],
                packed_same_level[box_key][0][3][1],
            )

            xmax1, xmax2, xmin1, xmin2 = (
                max(x1, x2),
                max(x3, x4),
                min(x1, x2),
                min(x3, x4),
            )
            ymax1, ymax2, ymin1, ymin2 = (
                max(y1, y2),
                max(y3, y4),
                min(y1, y2),
                min(y3, y4),
            )

            if xmax1 > xmin2 and xmax2 > xmin1 and ymax1 > ymin2 and ymax2 > ymin1:
                eqv_x = min(xmax1, xmax2) - max(xmin1, xmin2)
                eqv_y = min(ymax1, ymax2) - max(ymin1, ymin2)
                tot_base_area += eqv_x * eqv_y

        act_base_area = np.abs(x1 - x2) * np.abs(y1 - y2)
        return tot_base_area * 100 / act_base_area


    def is_unloadable(self, packed_boxes, temp_box_corners, dest):
        """
        Checks whether a box can be unloaded without any other boxes
        obstructing it. Boxes going to later destinations shoould not be
        directly in front of the temporary box.

        Parameters
        ----------
        packed_boxes : Dictionary of all the packed boxes (in standard form of packed_items)
        temporary_box_corners : Box whose unloadability needs to be calculated
                                List of 8 corner points with [x,y,z] co-ordinates
        dest : Code of the destination to which the box has been assigned to

        Returns
        -------
        True if the box is unloadable
        False otherwise
        """

        packed_boxes_mod = {k: v for k, v in packed_boxes.items() if v[3] != dest}

        # Check if there is any box blocking the front side (YZ plane) the temporary box
        for pk_key in packed_boxes_mod.keys():
            if (
                packed_boxes_mod[pk_key][0][0][1] >= temp_box_corners[7][1]
                and packed_boxes_mod[pk_key][0][7][0] > temp_box_corners[0][0]
                and packed_boxes_mod[pk_key][0][7][2] > temp_box_corners[0][2]
            ):
                return False
        return True

    def is_fit_inside(self, temp_box_corners, truck_size):
        """
        Checks whether a box can be fit inside the truck without
        the box dimensions exceeding the truck walls.

        Parameters
        ----------
        temporary_box_corners : Box whose dimension fit needs to be calculated.
                                List of 8 corners, each corner being [x,y,z]
        truck_size : List of truck dimensions : [L, B, H]

        Returns
        -------
        True if the box is perfectly fit inside the truck
        False othewise
        """
        if (
            temp_box_corners[7][0] > truck_size[0]
            or temp_box_corners[7][1] > truck_size[1]
            or temp_box_corners[7][2] > truck_size[2]
        ):
            return False
        return True

    def place_the_box(self, truck_size, all_boxes_dict):
        """
        Fills the given truck with all the unpacked boxes available.
        Parameters
        ----------
        truck_size : List containing truck size params in the format [length, width, height]
        all_boxes_dict: Dictionary with
                        keys   - destination code
                        values - array containing the box information
                                 in the order [length  width  height  box_type  sorting_parameter]

        Procedure
        ---------
        For every destination and every unpacked box, iterate through the packed box
        corners until the point where
        the unpacked box can be successfully places. Successful placement of a box need to
        adhere to all the constraints such as dimensional and volumetric constraints.

        Returns
        -------
        packed_items : A dictionary with box names as the keys and list of 8 corner points as values
        updated unpacked_items : Updated list of unpacked boxes after removing the boxes that
                                that have been places in the truck successfully
        res_vol : Residual volume in the truck - empty space not occupied by any of the boxes
        """
        total_truck_vol = self.volume(truck_size)
        # Chack the contents of the input
        if len(all_boxes_dict) == 0:
            return "Invalid box quantity"
        packed_items, non_stackables = {}, {}
        all_pvts = [[0, 0, 0]]
        total_box_vol = 0
        logger.debug("Started packing current Truck")
        # Outer most loop going through every destination of the input
        # print(all_boxes_dict)
        for dest_key in sorted(all_boxes_dict.keys(), reverse=True):
            # print(dest_key)
            unpacked_items = all_boxes_dict[dest_key]
            used_unp_indx = []
            # Next-in-line loop going through every unpacked box in the current destination
            for unp_indx in range(len(unpacked_items)):
                temp_total_box_vol = total_box_vol + self.volume(
                    unpacked_items[unp_indx, :3]
                )
                # Check if box volume exceeds the residual truck volume
                if total_truck_vol < temp_total_box_vol:
                    logger.debug("First box exceeded the truck dimensions")
                    break

                # all_pvts.sort(key=lambda x: (x[1], x[0], x[2]))
                # sort the pivot points
                srt_grd = [x for x in all_pvts if x[2] == 0]
                srt_abv = [x for x in all_pvts if x[2] != 0]


                if self.load_pattern == "Side" or self.load_pattern == "Default":
                    srt_grd.sort(key=lambda x: (x[1], x[0], x[2]))
                    srt_abv.sort(key=lambda x: (x[2], x[1], x[0]))
                    srt_abv.extend(srt_grd)
                    all_pvts = srt_abv

                elif self.load_pattern == 'Back' or self.load_pattern == 'Rear Loading':
                    srt_grd.sort(key=lambda x: (x[0], x[2], x[1]))
                    srt_abv.sort(key=lambda x: (x[2], x[0], x[1]))
                    srt_abv.extend(srt_grd)
                    all_pvts = srt_abv

                elif self.load_pattern == 'Uniform Dist.':
                    srt_grd.sort(key=lambda x: (x[1], x[0], x[2]))
                    srt_abv.sort(key=lambda x: (x[2], x[1], x[0]))
                    srt_grd.extend(srt_abv)
                    all_pvts = srt_grd

                # Go through every available pivot point to check if box can be placed there
                # If not, move to next point. If successful, break out of the loop.
                for pvt in all_pvts:
                    unp_box_corners = self.create_box(
                        pvt, unpacked_items[unp_indx, :3]
                    )

                    this_dest = unpacked_items[unp_indx, 5]
                    # Perform all the constraint checks to make sure that box is packed
                    # perfectly inside the truck
                    if (
                        self.is_fit_inside(unp_box_corners, truck_size)
                        and self.is_unloadable(
                            packed_items, unp_box_corners, this_dest
                        )
                        and not self.xy_intersect(unp_box_corners, non_stackables)
                        and not self.rec_intersect_check(
                            packed_items, unp_box_corners
                        )
                        and self.base_area(unp_box_corners, packed_items)
                        >= self.base_area_thr
                    ):
                        # Successful packing. Update the packed_items dictionary
                        curr_box = f"box-{len(packed_items)+1}"
                        packed_items[curr_box] = [
                            unp_box_corners,
                            unpacked_items[unp_indx, 3],
                            self.base_area(unp_box_corners, packed_items),
                            unpacked_items[unp_indx, 5],
                            unpacked_items[unp_indx, 4],
                        ]
                        # logger.debug(
                        #     f"Box : {unpacked_items[unp_indx, :3]} packed at {pvt}"
                        # )
                        # Update the non-stackable dictionary if current box is not stackable
                        if unpacked_items[unp_indx, 6] == 1:
                            non_stackables[
                                f"box-{len(non_stackables)+1}"
                            ] = packed_items[curr_box]
                        # Update the available pivot points so as to not use them again
                        all_pvts.extend(packed_items[curr_box][0][1:])
                        i_1 = all_pvts.index(pvt)
                        del all_pvts[i_1]
                        # Create an artificial pivot point so that packing is not
                        # aborted due to insufficiency in available pivot points
                        y_max = max(all_pvts, key=itemgetter(1))[1]
                        if [0, y_max, 0] not in all_pvts:
                            all_pvts.append([0, y_max, 0])
                        # Update used pivot points
                        used_unp_indx.append(unp_indx)
                        total_box_vol = temp_total_box_vol
                        break
                    else:
                        pass

            # Update the unpacked items dictionary so as to pass it to the next truck
            all_boxes_dict[dest_key] = np.array(
                [
                    unpacked_items[x]
                    for x in range(len(unpacked_items))
                    if x not in used_unp_indx
                ]
            )
        res_vol = total_truck_vol - total_box_vol
        return (
            packed_items,
            all_boxes_dict,
            res_vol,
        )

    def make_dict_from_list(self, lst, box_type_to_code):
        """
        Takes in a list that contains information about what type of box has been placed
        in the truckand converts into a dictionary.
        This is for generating a better summary.

        Parameters
        ----------
        lst : List of box types in a truck which is done with packing
        box_type_to_code : Dictionary that converts box type (integer code) to the
                           actual box code.

        Returns
        -------
        ret_dict : A dictionary that has box type as the keys and no of items
                of that box type placed in the truck.
        """
        ret_dict = {}
        for i in range(len(lst)):
            ret_dict[box_type_to_code[lst[i]]] = lst.count(lst[i])
        return ret_dict

    def packing(self, init_unpacked_items, box_type_to_code):
        """
        Following for loop will iterate through all the input trucks.
        Iteration will stop as soon as the list of unpacked items become empty.
        Also creates two dictionaries.
        The first dictionary contains the information about all the trucks with
        packed_items, residual volume, box_types in it
        Second dictionary contains the information about unpacked items after
        the corresponding truck has been filled to its max.

        Parameters
        ----------
        init_unpacked_items : The very first state of the unpacked items dictionary
        before loading the first truck

        box_type_to_code : Dictionary that converts box type (integer code) to the
                           actual box code.

        Returns
        -------
        per_truck_packed : Dictionary with
                           key : truck name (TRUCK-1 for example)
                           value : [packed_items, list of unique boxes packed,
                                    residual_volume of the truck]

        """
        a = perf_counter()  # Start the clock

        per_truck_packed = {}
        per_truck_unpacked = {}
        # Loop for every truck input
        for i in range(1, len(self.truck_size_dict) + 1):
            truck_name = f"TRUCK-{i}"
            # First truck needs explicit input of unpacked items dictionary.
            # Remaining trucks will eventually get from the packing of previous truck.
            if i == 1:
                try:
                    ret_values = self.place_the_box(
                        self.truck_size_dict[f"TRUCK-{i}"], init_unpacked_items
                    )
                    if ret_values == "Invalid box quantity":
                        raise CustomException(
                            "Packing aborted : There are no boxes to pack"
                        )
                except CustomException as error:
                    logger.info(error.message)
                else:
                    pkd, un_pkd, res_vol = ret_values

            else:
                if len(per_truck_unpacked[f"TRUCK-{i-1}"]) == 0:
                    logger.info("No more boxes left to pack")
                elif self.truck_size_dict[truck_name] is not None:
                    pkd, un_pkd, res_vol = self.place_the_box(
                        self.truck_size_dict[f"TRUCK-{i}"],
                        per_truck_unpacked[f"TRUCK-{i-1}"],
                    )

                else:
                    logger.info(
                        f"""There are {len(per_truck_unpacked[f"TRUCK-{i-1}"])}
                         but no more trucks remaining"""
                    )

            per_truck_packed[truck_name] = [
                pkd,
                self.make_dict_from_list(
                    [pkd[f"box-{x}"][1] for x in range(1, len(pkd) + 1)],
                    box_type_to_code,
                ),
                res_vol,
            ]
            logger.debug(
                f"\n>>>>  packed {len(per_truck_packed[truck_name][0])} items "
            )
            per_truck_unpacked[truck_name] = un_pkd
            logger.debug(
                f"\n>>>>  There are {len(per_truck_unpacked[truck_name])} items \
                not packed into {truck_name}"
            )

        b = perf_counter()  # Stop the clock
        logger.debug(f"\n>>>>  Took {b-a:0.3f} seconds to complete packing")
        return per_truck_packed


    def summary(self, base_dict):
        """
        Prints the summary of packing for those trucks which contains at least one box in it.

        Parameters
        ----------
        base_dict : A dictionary with truck names as the keys.
                    Values are tuples with following format
                    (dict of packed_items in the truck,
                    dict of box type and no of that type of boxes in the truck,
                    residual volume of the truck - empty space)

        Returns
        -------
        Function is for printing summary, with no return
        """
        logger.debug("\n>>>>  Summary\n")
        for truck_indx in range(len(base_dict)):
            tr_name = f"TRUCK-{truck_indx+1}"
            logger.debug(
                f"\t>>  {tr_name}\n\
            Truck Size : {self.truck_size_dict[tr_name]}\n\
            Total Vol. : {self.volume(self.truck_size_dict[tr_name])}\n\
            Total Boxes Packed : {len(base_dict[tr_name][0])}\n\
            Residual Vol. : {base_dict[tr_name][2]:0.3f} cu.metres\n\n\
            Boxes Packed : "
            )
            #         for b_key in base_dict[tr_name][1].keys():
            #             print(f"\t\t{b_key} {box_types[b_key][:3]} to {box_types[b_key][5]} ---> \
            # {base_dict[tr_name][1][b_key]}")
            logger.debug("\n\n")
            return f"{base_dict[tr_name][2]:0.3f}", f"{len(base_dict[tr_name][0])}"

    def js_input_main(self, truck_packed, ftor_dict):
        """
        Writes files that contain packed state information of all the trucks. These files
        are to be read and returned to the frontend.

        Parameters
        ----------
        truck_packed : Dictionary with
                       key : truck name (TRUCK-1 for example)
                       value : [packed_items, list of unique boxes packed,
                                residual_volume of the truck]
        ftor_dict : Dictionary that converts F-code (unique each box type) to
                    original shipment code


        """
        # Get the list of destinations and color codes to display in the UI
        dest_codes = {self.dest_to_dist[u]: u for u in self.dest_to_dist.keys()}

        color_list = [
            "#bada55",
            "#7fe5f0",
            "#ff0000",
            "#ff80ed",
            "#696969",
            "#133337",
            "#065535",
            "#5ac18e",
            "#f7347a",
            "#ffd700",
        ]

        dest_dictt = {
            dest_codes[x]: [color_list[x - 1], self.dest_to_dist[dest_codes[x]]]
                            for x in range(1, len(dest_codes) + 1)
        }
        # dest_dictt = {
            # dest_codes[x]: [color_list[x - 1] , x] for x in range(1, len(dest_codes) + 1)
        # }

        new_dict = {}
        # Write the packed boxes information to be passed onto the UI
        for tr_num in range(1, len(truck_packed) + 1):
            ret_list = []
            tr_name = f"TRUCK-{tr_num}"
            for key in truck_packed[tr_name][0].keys():
                # print(key)
                box_corners = truck_packed[tr_name][0][key][0]
                r_code = f"F-{truck_packed[tr_name][0][key][1]}"
                dest = truck_packed[tr_name][0][key][3]
                weight = truck_packed[tr_name][0][key][4]
                l = box_corners[7][0] - box_corners[0][0]
                b = box_corners[7][1] - box_corners[0][1]
                h = box_corners[7][2] - box_corners[0][2]
                ret_list.append(
                    [
                        [l, h, b],
                        [box_corners[0][0], box_corners[0][2], box_corners[0][1]],
                        color_list[dest - 1],
                        ftor_dict[r_code][0],
                        weight,
                        dest
                    ]
                )
            new_dict[tr_name] = ret_list
        with open("ui_input/truck_to_js.txt", "w") as f:
            f.write(str(new_dict))
        with open("ui_input/dest_colors.txt", "w") as f:
            f.write(str(dest_dictt))


def unpacked_js(df):
    '''Returns the box ids which are not packed into the truck after loading

    Input - Original dataframe
    Returns - box ids list
    '''
    original = df['Box ID']
    with open("ui_input/truck_to_js.txt", "r") as f:
        packed_boxes = literal_eval(f.readlines()[0])
    m = list(packed_boxes.values())
    packed_list = list(itertools.chain(*m))
    box_id_packed = [item[3] for item in packed_list]
    res = list((Counter(original) - Counter(box_id_packed)).elements())
    if res or len(original) == len(box_id_packed):
        return res
    return original.values.tolist()

def main_func(df1, truck_size_dict, dest_order, load_pattern):
    """
    Main function to create a packing heuristics object and execute the
    current packing request
    Calls required functions to create files that contain the results of packing.

    Parameters
    ----------
    truck_size_dict : Dicitonary of all the available trucks with key as truck
                      name value as its dimensions
    dest_order : Dictionary with key as the destination name and
                      value as an integer representing the order of unloading

    """
    # Get the input for the main function from the export_data_main class
    try:
        ret_values = export_data(df1, truck_size_dict, dest_order, load_pattern)
        if ret_values == "Exception Occured":
            raise CustomException("Error Occured during data export from excel")

    except CustomException as error:
        logger.debug(error.message)
    else:
        # Create the instance of the packing heuristics class
        box_types, f_to_r_code, dest_to_dist, truck_size_dict = ret_values

        heuristic = PackingHeuristic(truck_size_dict, box_types, dest_to_dist, 100, load_pattern)

        # Create the initial unpacked items dictionary
        all_box_dict, box_type_to_code = heuristic.create_box_list()
        init_unpacked_items = all_box_dict
        n_init_unpacked_items = sum([len(v) for v in all_box_dict.values()])

        # Perform the packing initiated from the initial unpacked items dictionary
        boxes_packed_per_truck = heuristic.packing(
            init_unpacked_items, box_type_to_code
        )
        res_vol = heuristic.summary(boxes_packed_per_truck)


        # Write the results to corresponding text files
        heuristic.js_input_main(boxes_packed_per_truck, f_to_r_code)
        with open("ui_input/truck_size.txt", "w") as f:
            truck_size_dict = {k: v.tolist() for k, v in truck_size_dict.items()}
            f.write(str(truck_size_dict) + "\n")
            f.write(str(n_init_unpacked_items) + "\n")
            f.write(str(res_vol) + "\n")

        res1 = unpacked_js(df1)
        with open("ui_input/unpacked.txt", "w") as f:
            f.write(str(res1) + "\n")