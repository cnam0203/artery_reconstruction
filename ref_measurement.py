import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def max_stable_mask(a, pos, ratio_threshold, distance, interval_size, dif_thresh, side): # thresh controls noise
    pass_steps = int(distance/interval_size)
    if pass_steps == 0:
        pass_steps = 1
        
    left_mask = None
    right_mask = None
    left_radius = None
    right_radius = None
    mean_radius = None
    #Left
    # if pos - 2*pass_steps >= 0 and side in [0, 1]:
    #     left_arr = a[pos - 2*pass_steps : pos - pass_steps]
    #     left_values, left_mask = side_stable_mask(left_arr, ratio_threshold, dif_thresh)
    #     left_radius = np.mean(left_values[left_mask == 1])

    # if pos + 2*pass_steps < a.shape[0] and side in [0, 2] :
    #     right_arr = a[pos + pass_steps : pos + 2*pass_steps]
    #     right_values, right_mask = side_stable_mask(right_arr, ratio_threshold, dif_thresh)
    #     right_radius = np.mean(right_values[right_mask == 1])

    if side in [0, 1]:
        if pos - pass_steps - 4 >= 0:
            left_arr = a[pos - pass_steps - 4 : pos - pass_steps]
            left_radius = np.mean(left_arr)
        if pos - pass_steps - 4 >= 0:
            left_arr = a[0 : pos - pass_steps + 1]
            left_radius = np.mean(left_arr)
        else:
            left_radius = a[0]

    if side in [0, 2]:
        if pos + pass_steps + 4 < a.shape[0]:
            right_arr = a[pos + pass_steps : pos + pass_steps + 5]
            right_radius = np.mean(right_arr)
        elif pos + pass_steps < a.shape[0]:
            right_arr = a[pos + pass_steps : a.shape[0]]
            right_radius = np.mean(right_arr)
        else:
            right_radius = a[a.shape[0]-1]
    
    if left_radius == None and right_radius != None:
        mean_radius = right_radius
    elif left_radius != None and right_radius == None:
        mean_radius = left_radius
    elif left_radius != None and right_radius != None:
        mean_radius = (left_radius + right_radius)/2
    
    return mean_radius

def side_stable_mask(arr, ratio_threshold, dif_thresh):
    a = np.copy(arr)
    is_end = False
    loop_count = 0
    while not is_end and loop_count <= 50:
        loop_count += 1
        max_value = round(a.max(), 1)
        mask = np.r_[ False, np.abs(a - max_value) < dif_thresh, False]
        idx = np.flatnonzero(mask[1:] != mask[:-1])
        s0 = (idx[1::2] - idx[::2]).argmax()
        valid_mask = np.zeros(a.size, dtype=int) #Use dtype=bool for mask o/p
        valid_mask[idx[2*s0]:idx[2*s0+1]] = 1

        if np.argwhere(valid_mask == 1).shape[0] >= ratio_threshold*a.shape[0]:
            is_end = True
        else:
            second_max_value = round(np.max(a[a != max_value]), 1)
            
            if second_max_value >= max_value - dif_thresh:
                a[a == max_value] = second_max_value
            else:
                a[a == max_value] = 0
            
    return a, valid_mask

def check_truth(level, ratio, case):
    if case == -3 and level == 0:
        return -1
    elif case == -3 and level != 0:
        if ratio >= 0.15:
            return 1
        else:
            return 0
    elif case == -2 and level != 0:
        return -1
    elif case == -1:
        if level == 0 and ratio <= 0.25:
            return 1
        elif level != 0 and ratio >= 0.15:
            return 1
        else:
            return 0
    else:
        is_correct = 0
        
        if level == 0:
            if ratio <= 0.25:
                is_correct = 1
            else:
                is_correct = 0
        elif level == 1:
            if ratio >= 0.15 and ratio <= 0.55:
                is_correct = 1
            else:
                is_correct = 0
        elif level == 2:
            if ratio >= 0.45 and ratio <= 0.75:
                is_correct = 1
            else:
                is_correct = 0
        elif level == 3:
            if ratio >= 0.65:
                is_correct = 1
            else:
                is_correct = 0
        else:
            if ratio >= 0.95:
                is_correct = 1
            else:
                is_correct = 0
        
        if case in [-3, -1, -2] or level == case:
            return is_correct
        else:
            return -1

# def check_truth(level, ratio, case):
#     if case == -3 and level == 0:
#         return -1
#     elif case == -3 and level != 0:
#         if ratio >= 0.2:
#             return 1
#         else:
#             return 0
#     elif case == -2 and level != 0:
#         return -1
#     else:
#         is_correct = 0
        
#         if level == 0:
#             if ratio <= 0.19:
#                 is_correct = 1
#             else:
#                 is_correct = 0
#         elif level == 1:
#             if ratio >= 0.2 and ratio <= 0.49:
#                 is_correct = 1
#             else:
#                 is_correct = 0
#         elif level == 2:
#             if ratio >= 0.5 and ratio <= 0.69:
#                 is_correct = 1
#             else:
#                 is_correct = 0
#         elif level == 3:
#             if ratio >= 0.7 and ratio <= 0.99:
#                 is_correct = 1
#             else:
#                 is_correct = 0
#         else:
#             if ratio >= 0.99:
#                 is_correct = 1
#             else:
#                 is_correct = 0
        
#         if case in [-3, -1, -2] or level == case:
#             return is_correct
#         else:
#             return -1

def find_stenosis_ratios(diameter_1, diameter_2, side, length_percentage=0.1):
    ref_min_distances = []
    ref_avg_distances = []
    avg_distances = diameter_2
    ratio_threshold = 0.1
    distance_threshold = 0.5
    distance = length_percentage*len(diameter_2)*distance_threshold
    interval_size = distance_threshold

    for i in range(len(diameter_1)):
        dif_thresh = 0.5*avg_distances[i]

        avg_distance = max_stable_mask(np.array(avg_distances), i, ratio_threshold, distance, interval_size, dif_thresh, side)                        
        ref_avg_distances.append(avg_distance)
    
    is_stop = False 
    count = 0
    while not is_stop or count < 1000:
        count += 1
        is_stop = True
        for idx, ring in enumerate(avg_distances):
            neighbor_avg_distances = []

            if ref_avg_distances[idx] == None:
                if idx > 0 and ref_avg_distances[idx-1] != None:
                    neighbor_avg_distances.append(ref_avg_distances[idx-1])
                if idx < (len(avg_distances) - 1) and ref_avg_distances[idx+1] != None:
                    neighbor_avg_distances.append(ref_avg_distances[idx+1])

                if len(neighbor_avg_distances):
                    ref_avg_distances[idx] = np.mean(np.array(neighbor_avg_distances))

        undefined_ranges = [distance for distance in ref_avg_distances if distance == 0 or distance is None]
        if len(undefined_ranges):
            is_stop = False

    stenosis_ratio_avg = np.maximum(1-np.array(diameter_1)/np.array(ref_avg_distances), 0)

    return stenosis_ratio_avg

def find_stenosis_ratio(start_idx, end_idx, diameter_1, diameter_2, side, length_percentage=0.1, is_peak=True):
    smooth_min = gaussian_filter1d(diameter_1, sigma=2)
    smooth_avg = gaussian_filter1d(diameter_2, sigma=2)

    ratios = find_stenosis_ratios(smooth_min, smooth_avg, side, length_percentage)
    if not is_peak:
        result = np.max(ratios[start_idx:end_idx], axis=0)
    else:
        sorted_indices = np.argsort(ratios[start_idx:end_idx])[::-1]
        top_ten_indices = sorted_indices[:10]
        peaks, _ = find_peaks(-smooth_min)

        result = 0
        for i in top_ten_indices:
            pos = start_idx + i
            for peak in peaks:
                if abs(peak-pos) <= 2:
                    result = ratios[start_idx:end_idx][i]
                    break
            
            if result != 0:
                break

    return round(result, 2)

def ratio_to_level_1(percent):
    if percent < 0.3:
        return '0'
    elif percent < 0.6:
        return '1'
    elif percent < 0.8:
        return '2'
    elif percent < 0.99:
        return '3'
    elif percent <= 1:
        return '4'
    else:
        return 'N/A'

def ratio_to_level(percent):
    if percent < 0.15:
        return 0
    elif percent < 0.26:
        return '0-1'
    elif percent < 0.45:
        return 1
    elif percent < 0.56:
        return '1-2'
    elif percent < 0.65:
        return 2
    elif percent < 0.76:
        return '2-3'
    elif percent < 0.95:
        return '3'
    elif percent == 1:
        return '3-4'
    else:
        return 'N/A'
    