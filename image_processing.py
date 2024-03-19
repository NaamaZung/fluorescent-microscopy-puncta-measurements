import cv2
import numpy as np
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import IO


# Example Usage:
# img = cv2.imread("images/LD_Control.tif")
# identify_cell_circles(img, show_plots=True)
# puncta_segmentation(img, show_plots=True)
# calculate_puncta_per_cell_in_image("images/LD_Control.tif", "output.csv")

# Reads microscopy image from {filename} and calculates amount of puncta per cell, storing
# results in {output_filename}. Optional {show_plots} to display images of process.
def calculate_puncta_per_cell_in_image(filename, output_filename, show_plots=False):
    img = cv2.imread(filename)
    # find cells
    circles = identify_cell_circles(img, show_plots)
    # find puncta
    labels = puncta_segmentation(img, show_plots)

    puncta_in_cell = IO.create_dataframe()
    circles_mask = np.zeros(labels.shape, np.uint8)
    circle_index = 0
    for (center_x, center_y, r) in circles:
        puncta_in_circle = set()
        # loop through pixels in square around circle
        for x in range(max(center_x - r, 0), min(center_x + r + 1, labels.shape[0])):
            for y in range(max(center_y - r, 0), min((center_y + r + 1), labels.shape[1])):
                # if in circle - add label to puncta_in_circle
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= r ** 2:
                    puncta_in_circle.add(labels[x, y].item())
        puncta_in_cell.loc[circle_index] = [center_x, center_y, len(puncta_in_circle) - 1]  # remove 1 for zero
        circle_index += 1
        cv2.circle(circles_mask, (center_x, center_y), r, 255, -1)

    # output to csv
    IO.dataframe_to_csv(puncta_in_cell, output_filename)

    if show_plots:
        circles_mask = cv2.bitwise_and(labels.astype(np.uint8), circles_mask)
        cv2.imshow('Circles', circles_mask)
        cv2.imshow('Original', img)
        cv2.waitKey(0)


# Creates a label per pixel corrosponding to background (0 value) or positive for individual detected puncta
def puncta_segmentation(img, show_plots=False):
    # Define the magenta color range
    lower_magenta = np.array([140, 0, 100])
    upper_magenta = np.array([180, 255, 255])

    # Create a mask for magenta regions
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_magenta, upper_magenta)

    kernel = np.ones((3, 3), np.uint8)

    # Apply mask to the original image
    magenta_regions = cv2.bitwise_and(img, img, mask=mask)
    if show_plots:
        cv2.imshow('Magenta Regions', magenta_regions)

    # Remove particles with erode
    magenta_regions = cv2.erode(magenta_regions, kernel, iterations=1)
    if show_plots:
        cv2.imshow('Magenta Regions eroded', magenta_regions)

    gray_magenta_regions = cv2.cvtColor(magenta_regions, cv2.COLOR_BGR2GRAY)
    if show_plots:
        cv2.imshow('gray_magenta_regions', gray_magenta_regions)

    ret1, bw_magenta_regions = cv2.threshold(gray_magenta_regions, 50, 255, cv2.THRESH_BINARY)
    if show_plots:
        cv2.imshow('bw_magenta_regions', bw_magenta_regions)

    # Watershed Segmentation- https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
    distance = ndi.distance_transform_edt(bw_magenta_regions)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=bw_magenta_regions)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=bw_magenta_regions)

    if show_plots:
        cv2.imshow('Magenta labels', labels.astype(np.uint8))
        cv2.waitKey(0)

    return labels


# Creates a list of circles (center_x,center_y,radius) corresponding to yeast cells detected in image
def identify_cell_circles(img, show_plots=False):
    inverted = np.invert(img)
    inverted = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    ret1, thresh = cv2.threshold(inverted, 215, 255, cv2.THRESH_BINARY)
    if show_plots:
        cv2.imshow('thresh', thresh)

    kernel = np.ones((3, 3), np.uint8)

    # fill in holes with dilate
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    if show_plots:
        cv2.imshow('sure_bg', thresh)

    # remove particles with erode
    thresh = cv2.erode(thresh, kernel, iterations=2)
    if show_plots:
        cv2.imshow('eroded', thresh)

    circles_graph = thresh.copy()

    # param1,param2 selected manually
    # https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
    circles = cv2.HoughCircles(circles_graph, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=60, param2=15, minRadius=5,
                               maxRadius=50)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

    if show_plots:
        circles_graph_color = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2RGB)
        circles_mask = np.zeros_like(circles_graph_color)
        for (x, y, r) in circles:
            cv2.circle(circles_graph_color, (x, y), r, (0, 255, 0), 2)
            cv2.circle(circles_mask, (x, y), r, (255, 255, 255), -1)
            cv2.rectangle(circles_graph_color, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)

        cv2.imshow('circles_graph', circles_graph_color)
        cv2.imshow('circles_mask', circles_mask)
        cv2.waitKey(0)

    return circles
