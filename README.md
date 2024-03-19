# fluorescent-microscopy-puncta-measurements
An open source tool for measurements of foci in microscopy images. 

The goal of this project is an easy-to-use software that will receive as an input a TIF image that contain several cells and will be able to:
* Mark the cells
* In each marked cell, calculate the number of puncta (using image analysis) and export this output to a spreadsheet file. 

To run (code and tests):
* Use the command `pip install -r requirements.txt` in your terminal
* run test_file.py

To generate results from code in project:
```
from image_processing import calculate_puncta_per_cell_in_image

# <image filename>,<output csv filename>
calculate_puncta_per_cell_in_image("images/LD_Control.tif", "output.csv")
```
by Naama Zung https://naamazung.github.io/