#! /usr/bin/env python

import amygda
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image",help="the path to the image")
parser.add_argument("--growth_pixel_threshold",default=130,type=int,help="the pixel threshold, below which a pixel is considered to be growth (0-255)")
parser.add_argument("--growth_percentage",default=3,type=float,help="if the central measured region in a well has more than this percentage of pixels labelled as growing, then the well is classified as growth.")
parser.add_argument("--measured_region",default=0.4,type=float,help="the size of the central measured region, as a decimal proportion of the whole well.")
parser.add_argument("--sensitivity",default=4,type=float,help="if the average growth in the control wells is more than (sensitivity x growth_percentage), then consider growth down to this sensitivity ")
parser.add_argument("--file_ending",default="-raw",type=str,help="the ending of the input file that is stripped. Default is '-raw' ")
options = parser.parse_args()

# parse the path to the input image and work out its stem of
cols=options.image.split('/')
image_name=cols[-1].split(options.file_ending)[0]
image_path=cols[0]+"/"
for i in cols[1:-1]:
    image_path+=i
    image_path+="/"

# create a new the measurement
plate=amygda.PlateMeasurement(image_path,new=True,categories={'PlateImage':image_name})

# create the path for the output images
plate_stem=plate.abspath+plate.image_name

# define some colours for the output
# the below are a contrasting set of colours taken from ColorBrewer 2
# http://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=7
# you are, of course, free to make up your own!
# Note these are (B,G,R) not (R,G,B)
pink=(138,41,231)
yellow=(2,171,230)
teal=(119,158,27)
green=(36,166,102)
black=(0,0,0)
white=(255,255,255)

# load the raw image
plate.load_image(plate_stem+"-raw.png")

# record that this image exists
plate.categories['IM_IMAGE_DOWNLOADED']=True

# apply a mean shift filter to smooth the background colours
plate.mean_shift_filter()

# apply the local histogram equalisation method to improve contrast
plate.equalise_histograms_locally()

# save the filtered image
plate.save_image(plate_stem+"-filtered.png")

# record that this image has been filtered
plate.categories['IM_IMAGE_FILTERED']=True

# load in the photo of the plate
plate.load_image(plate_stem+"-filtered.png")

# attempt to segment the wells
if plate.identify_wells():

    plate.categories['IM_WELLS_IDENTIFIED']=True

     # measure growth
    plate.measure_growth(region=options.measured_region,threshold_pixel=options.growth_pixel_threshold,threshold_percentage=options.growth_percentage,sensitivity=options.sensitivity)

    # save the numpy arrays containing the growth, positions of wells etc to disc
    plate.save_arrays(plate_stem+"-arrays.npz")

    # draw circles around the wells
    plate.annotate_well_circumference(color=black,linewidth=1)

    # write the drug and concentration
    plate.annotate_well_drugs_concs(color=black,fontsize=0.5)

    # add squares where the algorithm has detected growth
    plate.annotate_well_analysed_region(growth_color=yellow,region=options.measured_region,thickness=2)

    # save the final image with wells with identified growth marked by red squares
    plate.save_image(plate_stem+"-processed.png")

    # write the MICs to a simple plaintext file (they are stored in the JSON file but this is harder to read)
    plate.write_mics(plate_stem+"-mics.txt")
