#! /usr/bin/env python

import statefiles
import cv2, numpy
from datreant.core import Treant

class PlateMeasurement(Treant):

    """ The PlateMeasurement class is a Treant for storing and analysing a
    photograph of a 96 well plate.

    Args:
        new (True/False): whether this is a new PlateMeasurement (default is False)
        plate_image (str): path to the image file to analyse
        categories (dict): a dictionary containing meta-data for the image. Must contain the name of the image (imagename) as {'PlateImage':imagename}
        tags (str or list): strings containing tags to differentiate the Plates. Not required.
        well_dimensions (tuple): tuple of 2 integers defining the (rows, cols) of the plate. Default=(8,12)
    """

    _treanttype='PlateMeasurement'
    _backendclass = statefiles.PlateMeasurementFile

    def __init__(self, plate_image, new=False, categories=None, tags=None, well_dimensions=(8,12), configuration_path='plate-configuration/', plate_design='CRyPTIC1-V1'):

        Treant.__init__(self, plate_image, new=new, categories=categories, tags=tags)

        # store the image name provided in the categories
        self.image_name=self.categories['PlateImage']

        # store the (rows,cols) of the plate
        self.well_dimensions=well_dimensions

        # calculate and store the total number of wells
        self.number_of_wells = self.well_dimensions[0]*self.well_dimensions[1]

        # store the path and name of the plate
        self.configuration_path = configuration_path
        self.plate_design = plate_design

    def load_image(self,filename):
        """ Load and store the image and its dimensions, then initialise a series of arrays to record the results
        """

        self.image_path=filename

        # load the image as a 3-channel array
        # it needs to be colour for the mean shift filter to work
        self.image = cv2.imread(self.image_path)

        # set the downloaded flag to be True in case was missed during retrieval
        self.categories["IM_IMAGE_DOWNLOADED"]=True

        # this is a colour image
        self.image_colour=True

        # determine the dimensions of the image
        self.image_dimensions=self.image.shape

        # create numpy arrays to store details about the wells
        self.well_index = numpy.zeros(self.well_dimensions)
        self.well_radii = numpy.zeros(self.well_dimensions)
        self.well_centre = numpy.zeros(self.well_dimensions,dtype=(int,2))
        self.well_top_left = numpy.zeros(self.well_dimensions,dtype=(int,2))
        self.well_bottom_right = numpy.zeros(self.well_dimensions,dtype=(int,2))
        self.well_growth = numpy.zeros(self.well_dimensions,dtype=numpy.float64)
        self.well_drug_name = numpy.loadtxt(self.configuration_path+"/"+self.plate_design+"-drug-matrix.txt",delimiter=',',dtype=str)
        self.well_drug_conc = numpy.loadtxt(self.configuration_path+"/"+self.plate_design+"-conc-matrix.txt",delimiter=',',dtype=float)
        self.well_drug_dilution = numpy.loadtxt(self.configuration_path+"/"+self.plate_design+"-dilution-matrix.txt",delimiter=',',dtype=int)

        # identify the control wells
        self.well_positive_controls=[]
        for iy in range(0,self.well_dimensions[0]):
            for ix in range (0,self.well_dimensions[1]):
                if self.well_drug_conc[(iy,ix)]==0.0:
                    self.well_positive_controls.append((iy,ix))

        self.well_positive_controls_number=len(self.well_positive_controls)+1

        # create a list of the drug names
        self.drug_names=(numpy.unique(self.well_drug_name)).tolist()

    def save_arrays(self,filename):
        """ Save the numpy arrays with the well coordinates and growth etc in a single NPZ file.
        """

        # Ugly, but ensures that all the arrays keep their names when saved to the file.
        # Note that without compression the files are only 10Kb, which is << the size of the images
        # so there is no point compressing them. No difference in timing.
        numpy.savez(filename,   well_index=self.well_index,
                                well_radii=self.well_radii,
                                well_centre=self.well_centre,
                                well_top_left=self.well_top_left,
                                well_bottom_right=self.well_bottom_right,
                                well_growth=self.well_growth,
                                well_drug_name=self.well_drug_name,
                                well_drug_conc=self.well_drug_conc,
                                well_drug_dilution=self.well_drug_dilution,
                                threshold_pixel=self.threshold_pixel,
                                threshold_percentage=self.threshold_percentage,
                                sensitivity=self.sensitivity)

    def load_arrays(self,filename):
        """ Load the numpy arrays with the well coordinates and growth etc from a single NPZ file.
        """

        # Also ugly, but at least it works and is explicit
        npzfile=numpy.load(filename)
        self.well_index = npzfile['well_index']
        self.well_radii = npzfile['well_radii']
        self.well_centre = npzfile['well_centre']
        self.well_top_left = npzfile['well_top_left']
        self.well_bottom_right = npzfile['well_bottom_right']
        self.well_growth = npzfile['well_growth']
        self.well_drug_name = npzfile['well_drug_name']
        self.well_drug_conc = npzfile['well_drug_conc']
        self.well_drug_dilution = npzfile['well_drug_dilution']
        self.threshold_pixel=npzfile['threshold_pixel']
        self.threshold_percentage=npzfile['threshold_percentage']
        self.sensitivity=npzfile['sensitivity']

        # create a list of the drug names
        self.drug_names=(numpy.unique(self.well_drug_name)).tolist()

    def mean_shift_filter(self,spatial_radius=10,colour_radius=10):
        """ Apply a mean shift filter to produce a cleaner, more homogenous image.
        """

        # convert the 3 channel image to 1 channel
        if not self.image_colour:
            self._convert_image_to_colour()
            self.image_colour=True

        # apply the mean shift filter
        self.image=cv2.pyrMeanShiftFiltering(self.image,spatial_radius,colour_radius)

    def equalise_histograms(self):
        """ Apply a global histogram filter.

        This performs much worse than equalise_histograms_locally() and is only included for comparision
        """

        # convert the 3 channel image to 1 channel
        if self.image_colour:
            self._convert_image_to_grey()
            self.image_colour=False

        # equalise the image histogram globally (will not take account of the uneven lighting)
        self.image=cv2.equalizeHist(self.image)

    def equalise_histograms_locally(self):
        """ Apply a Contrast Limited Adaptive Histogram Equalization filter.

        Compared to a global histogram filter, this does a much better job of equalising the illumination across the plate.
        """

        # convert the 3 channel image to 1 channel
        if self.image_colour:
            self._convert_image_to_grey()
            self.image_colour=False

        # equalise the image histogram locally (will take account of the uneven lighting)
        # note that the tile grid matches the wells
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=self.well_dimensions)

        self.image = clahe.apply(self.image)

    def save_image(self,filename):
        """ Save the image to disc.
        """

        cv2.imwrite(filename,self.image)

    def _convert_image_to_colour(self):
        """ Convert the image to colour.
        """

        new_image=numpy.zeros(self.image.shape+(3,))
        for i in [0,1,2]:
            new_image[:,:,i]=self.image
        self.image=new_image

    def _convert_image_to_grey(self):
        """ Convert the image to greyscale
        """

        self.image=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def annotate_well_circumference(self,color=(0,0,0),linewidth=1):
        """ Draw circles around where AMyGDA has identified the wells on the plate.

        Args:
            color (B,G,R): Colour of the edge of the circle. Default is (0,0,0) which is black.
            linewidth (int): Width of the edge of the circle. Default is 1.
        """

        # convert the 1 channel image to 3 channel
        if not self.image_colour:
            self._convert_image_to_colour()
            self.image_colour=True

        for iy in range(0,self.well_dimensions[0]):
            for ix in range(0,self.well_dimensions[1]):

                centre=(int(self.well_centre[(iy,ix)][0]),int(self.well_centre[(iy,ix)][1]))
                radius=int(self.well_radii[(iy,ix)])
                cv2.circle(self.image,centre,radius,color,linewidth)

    def annotate_well_centres(self,color=(0,0,0),linewidth=2):
        """ Draw a small circle at the centre of each well.

        Args:
            color (B,G,R): Colour of the edge of the circle. Default is (0,0,0) which is black.
            linewidth (int): Width of the edge of the circle. Default is 2.
        """

        # convert the 1 channel image to 3 channel
        if not self.image_colour:
            self._convert_image_to_colour()
            self.image_colour=True

        for iy in range(0,self.well_dimensions[0]):
            for ix in range(0,self.well_dimensions[1]):

                centre=(int(self.well_centre[(iy,ix)][0]),int(self.well_centre[(iy,ix)][1]))
                radius=1
                cv2.circle(self.image,centre,radius,color,linewidth)

    def annotate_well_drugs_concs(self,color=(0,0,0),fontsize=0.4):
        """ Label each well with the concentration of drug it contains.

        Args:
            color (B,G,R): Colour of the writing. Default is (0,0,0) which is black.
            fontsize (float): Relative size of the writing. Default is 0.4.
        """

        # convert the 1 channel image to 3 channel
        if not self.image_colour:
            self._convert_image_to_colour()
            self.image_colour=True

        for iy in range(0,self.well_dimensions[0]):
            for ix in range(0,self.well_dimensions[1]):

                # label = "%02d" % self.well_index[(iy,ix)]
                label1 = "%s" % (self.well_drug_name[(iy,ix)])
                label2 = "%s" % (self.well_drug_conc[(iy,ix)])

                (a,b) = (int(self.well_centre[(iy,ix)][0]),int(self.well_centre[(iy,ix)][1]))

                cv2.putText(self.image, label1, (a-15,b-20), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0,0,0), 1)
                cv2.putText(self.image, label2, (a-15,b+30), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0,0,0), 1)


    def annotate_well_analysed_region(self,growth_color=(0,0,0),region=0.4,thickness=1): #,threshold_percentage=3,sensitivity=0):
        """ Draw coloured squares on the wells with detected bacterial growth.

        Args:
            growth_color (B,G,R): Colour of the square. Default is (0,0,0) which is black.
            region (float): The size of the square relative to the circle. Should match the value used in measure_growth. Default is 0.4.
            thickness (int): Width of the edge of the square. Default is 1.
        """

        # convert the 1 channel image to 3 channel
        if not self.image_colour:
            self._convert_image_to_colour()
            self.image_colour=True

        # check what threshold we should be using
        if self.sensitivity==0:
            growth_threshold_percentage=self.threshold_percentage
        elif (self.categories['IM_POS_AVERAGE']>(self.sensitivity*self.threshold_percentage)):
            growth_threshold_percentage=self.categories['IM_POS_AVERAGE']/self.sensitivity
        else:
            growth_threshold_percentage=self.threshold_percentage

        for iy in range(0,self.well_dimensions[0]):
            for ix in range(0,self.well_dimensions[1]):

                x=self.well_centre[(iy,ix)][0]
                y=self.well_centre[(iy,ix)][1]
                r=self.well_radii[(iy,ix)]*region

                if self.well_growth[iy,ix]>growth_threshold_percentage:
                    if self.categories["IM_POS1"] and self.categories["IM_POS2"]:
                        cv2.rectangle(self.image,(int(x-r),int(y-r)),(int(x+r),int(y+r)),growth_color,thickness=thickness)
                    else:
                        cv2.rectangle(self.image,(int(x-r),int(y-r)),(int(x+r),int(y+r)),(0,0,0),thickness=thickness)

    def delete_mics(self):

        for drug in self.drug_names:
            self.categories.remove("IM_"+drug.upper()+"MIC")
            self.categories.remove("IM_"+drug.upper()+"DILUTION")

    def measure_growth(self,threshold_pixel=130,threshold_percentage=3,region=0.4,sensitivity=4.0):
        """ Analyse each of the wells and decide if there is bacterial growth.

        This is all based on the pixels found in a central square region.
        Note: the pixels have intensities (0-255) where 0 is dark and 255 is light.

        Args:
            threshold_pixel (int): pixels with intensities above this value are assumed to be bacterial growth_threshold_percentage.
                Default is 130. Range 0 to 255.
            threshold_percentage (float): the proportion of the central region of each well that contains bacterial growth for
                the well to be classifed as containing bacterial growth
            region (float): the size of the central square relative to the estimated size of the well. Default is 0.4.
            sensitivity (float): if the average percentage growth in the control wells is greater than this value multiplied by
                threshold_percentage, then, rather than threshold_pixel, use the average percentage growth in the control wells divided by
                sensitivity. This is a higher threshold and better deals with artefacts. The default is 4.0.
                E.g. using defaults, for a plate with 20% average growth in the control wells a threshold of 5%, rather than 3%, would be applied.
        """

        # check the parameters lie within the required ranges
        assert 255>=threshold_pixel>=0, "threshold_pixel must take a value between 0 and 255"
        assert 100>=threshold_percentage>=0, "threshold_percentage must take a value between 0 and 100"
        assert 0.707>=region>=0, "if region is larger than sqrt(2) it will include parts of the well circumference."

        # convert the 3 channel image to 1 channel
        if self.image_colour:
            self._convert_image_to_grey()
            self.image_colour=False

        self.threshold_pixel=threshold_pixel
        self.threshold_percentage=threshold_percentage
        self.sensitivity=sensitivity

        for iy in range(0,self.well_dimensions[0]):
            for ix in range(0,self.well_dimensions[1]):

                x=self.well_centre[(iy,ix)][0]
                y=self.well_centre[(iy,ix)][1]
                r=self.well_radii[(iy,ix)]*region

                rect = self.image[int(y-r):int(y+r),int(x-r):int(x+r)]
                # rect = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
                rect_pixels = rect.flatten()

                self.well_growth[iy,ix] = numpy.sum([rect_pixels<self.threshold_pixel],dtype=numpy.float64)/(rect.shape[0]*rect.shape[1])*100

        counter=1
        positive_control_growth_total=0.0

        # start off assuming there is growth in the control wells
        self.categories['IM_POS_GROWTH']=True

        # loop through the control wells (any well with no conc of a drug)
        for control_well in self.well_positive_controls:

            positive_control_growth=self.well_growth[control_well]

            positive_control_growth_total+=positive_control_growth

            # store the growth in the control well as a % to 2DP
            self.categories['IM_POS'+str(counter)+'GROWTH']=float("%.2f" % (positive_control_growth))

            if positive_control_growth>self.threshold_percentage:
                self.categories['IM_POS'+str(counter)]=True
            else:
                self.categories['IM_POS'+str(counter)]=False
                # if there is isn't growth a single control well set this to False
                self.categories['IM_POS_GROWTH']=False

            counter+=1

        # calculate the average growth (%) in the control wells, limited to 2DP
        self.categories['IM_POS_AVERAGE']=float("%.2f" % (positive_control_growth_total/self.well_positive_controls_number))

        # reset all the image processing fields
        for drug in self.drug_names:
            self.categories["IM_"+drug.upper()+"MIC"]=None
            self.categories["IM_"+drug.upper()+"DILUTION"]=None

        number_drugs_inconsistent_growth=0

        self.categories["IM_DRUGS_INCONSISTENT_GROWTH"]=None

        # check what threshold we should be using
        if self.sensitivity==0:
            growth_threshold_percentage=self.threshold_percentage
        elif (self.categories['IM_POS_AVERAGE']>(self.sensitivity*self.threshold_percentage)):
            growth_threshold_percentage=self.categories['IM_POS_AVERAGE']/self.sensitivity
        else:
            growth_threshold_percentage=self.threshold_percentage

        # now iterate through the drugs
        for drug in self.drug_names:

            # remember to reverse the order so we go from low to high conc
            growth=self.well_growth[self.well_drug_name==drug][::-1]
            conc=self.well_drug_conc[self.well_drug_name==drug][::-1]
            dilution=self.well_drug_dilution[self.well_drug_name==drug][::-1]

            # start off with no MIC and not having seen anything
            mic_conc=None
            mic_dilution=None
            seen_growth=False
            seen_no_growth=False
            inconsistent_growth=False

            # iterate through the strip of wells
            for (g,c,d) in zip(growth,conc,dilution):

                # ..otherwise no choice but to apply a simple threshold
                if g>growth_threshold_percentage:

                    seen_growth=True

                    # but if we have already seen no growth
                    if seen_no_growth:
                        inconsistent_growth=True

                # ..otherwise there is no growth
                else:

                    # and we haven't set an MIC yet
                    if not mic_conc:

                        # ..but we have seen growth and not yet seen no growth
                        if seen_growth and not seen_no_growth:

                            # set the MIC values
                            mic_conc=c
                            mic_dilution=d

                    # and now remember that we've seen a well with no growth in it
                    seen_no_growth=True

            # deal with the case where there is growth in ALL wells
            #  (cannot infer that the concentration of the next doubling would be sufficient to kill the bug)
            #  (hence have to report as anomalous/no result)
            if seen_growth and not seen_no_growth and not mic_conc:
                mic_conc=numpy.max(conc)*2
                mic_dilution=numpy.max(dilution)+1

            # deal with the case where there is no growth in ALL wells
            #  (can infer that the concentration of the first well is at least an upper limit for the MIC)
            if seen_no_growth and not seen_growth and not mic_conc:
                mic_conc=numpy.min(conc)
                mic_dilution=0

            # what if there is anomalous growth?
            if inconsistent_growth:
                mic_conc=-1
                mic_dilution=-1
                number_drugs_inconsistent_growth+=1

            # what if there is no growth in the control wells?
            if self.categories['IM_POS_GROWTH']==False:
                mic_conc=-2
                mic_dilution=-2

            # translate into text for storing in the treant
            if mic_conc is None:
                mic_conc="None"
                mic_dilution="None"

            # pick up any cases where an MIC has not been assigned
            # (indicates an error in the above logic somewhere)
            assert mic_conc!="None"
            assert mic_conc!=None

            self.categories["IM_"+drug.upper()+"MIC"]=mic_conc
            self.categories["IM_"+drug.upper()+"DILUTION"]=mic_dilution

        assert number_drugs_inconsistent_growth>=0
        self.categories["IM_DRUGS_INCONSISTENT_GROWTH"]=number_drugs_inconsistent_growth

    def write_mics(self,filename):
        """ Write the results to a plaintext file.

        These are simply all the data stored in the JSON file, but this is difficult to read, so this
        function provides a simple way to produce human-readable files. It will include the MICs and dilutions
        for each well, as well as information about the growth in the control well(s).
        """

        OUTPUT = open(filename,'w')

        for field in sorted(self.categories.keys()):
            OUTPUT.write("%28s %20s" % (field, self.categories[field]))

        OUTPUT.close()

    def identify_wells(self,hough_param1=20,hough_param2=25,radius_tolerance=0.005):
        """ Using a Hough transform identify the wells.

        Only if the number of circles found is the same as the number of wells is True returned.

        Args:
            radius_tolerance (float): the rate at which the min/max radii are decreased/increased. Increasing it speeds up
                the search, but makes it more likely it will 'miss' identifying the correct number of wells. default=0.005
            hough_param1 (int) and hough_param2 (int): for an explanation of these, please see the OpenCV documentation
                e.g. https://docs.opencv.org/3.1.0/d4/d70/tutorial_hough_circle.html
        """

        # estimate the dimensions of the well
        estimate_well_y = float(self.image_dimensions[0])/self.well_dimensions[0]
        estimate_well_x = float(self.image_dimensions[1])/self.well_dimensions[1]

        # verify that the estimated dimensions of the wells are within 5% of one another
        if estimate_well_x > estimate_well_y:
            if estimate_well_x > 1.05*estimate_well_y:
                print(self.image_name+" has estimated well dimensions more than 10% different - check the image")
                return False
        else:
            if estimate_well_y > 1.05*estimate_well_x:
                print(self.image_name+" has estimated well dimensions more than 10% different - check the image")
                return False

        # now estimate the radius as the mean of half the estimated well dimension
        estimated_radius=(estimate_well_x + estimate_well_y)/4.

        radius_multiplier=1.+radius_tolerance

        # read in a greyscale image for identifying
        grey_image = cv2.imread(self.image_path,0)

        # iterate until the correct number of circles have been found
        while True:

            circles=None

            while circles is None:

                # detect circles using the Hough transform
                circles=cv2.HoughCircles(grey_image,cv2.HOUGH_GRADIENT,1,50,param1=hough_param1,param2=hough_param2,minRadius=int(estimated_radius/radius_multiplier),maxRadius=int(estimated_radius*radius_multiplier))

                # increase the range of radii that will be searched for
                radius_multiplier+=radius_tolerance

            # count how many circles there are
            number_of_circles=len(circles[0])

            # check to see if there are enough circles
            if number_of_circles>=self.number_of_wells:
                break
            elif number_of_circles>self.number_of_wells:
                raise(SystemError, str(number_of_circles)+" circles found, this is too many!")
            else:
                radius_multiplier+=radius_tolerance

        well_counter=0

        # move along the wells in the x-direction (columns)
        for ix in range(0,self.well_dimensions[1]):

            # move along the wells in the y-direction (rows)
            for iy in range(0,self.well_dimensions[0]):

                # calculate the bottom left and top right coordinates of the well
                top_left=(int(ix*estimate_well_x), int(iy*estimate_well_y))
                bottom_right=(int((ix+1)*estimate_well_x), int((iy+1)*estimate_well_y))

                number_of_circles_in_well=0

                # loop over the list of identified wells
                for ic in circles[0,]:

                    # check to see if the circle lies with the well
                    if top_left[0] < ic[0] < bottom_right[0]:
                        if top_left[1] < ic[1] < bottom_right[1]:
                            number_of_circles_in_well+=1
                            circle=ic

                if number_of_circles_in_well!=1:
                    return False

                well_centre=(circle[0],circle[1])
                well_radius=circle[2]
                well_extent=1.2*well_radius

                x1=max(0,int(well_centre[0]-well_extent))
                x2=min(self.image_dimensions[1],int(well_centre[0]+well_extent))

                y1=max(0,int(well_centre[1]-well_extent))
                y2=min(self.image_dimensions[0],int(well_centre[1]+well_extent))

                self.well_index[iy,ix]=well_counter
                self.well_centre[iy,ix] = well_centre
                self.well_radii[iy,ix] = well_radius
                self.well_top_left[iy,ix] = (x1,y1)
                self.well_bottom_right[iy,ix] = (x2,y2)

                well_counter+=1

        if well_counter==self.number_of_wells:
            return True
        else:
            return False
