# Automated Mycobacterial Growth Detection Algorithm (AMyGDA)

This is a python module that takes a photograph of a 96 well plate and assesses each well for the presence of bacterial growth (here *Mycobacterial tuberculosis*). Since each well contains a different concentration of a different antibiotic, the minimum inhibitory concentration, as used in clinical microbiology, can be determined.

The development of this software was funded by the National Institute for Health Research (NIHR) Oxford Biomedical Research Centre (BRC) to aid the [CRyPTIC project](http://www.crypticproject.org).

This software was downloaded from [https://fowlerlab.org/software/amygda](https://fowlerlab.org/software/amygda).

Philip W Fowler

philip.fowler@ndm.ox.ac.uk

4 December 2017

## Pre-requsities

The following python modules should be installed before running AMyGDA.

- [numpy](http://www.numpy.org). It is possible your python installation includes numpy. To check, issue the following in a terminal

		$ python -c "import numpy"

	If you see an error, indicating numpy is not installed, please install the scipy stack by following [these instructions](https://www.scipy.org/install.html).

- [opencv-python](https://pypi.python.org/pypi/opencv-python). This can be installed using  standard python tools, such as pip

		$ pip install opencv-python
	AMyGDA was developed and tested using version 3.2.0 of OpenCV. If you do not have sudo access on your machine you can install this (and any other python module) in your $HOME directory using the following command
	
		$ pip install opencv-python --user	

- [datreant](http://datreant.readthedocs.io/en/latest/). This provides a neat way of storing and discovering metadata for each image using the native filesystem. It is not essential for the operation of AMyGDA, but the code would need re-factoring to remove this dependency. Again it can be installed using pip

		$ pip install datreant.core  
		
	Note that datreant works best if each image is containing within its own folder. datreant automatically stores all metadata associated with each image within a JSON file in the same location as the input file. 

## Tutorial

The code is structured as a python module; all files for which can be found in the amygda/ subfolder. 

	$ ls
	LICENCE.md                   amygda                       plate-configuration
	README.md                    analyse-plate-with-amygda.py sample-images

analyse-plate-with-amygda.py is a simple python file showing how the module can be used to analyse a single image. The same twenty images that are used in the accompanying paper (see below) are provided so you can reconstruct several of the images in the manuscript and Supplementary Information. All these images are organised as follows 

	$ ls sample-images/
	01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
	
	$ ls sample-images/01/
	image-01-raw.png

To process and analyse a single image using the default settings is simply

	$ ./analyse-plate-with-amygda.py --image sample-images/01/image-01-raw.png 
	
And should take no more than 10 seconds. No output is written to the terminal, instead you will find a series of new files have been written in the samples-images/01 folder.

	$ ls sample-images/01/
	PlateMeasurement.0460f5d7-f3b2-45f7-9cdc-2bd407ba0790.json
	image-01-arrays.npz
	image-01-filtered.png
	image-01-mics.txt
	image-01-processed.png
	image-01-raw.png	

* The JSON file contains all the MICs and other metadata about the plate and can be automatically discovered and read using the datreant module to make systematic analyses simpler. 
* image-01-mics.txt contains the same information as the JSON file but in a simpler format that is easier for humans to read.
* image-01-arrays.npz contains a series of numpy arrays that specify e.g. the percentage growth in each well
* image-01-raw.png is the original image of the plate.
* image-01-filtered.png is a PNG of the plate following mean shift filtering and application of a Contrast Limited Adaptive Histogram Equalization filter to improve contrast and equalise the illumination across the plate.
* image-01-processed.png adds some annotation; specifically the locations of the wells are drawn, each well is labelled with the name and concentration of drug and wells which AMyGDA has classified as containing bacterial growth are highlighted with a coloured square. 

To see the other options available for the analyse-plate-with-amygda.py python script

	$ ./python analyse-plate-with-amygda.py --help
	usage: analyse-plate-with-amygda.py [-h] [--image IMAGE]
                                    [--growth_pixel_threshold GROWTH_PIXEL_THRESHOLD]
                                    [--growth_percentage GROWTH_PERCENTAGE]
                                    [--measured_region MEASURED_REGION]
                                    [--sensitivity SENSITIVITY]
                                    [--file_ending FILE_ENDING]

	optional arguments:
	  -h, --help            show this help message and exit
	  --image IMAGE         the path to the image
	  --growth_pixel_threshold GROWTH_PIXEL_THRESHOLD
                        the pixel threshold, below which a pixel is considered
                        to be growth (0-255)
	  --growth_percentage GROWTH_PERCENTAGE
                        if the central measured region in a well has more than
                        this percentage of pixels labelled as growing, then
                        the well is classified as growth.
	  --measured_region MEASURED_REGION
                        the size of the central measured region, as a decimal
                        proportion of the whole well.
	  --sensitivity SENSITIVITY
                        if the average growth in the control wells is more
                        than (sensitivity x growth_percentage), then consider
                        growth down to this sensitivity
	  --file_ending FILE_ENDING
                        the ending of the input file that is stripped. Default
                        is '-raw'

To analyse all plates, you can either use a simple bash loop
	
		$ for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20; do
			./analyse-plate-with-amygda.py --image sample-images/$i/image-$i-raw.png
		  done;
	
Alternatively if you have [GNU parallel](https://www.gnu.org/software/parallel/) installed you can use all the cores on your machine to speed up the process.

		$ find sample-images/ -name '*raw.png' | parallel ./analyse-plate-with-amygda.py --image {}	

To delete all the output files, thereby returning sample-images/ to its clean state, a bash script is provided. Use with caution!

	$ cd samples-images/
	
	$ ls 01/
	PlateMeasurement.1d6f5af8-6005-4bb9-93cc-3390cefebfe4.json image-01-mics.txt
	image-01-arrays.npz                                        image-01-processed.png
	image-01-filtered.png                                      image-01-raw.png
	
	$ bash remove-output-images.sh
	
	$ ls 01/
	image-01-raw.png
	
	
## Citing

A [preprint](https://doi.org/10.1101/229427) is available from the biorXiv and the manuscript is has been submitted to a peer-reviewed journal. Until it is accepted, please cite the biorXiv paper. The final citation will be added here when it is known.

	Automated detection of Mycobacterial growth on 96-well plates for rapid and accurate Tuberculosis drug susceptibility testing
	Philip W Fowler, Ana Luiza Gibertoni Cruz, Sarah J Hoosdally, Lisa Jarrett, Emanuele Borroni, Matteo Chiacchiaretta, Priti Rathod, Timothy M Walker, Esther Robinson, Timothy EA Peto, Daniela Maria M. Cirillo, E Grace Smith, Derrick W Crook
	bioRxiv 229427; doi: https://doi.org/10.1101/229427

## Licence

The software is available subject to the terms of the attached academic-use licence.

## Adapting for different plate designs

AMyGDA is written to be agnostic to the particular design of plate, or even the number of wells on each plate. The concentration (or dilution) of drug in each well is defined by a series of plaintext files in 

	plate-configuration/
	
For example the drugs on the CRyPTIC1 V1 plate is defined within the 

	plate-configuration/CRyPTIC1-V1-drug-matrix.txt

file and looks like.

	BDQ,KAN,KAN,KAN,KAN,KAN,ETH,ETH,ETH,ETH,ETH,ETH
	BDQ,AMI,EMB,INH,LEV,MXF,DLM,LZD,CFZ,RIF,RFB,PAS
	BDQ,AMI,EMB,INH,LEV,MXF,DLM,LZD,CFZ,RIF,RFB,PAS
	BDQ,AMI,EMB,INH,LEV,MXF,DLM,LZD,CFZ,RIF,RFB,PAS
	BDQ,AMI,EMB,INH,LEV,MXF,DLM,LZD,CFZ,RIF,RFB,PAS
	BDQ,AMI,EMB,INH,LEV,MXF,DLM,LZD,CFZ,RIF,RFB,PAS
	BDQ,AMI,EMB,INH,LEV,MXF,DLM,LZD,CFZ,RIF,RFB,PAS
	BDQ,EMB,EMB,INH,LEV,MXF,DLM,LZD,CFZ,RIF,POS,POS	
Adding a new plate design is simply a matter of creating new files specifying the drug, concentration and dilution of each well. Note that changing the *number* of wells at present also involves specifying the well_dimensions when creating a PlateMeasurement object. Currently this defaults to (8,12) i.e. a 96-well plate in landscape orientation.
