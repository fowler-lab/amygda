import unittest

import amygda


class TestAMyGDA(unittest.TestCase):


    def setUp(self):

        self.image=amygda.PlateMeasurement("examples/sample-images/08/",categories={'ImageFileName':"image-01-raw.png"},configuration_path="config/",pixel_intensities=False)

    def test__scale_value(self):

        test_pairs = ( (3,2), (3.0,1.5) )

        self.image.scaling_factor=0.5

        for input, output in test_pairs:

            self.assertEqual(self.image._scale_value(input),output)

if __name__=="__main__":

    unittest.main(  )
