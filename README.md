# horizon-detection
detects horizon line in images of landscapes

This code works to detect the horizon line in images of landscapes which can be used by drones. 
To run code, run
python horizon_detect.py --scale <scale_value> <image_name>
Example: python horizon_detect.py --scale 50 “image3.png”

Image with line will be saved to current directory. 

The value of scale chosen is a tradeoff between speed and accuracy. Higher the value of
scale chosen, higher will be the accuracy in detecting the horizon. However, this would
mean more processing time, which would not work well with real time video.

I have performed the algorithm by down-sampling the image to a much smaller scale
and then extrapolating the line obtained onto the original image size.
The algorithm works in such a way that it finds a straight line to maximize an
optimization function, such that all the similar sky pixels are grouped above the line and
the ground pixels are grouped below the line. Further details can be found in the
comments of the code.
