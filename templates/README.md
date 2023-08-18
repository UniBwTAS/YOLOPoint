# Templates
Use templates to filter out keypoint and object detections in unwanted static areas, e.g. the vehicle hood.

Templates should be stored as grayscale png images, where all unwanted area pixels have the value 0,
and the rest 1.

All templates should have exactly the same width and height as the input image.

![example_template](temp_front_bw.png "Template for front camera")
