# Poes-estimation-with-Yolo-Pose
YOLO pose was used for infering multiple human joints. 
The mid point between the hip joints are treated as center of mass.
Spine is drawn from mid-point of shoulder to hip midpoint for calculation of body orientation in future
There are of course issue of occulsion, however I am overall quite happy with the tracking of the humans using the bounding box centroid
Perhaps in future, other tracking algorithm like CSRT, GOTRUN etc could be applied (note that this will decrease the FPS further for CPU runned model (current FPS is ~9s on CPU)
