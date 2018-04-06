# FGSM

TO REPRODUCE,
part1. & part 2
just run in cmd the following, 
	"python part1/2.py eps", 
where eps is the number of eps chosen. This will produce the success rate and accuracy for this eps. The model is already saved in checkpoints and willalso generate variable data in the same file.

part3
to run is the same as part1 & part2, but the pretrained data variables and models are also saved in checkpoints. 

FOR IMAGES, 
they will automically be saved inside the images file once part1 to part3 are all run. 

FOR EVALUATIONS,
it would be better to change the code inside the .ipynb file, then find the following line,
	"xt=mnist.train.images
	 yt=mnist.train.labels"
and change the xt, yt values to other ndarrays.