Introduction:
	This project aims to preform self driving Semantic Segmentation with UNet from Scratch.



Dataset: 
	https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge



Build: 
	NIVIDIA RTX 4060
	Cuda 12.1
	Anaconda 3 (Python 3.11)
	PyTorch version: 2.1.2



Generate ".py" file from ".ui" file:
	1) open Terminal. Navigate to directory
	2) Type "pyuic5 -x qt_main.ui -o qt_main.py"



Core Project Structure:
	GUI.py (Run to generate a GUI)
	main.py (Run to train model)
	unet.py
	qt_main.py
	training.py
	visualization.py
	