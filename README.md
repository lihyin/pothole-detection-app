# Pothole Detection Android App
* It is modified from the [Tensorflow Android Demo App](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android). Set up the project by following [this link](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android) accordingly. 

* Use [Pothole Detection Lambda Function](https://github.com/lihyin/pothole-detection-lambda) to receive data into AWS Mysql Database
* Use [Pothole Detection AWS Sagemaker Validation Job](https://github.com/lihyin/pothole-detection-labeling-batch) to generate  the AWS Sagemaker labelling dataset from the detection result. The generated datasets will be used for validation at AWS Sagemaker Groundtruth. Then we can use the validated dataset to re-train the detection model and improve the accuracy.
