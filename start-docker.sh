sudo docker run -it -d --name segmentation \
--mount type=bind,source=/home/developer/Desktop/SEG-PGD-adversarial/,target=./SEG-PGD-adversarial \
--mount type=bind,source=/home/developer/Desktop/city-scapes-dataset/,target=./images \
segmentation