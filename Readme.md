**********************************************
**Subject		: SMAI**

**Project Name	: GAN**

**Created by Team : kNN-Classifiers**

**********************************************
**Generative Adversarial Networks(GAN)**

**Idea:** Set up a game between two players, Generator and Discriminator

**Generator:**
Creates samples that are intended to come from the same distribution as the training data
The generator is trained to fool the discriminator

**Discriminator:**
Examines samples to determine whether they are real or fake
The discriminator learns using traditional supervised learning techniques, dividing inputs into two classes (real or fake)
*********************************************
Given the overview of GAN, we will now see the execution of code:

**Dependencies:**

You have to install pytorch and torchvision, if not present already, using the following commands:

**If you are using python3(3.5):**

pip3 install http://download.pytorch.org/whl/cu75/torch-0.1.12.post1-cp35-cp35m-linux_x86_64.whl

pip3 install torchvision

**If you are using python2(2.7):**

pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post1-cp27-none-linux_x86_64.whl

pip install torchvision

**Note :** 

1. The code is written in python3.5. To run the code, you have to follow 1st step using python3

2. These commands can be generated from the website pytorch.org as well.
*********************************************

**How to run:**

python3 gan.py --dataroot <path-to-download-and-use-CIFAR-data> --out_folder <path-to-save-fakeImages>

**Note :** 

python3 is used as we used python3.5 libraries in code.
*****************************************
The dataset we used for this GAN implementation is CIFAR data which are 60000 32x32 color images out of which 50000 are training images and 10000 are testing images. These images are classified into 10 classes - airplane, automobile, bird, cat, deer, dog, frog, horse, shiop, truck.
*****************************************
**Important features used:**

* Learning rate   : 0.0002

* No of iterations: 25

* Batch Size	    : 100 (100 images are processed at a time)

* Loss function   : Binary Cross Entropy(BCE Loss)

* Noise Dimension : 100