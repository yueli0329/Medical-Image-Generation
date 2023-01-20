# Medical-Image-Generation

This project is about image color normalization, one method of data augmentation in Computer vision. <br>

##  Data Resource
The dataset used for this project is a dataset from Kaggle. The original dataset consisted of 162 whole mount slide images of Breast Cancer (BCa) specimens scanned at 40x.[link](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
![Main](https://github.com/yueli0329/Medical-Image-Generation/blob/main/Fake%20Images/negative.jpeg)

## Neural Network Architecture 
The components of Pix2Pix architecture are a follows: <br><br>

**Generator**: A U-Net, which has skip connections, is used so that low level information can be passed from input image to output image. Noise in the form of dropout is applied to several layers of the generator instead of an input noise used in cGAN models.	<br><br>

**Discriminator**: The discriminator is a PatchGAN network. It classifies smaller patches of the input image (either real or fake) instead of discriminating the entire image at once.<br><br>

Here the RGB color and corresponding grayscale transform of an image tile serve as target/condition image pairs. Grayscale images are normalized across channels and serve as a neutral template for stain-style transfer. As training progresses, the stain style generalizes capturing the statistics over the entire training set which had been acquired from different labs. It should be noted that while the Pix2Pix algorithm requires paired data for training, it is easy to satisfy this by applying grayscale transforms. 

## Experiments
**Training**: Color/grayscale image pairs are derived from the H&E stained PCam dataset. The grayscale images serve as input to the generator. The generator outputs stained color images. The discriminator has two input pairs: the generated image/grayscale image pair serves as the fake input, the color/grayscale image pair is the real input.The network is trained using most of the same parameters as the original Pix2Pix paper.<br><br>

**Evaluation**: We plot the target image and generated image during the training cycle. In real world implementations, the target image and generated image are compared using metrics such as PSNR, SSIM as well as human visual perception. The generated image can also be validated on a clinical use-case such as classification.<br><br>

## Results
We implemented Pix2Pix to our Tissue slice dataset. We selected 20,000 images from the train set to train the network. Since the training process was really time consuming, the epoch was set to 6. The results were shown. The input of the generator is a grayscale image and output of the generator is a colored image. After three epochs, the generated image was more detailed and intense in color.  Therefore, this Pix2Pix method can help to normalize the color of the stained images in the train set. <br>

![Main](https://github.com/yueli0329/Medical-Image-Generation/blob/main/Fake%20Images/Fake%20image%20-%20Epoch%200.jpeg)
![Main](https://github.com/yueli0329/Medical-Image-Generation/blob/main/Fake%20Images/Fake%20image%20-%20Epoch%201.jpeg)
![Main](https://github.com/yueli0329/Medical-Image-Generation/blob/main/Fake%20Images/Fake%20image%20-%20Epoch%202.jpeg)
![Main](https://github.com/yueli0329/Medical-Image-Generation/blob/main/Fake%20Images/Fake%20image%20-%20Epoch%203.jpeg)
![Main](https://github.com/yueli0329/Medical-Image-Generation/blob/main/Fake%20Images/Fake%20image%20-%20Epoch%204.jpeg)
![Main](https://github.com/yueli0329/Medical-Image-Generation/blob/main/Fake%20Images/Fake%20image%20-%20Epoch%205.jpeg)
![Main](https://github.com/yueli0329/Medical-Image-Generation/blob/main/Fake%20Images/Fake%20image%20-%20Epoch%206.jpeg)


## Files
- Code: python file contains training and evaluation process
 
## Reference 
1. Pix2Pix Kaggle.[link](https://www.kaggle.com/code/shir0mani/stain-transfer-w-pix2pix-pytorch-lightning)
2. Salehi, P., & Chalechale, A. (2020). Pix2Pix-based stain-to-stain translation: A solution for robust stain normalization in histopathology images analysis. 2020 International Conference on Machine Vision and Image Processing (MVIP). https://doi.org/10.1109/mvip49855.2020.9116895
