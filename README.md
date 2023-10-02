# Attention-based super-resolution GAN for drone image small object detection

We introduce a compact object detection model incorporating a GAN-based super-resolution step to enhance detection results. Previous research on small object detection faced challenges due to the limited pixel count in tiny objects when using CNN models alone. Our approach aims to address this issue and generate improved outcomes. We employ ESRGAN as our base architecture to elevate image resolution. Additionally, we introduce a dual-step attention mechanism—spatial and channel—to further enhance the super-resolution output. The spatial unit identifies the important spatial locations of information, while the channel unit extracts meaningful content from the image.

Our choice of a GAN-based architecture for super-resolution, as opposed to relying solely on Deep CNNs, is driven by the observation that CNN-only methods struggle to capture finer details post-enhancement. This is crucial for achieving superior super-resolution results. The limitation of CNN-based methods lies in their objective function, primarily tied to Mean Squared Reconstruction Error, which excels in peak signal-to-noise ratios but lacks the ability to preserve high-frequency details, intricate patterns, and textures. ESRGAN addresses this by incorporating not only the typical Adversarial Loss of GANs but also introducing an additional "content loss" grounded in perceptual similarity rather than pixel-level matching. Our innovation lies in combining ESRGAN with the attention mechanism. This synergy is particularly valuable given the scale and quantity of images, as it enhances feature extraction efficiency while minimizing required training time.

## Motivation ##

Our problem statement holds significance due to its real-world applications, especially in surveillance and security sectors where rapid object detection using drones is crucial. Our motivation is to enhance object detection in drone images through our attention-boosted GAN super-resolution architecture. Additionally, we aim to utilize channel and spatial attention in a GAN to enhance super-resolution quality while reducing training time. This project aligns with our course as it applies key concepts such as attention and generative adversarial networks, particularly in the super-resolution aspect involving conditional GANs for enhanced image generation.

## Proposed Approach ##

In our implementation, we've explored various super-resolution models such as SRCNN and EDSR. We've also built an ESRGAN from scratch in Python using Keras and TensorFlow, following the architecture outlined in the paper. This architecture initially downscales low-resolution images to capture information in dense vectors. During the upsampling phase, the model learns to represent this dense information in high resolution.

To enhance the super-resolution model's accuracy, we've introduced an Attention mechanism within the residual. This attention mechanism comprises two sub-architectures: "Spatial" and "Channel." The Spatial unit identifies the significant locations of information in the image, while the Channel unit extracts meaningful content. After training the super-resolution model, we process all low-resolution images (64x64) through the trained model, upscaling them to 256x256. These enhanced images are then stored in a folder and used to train the YoloV5 model for the detection of small objects, specifically cars. The image displayed below shows the comparision of our model with SR-ResNet.

![img1](https://github.com/ChandanaGiridhar/Drone_Small_Object_Detection/blob/main/1_model.png)

## Conclusion ##

Our proposal involves implementing a GAN-based super-resolution process preceding object detection. Specifically, we suggest employing the ESRGAN architecture to enhance image resolution and integrating a two-step attention mechanism (comprising spatial and channel attention) to further enhance the output quality. We prefer GAN-based super-resolution over Deep CNNs due to the latter's limitation in preserving finer details post-enhancement.

In comparison to other small object detection algorithms like PP-YOLOE-Plus, Cascade R-CNN + (Normalized Wassertian Distance), and TPH-YOLOv5, we hypothesize the following:
- The inclusion of attention within ESRGAN will reduce training time and enhance super-resolution output quality.
- The improved super-resolution step will boost overall object detection accuracy.

Our assumptions are validated through the results presented above. Additionally, we aim to enhance the system further by exploring alternative super-resolution methods like Diffusion to improve image quality. Furthermore, we plan to create an end-to-end architecture that seamlessly combines super-resolution and object detection processes.

## Related Links ##

Complete Report - [Click Here](https://github.com/ChandanaGiridhar/Drone_Small_Object_Detection/blob/main/SmallObjectDetectionForDrone_Report.pdf)

Implementation - [Click Here](https://github.com/ChandanaGiridhar/Drone_Small_Object_Detection/tree/main/implementation)
