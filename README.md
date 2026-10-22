# NTU-2025Fall-DLCV

Deep Learning for Computer Vision 深度學習於電腦視覺 by Frank Wang 王鈺強 learn it as audit student

This course has three homeworks and one final project.

   1. 
      Problem 1: Self-supervised pre-training for image classification and 
      Problem 2: Semantic segmentation
   
   2. Diffusion Models
      Problem 1: Conditional Diffusion Models (30%) [digit dataset - MNIST-M & SVHN]
      Problem 2: DDIM (35%) [face dataset]
      Problem 3: ControlNet (35%) [ControlNet dataset]

   3. Problem 1: Zero-shot inference with LLaVA
      ○ 1-1 Zero-shot with origin LLAVA
      ○ 1-2 Visual Contrastive Decoding
      
      Problem 2 :PEFT on Vision and Language Model for Image Captioning

Browse 2025 [course website](https://vllab.ee.ntu.edu.tw/dlcv.html) for more details.


## Outline

- [HW1 Spec](./DLCV%20Fall%202022%20-%20hw1_intro.pdf): [Report](./HW1/hw1_r11944004.pdf)
  - Image classification ← Pretrained BEiT v1
    - Accuracy: 0.9360
  - Image segmentation ← Pretrained Deeplab v3
    - mIoU: 0.7438
      
- [HW2 Spec](./DLCV%20Fall%202022%20-%20hw2_intro.pdf): [Report](./HW2/hw2_r11944004.pdf)
  - Face generation with GAN ← SNGAN (DCGAN with spectral normalization)
    - FID: 25.986
    - Face recognition accuracy: 0.9110
  - Digit generation with diffusion model ← Ho et al. Classifier-Free Diffusion Guidance.
    - Digit classifier accuracy: 0.9990
  - Domain adversarial network on MNIST-M, SVHN, and USPS
    - M→S Accuracy: 0.4943
    - M→U Accuracy: 0.9025
      
- [HW3 Spec](./DLCV%20Fall%202022%20-%20hw3_intro.pdf): [Report](./HW3/hw3_r11944004.pdf)
  - Zero-Shot image classification with CLIP ← CLIP L/14
    - Accuracy: 0.8124
  - Image captioning with pretrained encoder ← Pretrained DeiT v3 as encoder
    - CIDEr score: 0.9413
    - CLIP score: 0.7310
  - Attention map visualization for image captioning
  - 
- [HW4 Spec](./DLCV%20Fall%202022%20-%20hw4_intro.pdf): [Report](./HW4/hw4_r11944004.pdf)
  - 3D novel view synthesis ← DVGO (voxelized NeRF)
    - PSNR: 35.6029
    - SSIM: 0.9769
  - Self-Supervised pretraining for image classification ← BYOL
    - Accuracy: 0.5985
    - Outperforms the supervised equivalent in both full fine-tuning and frozen backbone evaluation.
