# NTU-2025Fall-DLCV

Deep Learning for Computer Vision 深度學習於電腦視覺 by Frank Wang 王鈺強 learn it as audit student

This course has three homeworks and one final project.

- 1. Self-Supervised pretraining for image classification 
  - Problem 1: Self-supervised pre-training for image classification and 
  - Problem 2: Semantic segmentation
   
- 2. Diffusion Models
  - Problem 1: Conditional Diffusion Models (30%) [digit dataset - MNIST-M & SVHN]
  - Problem 2: DDIM (35%) [face dataset]
      

- 3. Problem 1: Zero-shot inference with LLaVA
  - 1-1 Zero-shot with origin LLAVA
  - 1-2 Visual Contrastive Decoding
      
  - Problem 2 :PEFT on Vision and Language Model for Image Captioning

Browse 2025 [course website](https://vllab.ee.ntu.edu.tw/dlcv.html) for more details.


## Outline

- [HW1 Spec](./DLCV%20Fall%202022%20-%20hw1_intro.pdf): [Report](./HW1/hw1_m11115013.pdf)

  - Self-Supervised pretraining for image classification ← DINO v1 
    - Accuracy: 0.9360
  - Image segmentation ← Pretrained SAM v1
    - mIoU: 0.7438
      
- [HW2 Spec](./DLCV%20Fall%202022%20-%20hw2_intro.pdf): [Report](./HW2/hw2_m11115013.pdf)
  - Digit generation with diffusion model ←  Classifier-Free on MNIST-M, SVHN 
    - MNIST-M acc = 1.0000 (correct/total = 250/250)
    - SVHN acc = 0.9880 (correct/total = 247/250)
    - acc = 0.9940

      
- [HW3 Spec](./DLCV%20Fall%202022%20-%20hw3_intro.pdf): [Report](./HW3/hw3_m11115013.pdf)
  - Zero-Shot inference with LLaVA ← CLIP L/14
    - Zero-shot with origin LLAVA
    - Accuracy: 0.6150 (615/1000)

    - Visual Contastive Decoding
    - Accuracy: 0.6450 (645/1000)


  - PEFT on Vision and Language Model for Image Captioning ← Pretrained Vision Transformer as encoder
    - CIDEr score: 1.0015
    - CLIP score: 0.7404
    
  - Strong baseline  - CIDEr of 0.90, CLIPScore of 0.71
  - Attention map visualization for image captioning
  -
- [HW4 Spec](./DLCV%20Fall%202022%20-%20hw4_intro.pdf): [Report](./HW4/hw4_m11115013.pdf)
  - 3D novel view synthesis ← DVGO (3D Gaussian Splatting)
    - PSNR: 40.1520
    - SSIM: 0.9886
    - LPIPS: 0.0214949
 
  
    

  
   