from dust3r.dust3r.model import AsymmetricCroCo3DStereo
import torch

model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)