import sys
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np

from src.dataset import DisasterAssessmentDataset, generate_mock_data, get_val_transforms
from src.model import UNet, SiameseUNet, SimpleViTSegmentation, RiskLSTM
from src.evaluate import evaluate_model
from src.simulate import simulate_real_time_stream
from src.utils import overlay_mask
from src.xai import generate_gradcam

def run_tests():
    print("Test 1: Mock Data Generation...")
    df = generate_mock_data('debug_data', num_samples=2)
    dataset = DisasterAssessmentDataset(df, 'debug_data', transform=get_val_transforms())
    img, tab, mask = dataset[0]
    img = img.unsqueeze(0)
    tab = tab.unsqueeze(0)
    mask = mask.unsqueeze(0)
    print("Mock Data Generated.")

    print("Test 2: Basic U-Net...")
    model1 = UNet(in_channels=6, out_classes=5)
    model1.eval()
    with torch.no_grad():
        out1 = model1(img)
    print(f"U-Net Output Shape: {out1.shape}")

    print("Test 3: Siamese U-Net...")
    model2 = SiameseUNet(in_channels=3, out_classes=5)
    model2.eval()
    with torch.no_grad():
        out2 = model2(img)
    print(f"Siamese U-Net Output Shape: {out2.shape}")

    print("Test 4: Vision Transformer...")
    model3 = SimpleViTSegmentation(in_channels=6, out_classes=5)
    model3.eval()
    with torch.no_grad():
        out3 = model3(img)
    print(f"ViT Output Shape: {out3.shape}")

    print("Test 5: Risk LSTM...")
    model4 = RiskLSTM(input_size=4)
    model4.eval()
    with torch.no_grad():
         out4 = model4(tab.unsqueeze(1))
    print(f"Risk LSTM Output Shape: {out4.shape}")

    print("Test 6: Evaluate Pipeline...")
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    res = evaluate_model(model1, val_loader, torch.device('cpu'))
    print(f"Evaluate metrics: {res}")

    print("Test 7: Simulation Output...")
    simulate_real_time_stream(df, 'debug_data', delay_seconds=0.1)
    
    print("Test 8: XAI / Grad-CAM...")
    # Requires grad
    img.requires_grad_(True)
    cam = generate_gradcam(model1, model1.outc.conv, img)
    print(f"Grad-CAM shape: {cam.shape}")

    print("ALL TESTS PASSED WITH NO ERRORS.")

if __name__ == '__main__':
    run_tests()
