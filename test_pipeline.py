import sys
import unittest
import torch
from src.model import UNet
from src.dataset import DisasterAssessmentDataset, get_train_transforms, generate_mock_data
from src.utils import calculate_iou, calculate_metrics
import shutil
import os

class TestDisasterAssessmentPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = './test_data'
        cls.metadata = generate_mock_data(cls.test_dir, num_samples=2)
        
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_dataset(self):
        transforms = get_train_transforms()
        dataset = DisasterAssessmentDataset(self.metadata, self.test_dir, transform=transforms)
        
        self.assertEqual(len(dataset), 2)
        
        img_tensor, mask_tensor = dataset[0]
        
        self.assertEqual(img_tensor.shape[0], 6) # 3 + 3 channels
        self.assertEqual(img_tensor.shape[1], 256)
        self.assertEqual(img_tensor.shape[2], 256)
        
        self.assertEqual(mask_tensor.shape[0], 256)
        self.assertEqual(mask_tensor.shape[1], 256)

    def test_model(self):
        model = UNet(in_channels=6, out_classes=5)
        # Batch size 2
        dummy_input = torch.randn(2, 6, 256, 256)
        
        output = model(dummy_input)
        
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 5) # 5 classes
        self.assertEqual(output.shape[2], 256)
        self.assertEqual(output.shape[3], 256)

    def test_metrics(self):
        # Create dummy predictions and targets
        preds = torch.randn(2, 5, 256, 256)
        targets = torch.randint(0, 5, (2, 256, 256))
        
        metrics = calculate_metrics(preds, targets)
        self.assertIn('accuracy', metrics)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        
        iou = calculate_iou(preds, targets)
        self.assertIsInstance(iou, float)

if __name__ == '__main__':
    unittest.main()
