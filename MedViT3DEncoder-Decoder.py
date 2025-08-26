import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from PIL import Image  # Import PIL Image

NUM_EPOCHS = 10
BATCH_SIZE = 10
lr = 0.005

class MedViTEncoder(nn.Module):
    def __init__(self, in_channels, encoder_out_channels, num_layers, downsample_factors, embed_dim, num_heads, mlp_dim):
        super(MedViTEncoder, self).__init__()
        self.layers_2d = nn.ModuleList()
        self.layers_3d = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_layers):
            # 2D layers
            self.layers_2d.append(
                nn.Conv2d(current_channels, encoder_out_channels, kernel_size=3, stride=1, padding=1)
            )
            self.layers_2d.append(nn.ReLU(inplace=True))
            if downsample_factors[i] > 1:
                self.layers_2d.append(
                    nn.AvgPool2d(kernel_size=downsample_factors[i], stride=downsample_factors[i])
                )

            # 3D layers
            self.layers_3d.append(
                nn.Conv3d(current_channels, encoder_out_channels, kernel_size=3, stride=1, padding=1)
            )
            self.layers_3d.append(nn.ReLU(inplace=True))
            if downsample_factors[i] > 1:
                self.layers_3d.append(
                    nn.AvgPool3d(kernel_size=downsample_factors[i], stride=downsample_factors[i])
                )

            current_channels = encoder_out_channels

    def forward(self, x):
        if x.dim() == 4:  # 2D image (B, C, H, W)
            for layer in self.layers_2d:
                x = layer(x)
        elif x.dim() == 5:  # 3D image (B, C, D, H, W)
            for layer in self.layers_3d:
                x = layer(x)
        else:
            raise ValueError("Input tensor must be 4D (2D image) or 5D (3D image)")
        return x


class MedViTDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, upsample_factors):
        super(MedViTDecoder, self).__init__()
        self.layers_2d = nn.ModuleList()
        self.layers_3d = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_layers):
            # 2D layers
            if upsample_factors[i] > 1:
                self.layers_2d.append(
                    nn.ConvTranspose2d(current_channels, current_channels, kernel_size=upsample_factors[i], stride=upsample_factors[i])
                )
            self.layers_2d.append(
                nn.Conv2d(current_channels, out_channels if i == num_layers - 1 else current_channels, kernel_size=3, stride=1, padding=1)
            )
            if i != num_layers - 1:
                self.layers_2d.append(nn.ReLU(inplace=True))

            # 3D layers
            if upsample_factors[i] > 1:
                self.layers_3d.append(
                    nn.ConvTranspose3d(current_channels, current_channels, kernel_size=upsample_factors[i], stride=upsample_factors[i])
                )
            self.layers_3d.append(
                nn.Conv3d(current_channels, out_channels if i == num_layers - 1 else current_channels, kernel_size=3, stride=1, padding=1)
            )
            if i != num_layers - 1:
                self.layers_3d.append(nn.ReLU(inplace=True))


    def forward(self, x):
        if x.dim() == 4:  # 2D image (B, C, H, W)
            for layer in self.layers_2d:
                x = layer(x)
        elif x.dim() == 5:  # 3D image (B, C, D, H, W)
            for layer in self.layers_3d:
                x = layer(x)
        else:
            raise ValueError("Input tensor must be 4D (2D image) or 5D (3D image)")
        return x


class MedViT(nn.Module):
    def __init__(self, in_channels, encoder_out_channels, decoder_out_channels, num_encoder_layers, num_decoder_layers, downsample_factors, upsample_factors, embed_dim, num_heads, mlp_dim):
        super(MedViT, self).__init__()
        self.encoder = MedViTEncoder(in_channels, encoder_out_channels, num_encoder_layers, downsample_factors, embed_dim, num_heads, mlp_dim)
        self.decoder = MedViTDecoder(encoder_out_channels, decoder_out_channels, num_decoder_layers, upsample_factors)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 修正模型初始化（将 in_channels 改为1）
# 正确调用示例（需根据实际参数值调整）
model = MedViT(
    in_channels=1,                  # 输入通道数（如1通道灰度图）
    encoder_out_channels=64,        # ✅ 修正参数名
    decoder_out_channels=1,         # 解码器输出通道数（如分割任务的1通道Mask）
    num_encoder_layers=3,           # 编码器层数
    num_decoder_layers=3,           # 解码器层数
    downsample_factors=[2, 2, 2],   # 编码器下采样倍数
    upsample_factors=[2, 2, 2],     # 解码器上采样倍数
    embed_dim=768,                  # Transformer嵌入维度（若模型包含Transformer）
    num_heads=8,                    # Transformer头数（若模型包含Transformer）
    mlp_dim=2048                    # Transformer MLP维度（若模型包含Transformer）
)

#model = MedViT(in_channels=1, encoder_out_channels=64, decoder_out_channels=1, num_encoder_layers=3, num_decoder_layers=3, downsample_factors=[2, 2, 2], upsample_factors=[2, 2, 2])

# Create dummy data for demonstration purposes
class DummyDataset(Dataset):
    def __init__(self, num_samples, is_3d=False, transform=None):
        self.num_samples = num_samples
        self.is_3d = is_3d
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.is_3d:
            # Dummy 3D image and mask (e.g., 1 channel, 32x32x32)
            image = torch.randn(1, 32, 32, 32)
            mask = torch.randint(0, 2, (1, 32, 32, 32)).float()
        else:
            # Dummy 2D image and mask (e.g., 1 channel, 224x224)
            # Generate numpy array instead of tensor for ToPILImage, and ensure it's in HxWxC format
            image = np.random.randn(224, 224, 1).astype(np.float32)
            mask = torch.randint(0, 2, (1, 224, 224)).float()


        if self.transform:
            # Convert numpy array to PIL Image before applying the transform
            if not self.is_3d:
                image = Image.fromarray(image.squeeze().astype(np.uint8)) # Convert to uint8 for PIL

            image = self.transform(image)

        return image, mask

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Create dummy datasets for 2D and 3D
train_dataset_2d = DummyDataset(num_samples=100, is_3d=False, transform=train_transform)
test_dataset_2d = DummyDataset(num_samples=20, is_3d=False, transform=test_transform)

# Note: For 3D data, you would need a different transform that handles 3D tensors.
# train_dataset_3d = DummyDataset(num_samples=100, is_3d=True, transform=transform_3d)
# test_dataset_3d = DummyDataset(num_samples=20, is_3d=True, transform=transform_3d)


# Use the 2D dummy dataset for now
train_loader = DataLoader(dataset=train_dataset_2d, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = DataLoader(dataset=train_dataset_2d, batch_size=2 * BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset_2d, batch_size=2 * BATCH_SIZE, shuffle=False)


print(train_dataset_2d)
print("===================")
print(test_dataset_2d)


# define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()


optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# train

for epoch in range(NUM_EPOCHS):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for inputs, targets in tqdm(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()



# evaluation

def test(split, data_loader):
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Assuming binary segmentation for simplicity
            # Apply sigmoid to outputs to get probabilities
            outputs = torch.sigmoid(outputs)

            y_true.extend(targets.cpu().numpy().flatten())
            y_score.extend(outputs.cpu().numpy().flatten())

        # You would typically calculate metrics like Dice or IoU for segmentation
        # For this example, we'll just print a message
        print(f'{split} evaluation complete.')
        # You can add your evaluation metric calculations here

print('==> Evaluating ...')
test('train', train_loader_at_eval)
test('test', test_loader)