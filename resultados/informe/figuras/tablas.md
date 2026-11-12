#### IoU per part (%)

| Part | YOLO26x-seg | RF-DETR Seg 2XL | DINOv3 ViT-L + decoder | SAM 3 (zero-shot) |
|---|---|---|---|---|
| Body | 90.9 | 93.1 | **93.6** | 77.2 |
| Licence plate | 89.3 | 90.1 | 82.4 | **90.8** |
| Wheel | 91.1 | 93.5 | **94.9** | 89.8 |
| Windscreen | 92.2 | 92.6 | **94.1** | 45.9 |
| Rear window | 88.2 | 88.0 | **91.6** | 73.9 |
| Left front window | 84.9 | 88.3 | **90.7** | 14.7 |
| Left front door | 89.2 | 92.9 | **94.1** | 21.4 |
| Left rear window | 83.8 | 88.1 | **89.3** | 1.6 |
| Left rear door | 91.5 | 90.2 | **93.4** | 3.1 |
| Right front window | 80.3 | 86.4 | **89.1** | 8.8 |
| Right front door | 89.0 | **94.8** | 94.6 | 12.9 |
| Right rear window | 84.5 | **88.0** | 88.0 | 8.0 |
| Right rear door | 87.9 | 91.4 | **93.8** | 0.8 |
| **Mean** | 87.9 | 90.6 | 91.5 | 34.6 |

#### Pixel accuracy per part (%)

| Part | YOLO26x-seg | RF-DETR Seg 2XL | DINOv3 ViT-L + decoder | SAM 3 (zero-shot) |
|---|---|---|---|---|
| Body | 94.3 | **96.7** | 96.7 | 82.8 |
| Licence plate | 94.9 | 92.9 | 86.3 | **97.5** |
| Wheel | **97.7** | 95.8 | 97.0 | 97.7 |
| Windscreen | **97.9** | 96.6 | 96.8 | 94.5 |
| Rear window | 94.5 | 92.3 | **95.1** | 91.4 |
| Left front window | **97.1** | 94.0 | 94.9 | 52.9 |
| Left front door | 96.5 | 96.5 | **97.3** | 64.1 |
| Left rear window | 92.5 | 92.2 | **93.7** | 1.7 |
| Left rear door | 97.3 | 92.1 | **97.3** | 6.5 |
| Right front window | **95.7** | 94.7 | 94.8 | 21.5 |
| Right front door | 96.3 | 97.1 | **97.7** | 28.7 |
| Right rear window | **94.3** | 93.0 | 93.5 | 14.3 |
| Right rear door | 92.6 | 93.8 | **97.3** | 1.8 |
| **Mean** | 95.5 | 94.4 | 95.3 | 50.4 |

#### Detection F1 per part (%)

| Part | YOLO26x-seg | RF-DETR Seg 2XL | DINOv3 ViT-L + decoder | SAM 3 (zero-shot) |
|---|---|---|---|---|
| Body | 96.4 | 77.2 | **98.8** | 76.6 |
| Licence plate | 90.9 | **93.1** | 91.6 | 91.8 |
| Wheel | 91.1 | **93.9** | 93.7 | 90.9 |
| Windscreen | 93.4 | 94.0 | **94.5** | 61.2 |
| Rear window | 88.6 | 90.9 | **91.7** | 55.0 |
| Left front window | 91.4 | 90.1 | **92.9** | 19.8 |
| Left front door | **93.6** | 93.4 | 93.4 | 18.5 |
| Left rear window | 92.2 | 93.3 | **93.8** | 21.9 |
| Left rear door | 92.3 | **92.5** | 91.7 | 0.5 |
| Right front window | 91.4 | 90.5 | **93.6** | 3.3 |
| Right front door | 92.6 | **93.1** | 92.6 | 5.9 |
| Right rear window | 91.3 | 93.1 | **93.6** | 16.7 |
| Right rear door | **91.3** | 91.1 | 90.1 | 6.0 |
| **Mean** | 92.0 | 91.2 | 93.2 | 36.0 |

#### Detection recall per part (%)

| Part | YOLO26x-seg | RF-DETR Seg 2XL | DINOv3 ViT-L + decoder | SAM 3 (zero-shot) |
|---|---|---|---|---|
| Body | 93.9 | 77.1 | **99.1** | 76.7 |
| Licence plate | 86.3 | 91.3 | 91.9 | **96.3** |
| Wheel | 89.5 | 94.2 | **94.9** | 93.7 |
| Windscreen | 91.9 | **97.1** | 95.0 | 93.0 |
| Rear window | 84.5 | 90.7 | **95.2** | 90.3 |
| Left front window | 89.8 | 90.9 | **93.7** | 28.7 |
| Left front door | 91.9 | 93.1 | **93.2** | 24.1 |
| Left rear window | 91.1 | 92.2 | **94.8** | 32.2 |
| Left rear door | 90.3 | 91.9 | **92.8** | 0.8 |
| Right front window | 90.8 | 92.1 | **94.7** | 5.5 |
| Right front door | 91.3 | 92.0 | **93.6** | 9.4 |
| Right rear window | 90.9 | 93.5 | **95.8** | 30.0 |
| Right rear door | 89.7 | 91.0 | **91.8** | 9.7 |
| **Mean** | 90.2 | 91.3 | 94.4 | 45.4 |

#### Left/right and front/back confusion (% of pixels, lower is better)

| Pair | YOLO26x-seg | RF-DETR Seg 2XL | DINOv3 ViT-L + decoder | SAM 3 (zero-shot) |
|---|---|---|---|---|
| Front windows L↔R | 0.35 | 0.72 | **0.34** | 21.07 |
| Front doors L↔R | **0.30** | 0.30 | 0.30 | 34.01 |
| Rear windows L↔R | 0.32 | 0.40 | **0.30** | 16.21 |
| Rear doors L↔R | 0.44 | 0.46 | **0.37** | 15.51 |
| Windscreen ↔ rear window | 0.11 | 0.46 | **0.09** | 2.15 |
