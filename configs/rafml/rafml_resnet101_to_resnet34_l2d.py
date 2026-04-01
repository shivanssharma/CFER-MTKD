from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import os
from tools.cut_out_pil import CutoutPIL
# ================== Dataset ==================
dataset = "rafml"
img_dir = "Image/aligned/aligned"
label_file = "EmoLabel/multilabel.txt"       
au_label_file = "./Rafau label/RAFAU_label.txt"
partition_file = "EmoLabel/partition_label.txt"
num_classes = 6   # Surprise, Fear, Disgust, Happiness, Sadness, Anger
num_au_classes = 37    # or however many AUs you have
lambda_au = 0.5
img_size = 224

# ================== Training ==================
batch_size = 64

# Teacher learning rate & epochs
lr_t = 0.0001
weight_decay = 0.0001
stop_epoch_t =80       
max_epoch_t = 80
training_mode="MTL"

# Student learning rate & epochs
lr_s = 0.0001
stop_epoch_s = 80
max_epoch_s = 80

# Data augmentation

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(3/4, 4/3)),
    transforms.RandomHorizontalFlip(p=0.5),
    AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    CutoutPIL(cutout_factor=0.4),     
    transforms.ColorJitter(0.2,0.2,0.2,0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

test_transform = transforms.Compose([
    transforms.Resize(int(img_size * 1.14)),  # short edge scaling for center crop
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])
# ================== Models ==================
model_t = "resnet101_mtl"   # teacher model
model_s = "resnet34_mtl"    # student model

teacher_pretrained = False
student_pretrained = True

teacher = model_t
student = model_s

# ================== Loss balancing ==================
lambda_mld = 10      # 10
lambda_le = 1.0       # label embedding / structural loss scale
criterion_t2s_para = dict(
    name="L2D",
    para=dict(
        lambda_ft=0.0,
        ft_dis=None,
        lambda_le=1.0,
        le_dis=dict(
            name="LED",
            para=dict(
                lambda_cd=100,   #100.0
                lambda_id=1000    #1000.0
            )
        ),
        lambda_logits=10,  #this is 10
        logits_dis=dict(
            name="MLD",
            para=dict()
        )
    )
)

# ================== Helper ==================
img_dir = os.path.join(img_dir)
label_file = os.path.join(label_file)
partition_file = os.path.join(partition_file)


opt_type = 'adam'           # 'adam' or 'sgd'
onecycle_max_lr = 5e-4
lr_s = 1e-4                 # base lr for Adam
weight_decay = 1e-4
loss_type = 'bce'          
use_tta = True





