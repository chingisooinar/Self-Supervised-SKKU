Hyperparameters



|Parameter|Meaning|
| - | - |
|low-dim|categories|
|lr|Learning rate|
|b|Batch size|
|model|Network name|
|Instance|Run title|
|fixmatch|Whether to use Fixmatch|
|threshold|Threshold for pseudo labels|
|T|temperature|
|r|Checkpoint name|
|test-only|Whether to run inference only|
|uda|Whether to use UDA|
|ensemble|Whether to run Ensemble inference|
**Example for training using FixMatch**

train.py --low-dim 10 --lr 1e-4 --b 64 --model vgg19 --instance NAME --fixmatch True --threshold 0.95 --T 0.5

**Example for training using UDA**

train.py --low-dim 10 --lr 1e-4 --b 64 --model vgg19 --NAME --uda True

**Example for training Supervised only**

train.py --low-dim 10 --lr 1e-4 --b 64 --model vgg19 --NAME

**Example for inference**

train.py --low-dim 10 --lr 1e-4 --b 64 --model vgg16 --instance NAME -r NAME.t7 --test-only

**Example for Ensemble inference**

python3 train.py --low-dim 10 --lr 1e-4 --b 64 --model vgg19,vgg16 --instance ensemble -r NAME.t7,NAME2.t7 --test-only --ensemble True
