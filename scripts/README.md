# Scripts for image generation

Generating 5000 images 256px x 256px takes about 1.5h.

```
python pykitPIV-generate-images.py --n_images 100 --size_buffer 10 --image_height 256 --image_width 256
```

# Scripts for training RL

```
python train-RL.py --case_name 'TEST' --n_iterations 4 --discount_factor 0.95 --batch_size 32 --memory_size 64 --n_episodes 2000 --epsilon_start 0.05 --alpha_lr 1 --initial_learning_rate 0.00001
```