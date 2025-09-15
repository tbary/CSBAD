#CAM1,2
HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=12345 n_samples=64,128,256,384 n_samples_teacher=32 cam=1,2 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3 

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=12345 n_samples=128,256,512,768 n_samples_teacher=64 cam=1,2 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=12345 n_samples=256,512,1024,1536 n_samples_teacher=128 cam=1,2 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=12345 n_samples=512,1024,2048,3072 n_samples_teacher=256 cam=1,2 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

#CAM3
HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=5 n_samples=64,128,256,384 n_samples_teacher=32 cam=3 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=5 n_samples=128,256,512,768 n_samples_teacher=64 cam=3 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=5 n_samples=256,512,1024,1536 n_samples_teacher=128 cam=3 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=5 n_samples=512,1024,2048,3072 n_samples_teacher=256 cam=3 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

#CAM4
HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=23 n_samples=64,128,256,384 n_samples_teacher=32 cam=4 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=23 n_samples=128,256,512,768 n_samples_teacher=64 cam=4 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=23 n_samples=256,512,1024,1536 n_samples_teacher=128 cam=4 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=23 n_samples=512,1024,2048,3072 n_samples_teacher=256 cam=4 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

#CAM5
HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=1235 n_samples=64,128,256,384 n_samples_teacher=32 cam=5 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=1235 n_samples=128,256,512,768 n_samples_teacher=64 cam=5 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=1235 n_samples=256,512,1024,1536 n_samples_teacher=128 cam=5 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=1235 n_samples=512,1024,2048,3072 n_samples_teacher=256 cam=5 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

#CAM6
HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=134 n_samples=64,128,256,384 n_samples_teacher=32 cam=6 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=134 n_samples=128,256,512,768 n_samples_teacher=64 cam=6 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=134 n_samples=256,512,1024,1536 n_samples_teacher=128 cam=6 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=134 n_samples=512,1024,2048,3072 n_samples_teacher=256 cam=6 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

#CAM7
HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=4 n_samples=64,128,256,384 n_samples_teacher=32 cam=7 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=4 n_samples=128,256,512,768 n_samples_teacher=64 cam=7 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=4 n_samples=256,512,1024,1536 n_samples_teacher=128 cam=7 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=4 n_samples=512,1024,2048,3072 n_samples_teacher=256 cam=7 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

#CAM8

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=3 n_samples=64,128,256,384 n_samples_teacher=32 cam=8 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=3 n_samples=128,256,512,768 n_samples_teacher=64 cam=8 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=3 n_samples=256,512,1024,1536 n_samples_teacher=128 cam=8 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=3 n_samples=512,1024,2048,3072 n_samples_teacher=256 cam=8 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

#CAM9
HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=12 n_samples=64,128,256,384 n_samples_teacher=32 cam=9 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=12 n_samples=128,256,512,768 n_samples_teacher=64 cam=9 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=12 n_samples=256,512,1024,1536 n_samples_teacher=128 cam=9 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=12 n_samples=512,1024,2048,3072 n_samples_teacher=256 cam=9 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3


#CAM10,11,12,13,14,15

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=1 n_samples=64,128,256,384 n_samples_teacher=32 cam=10,11,12,13,14,15 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=1 n_samples=128,256,512,768 n_samples_teacher=64 cam=10,11,12,13,14,15 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=1 n_samples=256,512,1024,1536 n_samples_teacher=128 cam=10,11,12,13,14,15 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3

HYDRA_FULL_ERROR=1 python3 train.py --multirun project=sb-ICIP3 week=1 n_samples=512,1024,2048,3072 n_samples_teacher=256 cam=10,11,12,13,14,15 strategy=strat1 teacher_strategy=strat2 student=yolo11n teacher=yolo11x embeddings_dir=dir3