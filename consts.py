from pathlib import Path

UNLABELED_DATA_DIR = 'unlabeled_data_dir'
DOWNLOAD_COMMAND = 'download_model_command'
TRAIN_COMMAND = 'train_command'
MODELS_DIR = Path('..', 'models')
models = {
    'shufflenetv2k16w': {
        DOWNLOAD_COMMAND: f'wget -o {str(MODELS_DIR)} https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.11.0/shufflenetv2k16w-200510-221334-cif-caf-caf25-o10s-604c5956.pkl',
        TRAIN_COMMAND: {f"""
                        time CUDA_VISIBLE_DEVICES=0,1 python3 -m openpifpaf.train \
                      --lr=0.05 \
                      --momentum=0.9 \
                      --epochs=% \
                      --lr-warm-up-epochs=1 \
                      --lr-decay 220 \
                      --lr-decay-epochs=30 \
                      --lr-decay-factor=0.01 \
                      --batch-size=32 \
                      --square-edge=385 \
                      --lambdas 1 1 0.2   1 1 1 0.2 0.2    1 1 1 0.2 0.2 \
                      --auto-tune-mtl \
                      --weight-decay=1e-5 \
                      --update-batchnorm-runningstatistics \
                      --ema=0.01 \
                      --checkpoint {str(MODELS_DIR)}/shufflenetv2k16w-200504-145520-cif-caf-caf25-d05e5520.pkl --extended-scale --orientation-invariant=0.1
                        """}
    }
}
