from models.baseline_cnn import build_baseline_cnn

model = build_baseline_cnn(input_shape=(32, 32, 3))

model.summary()
