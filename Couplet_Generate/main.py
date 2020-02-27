from model import Model


"""tf 1.12+"""

train_input_file = 'path'
train_target_file = 'path'
test_input_file = 'path'
test_target_file = 'path'
vocab_file = 'path'
num_units = 1024
layers = 4
dropout = 0.2
batch_size = 32
learning_rate = 0.001
output_dir = 'path'
save_step = 100
eval_step = 1000
param_histogram = False
restore_model = False

epochs = 10

train_mode = True
infer_mode = False


def main():
    if train_mode:
        init_train = True
        init_infer = False
        model = Model(train_input_file, train_target_file,
            test_input_file, test_target_file, vocab_file,
            num_units, layers, dropout,
            batch_size, learning_rate, output_dir,
            save_step=save_step, eval_step=eval_step,
            param_histogram=param_histogram, restore_model=restore_model,
            init_train=init_train, init_infer=init_infer)

        model.train(epochs=epochs)

    elif infer_mode:
        init_train = False
        init_infer = True
        model = Model(None, None, None, None,
            vocab_file=vocab_file,
            num_units=num_units, layers=layers, dropout=dropout,
            batch_size=batch_size, learning_rate=learning_rate,
            output_dir=output_dir,
            restore_model=True, init_train=init_train, init_infer=init_infer)

        inputs = input()
        outputs = model.infer(inputs)
        print(outputs)


if __name__ == '__main__':
    main()