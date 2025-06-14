
from utils import *
from models import Translator, compile_and_train
import matplotlib.pyplot as plt
if __name__ == "__main__":
    dataset = load_data()

    vectorizers = load_vectorizers(dataset, input_lang = 'en', output_lang = 'vi')
    processed_dataset = preprocess_data(vectorizers, dataset = dataset, input_lang='en', output_lang = 'vi')
    x = processed_dataset['en']
    y = processed_dataset['vi']

    del processed_dataset
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_val_split(x, y, train_size=0.8, test_size=0.1)
    en_vocab_size = vectorizers['en'].vocabulary_size()
    vi_vocab_size = vectorizers['vi'].vocabulary_size()


    def make_dataset(x, y, batch_size=64, shuffle=True):
        # Ensure tensors
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        # Shift decoder input và output
        decoder_input = y[:, :-1]
        decoder_target = y[:, 1:]

        # Tạo dataset
        dataset = tf.data.Dataset.from_tensor_slices(((x, decoder_input), decoder_target))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    train_data = make_dataset(x_train, y_train)
    val_data = make_dataset(x_val, y_val)
    model = Translator(input_vocab_size=en_vocab_size, output_vocab_size=vi_vocab_size, units = 512, num_heads=2)

    translator, history = compile_and_train(model = model, train_data= train_data, val_data = val_data, epochs = 20)
    translator.save("my_model.keras")
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

