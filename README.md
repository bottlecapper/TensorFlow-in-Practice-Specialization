# TensorFlow-in-Practice-Specialization
Course_1_Part_4_Lesson_2: mnist.load_data, fashion_mnist, models.Sequential, tf.keras.layers.Flatten, tf.keras.layers.Dense, model.compile, model.predict, model.fit, callbacks

Course_1_Part_6_Lesson_2: tf.keras.layers.Conv2D, tf.keras.layers.MaxPooling2D, model.summary(), model.evaluate, Visualizing the Convolutions and Pooling

Course_1_Part_6_Lesson_3: convolution, (2, 2) pooling filter from scratch

Course_1_Part_8_Lesson_3: horse-or-human, os.path.join, os.listdir, display a batch of 8 horse and 8 human pictures, ImageDataGenerator, train_datagen.flow_from_directory, Visualizing Intermediate Representations, os.kill(os.getpid(), signal.SIGKILL)

Course_2_Part_2_Lesson_2: Evaluating Accuracy and Loss for the Model

Course_2_Part_4_Lesson_4: ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

Course_2_Part_6_Lesson_3: pre_trained_model = InceptionV3(...), pre_trained_model.load_weights, pre_trained_model.get_layer, layer.trainable, last_layer.output

Course_2_Part_8_Lesson_2: rps (rock, paper, scissors), tf.keras.layers.Dropout

Exercise_1_Cats_vs_Dogs: cats-v-dogs, split_data

Exercise_1_House_Prices: mean_squared_error, model.predict 

Exercise_2_Cats_vs_Dogs_using_augmentation: cats-and-dogs, os.mkdir, split_data

Exercise_3_Horses_vs_humans_using_Transfer_Learning: InceptionV3, path_inception, pre_trained_model.load_weights, pre_trained_model.get_layer, last_layer.output, layers.Dropout, horse-or-human, ImageDataGenerator

Exercise_4_Multi_class_classifier: csv.reader, get_data, np.expand_dims, train_datagen.flow, validation_datagen.flow

Course_3_Week_1_Exercise: bbc-text, Tokenizer, pad_sequences, stopwords, csv_reader, tokenizer.fit_on_texts, tokenizer.word_index, tokenizer.texts_to_sequences, pad_sequences

Course_3_Week_1_Lesson_2: Tokenizer, pad_sequences, tokenizer.fit_on_texts, tokenizer.word_index, tokenizer.texts_to_sequences

Course_3_Week_1_Lesson_3: json.load, oov_token="<OOV>"
  
Course_3_Week_2_Exercise: train_padded, validation_padded, training_label_seq, validation_label_seq, tf.keras.layers.Embedding, tf.keras.layers.GlobalAveragePooling1D, tf.keras.layers.Dense, sparse_categorical_crossentropy, plot_graphs, decode_sentence, get_weights, out_m.write, out_v.write, vecs.tsv, meta.tsv

Course_3_Week_2_Lesson_1: tfds.load, imdb_reviews, trunc_type, decode_review, binary_crossentropy, test (tokenizer.texts_to_sequences)

Course_3_Week_2_Lesson_2: padding_type, decode_sentence, get_weights, test (tokenizer.texts_to_sequences)

Course_3_Week_2_Lesson_3: imdb_reviews/subwords8k, info.features['text'].encoder, train_dataset.shuffle

Course_3_Week_3_Lesson_1a: Single Layer LSTM, tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))

Course_3_Week_3_Lesson_1b: Multiple Layer LSTM, tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))

Course_3_Week_3_Lesson_1c: Multiple Layer GRU, tf.keras.layers.Conv1D, tf.keras.layers.GlobalAveragePooling1D

Course_3_Week_3_Lesson_2: sarcasm.json, labels.append(item['is_sarcastic']), model.save("test.h5")

Course_3_Week_3_Lesson_2c: sarcasm.json

Course_3_Week_3_Lesson_2d: imdb_reviews

Course_3_Week_4_Lesson_1: n_gram_sequence, tf.keras.utils.to_categorical

Course_3_Week_4_Lesson_2: irish-lyrics-eof.txt, tf.keras.utils.to_categorical, model.add(Bidirectional(LSTM(150)))

NLP_Course_Week_3_Exercise: training_cleaned.csv, csv.reader, glove.6B.100d.txt, tf.keras.layers.Embedding, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dense

NLP_Week4_Exercise_Shakespeare: sonnets.txt, ku.to_categorical, model.add(Embedding), Bidirectional(LSTM), Dropout, LSTM, test (seed_text)

S+P_Week_1_Lesson_2: plot_series, trend, seasonal_pattern, seasonality, series = baseline + trend + seasonality, white_noise, split_time, autocorrelation, impulses, series_diff, autocorrelation_plot, ARIMA, pd.read_csv, series.autocorr

S+P_Week_1_Lesson_3: plot_series, trend, seasonal_pattern, seasonality, noise, naive_forecast, mean_squared_error, mean_absolute_error, moving_average_forecast, diff_moving_avg_plus_past

S+P_Week_1_Lesson_3: diff_moving_avg_plus_smooth_past

S+P_Week_2_Exercise: windowed_dataset, forecast

S+P_Week_2_Lesson_1: tf.data.Dataset.range(10), dataset.window, dataset.flat_map, dataset.map, dataset.shuffle, dataset.batch

S+P_Week_2_Lesson_2: windowed_dataset, forecast

S+P_Week_2_Lesson_3: windowed_dataset, tf.keras.callbacks.LearningRateScheduler, forecast[split_time-window_size:]

S+P_Week_3_Exercise: tf.keras.backend.clear_session(), tf.keras.layers.Lambda, Bidirectional, Bidirectional, Dense, Lambda

S+P_Week_3_Lesson_2_RNN: tf.keras.layers.Lambda, SimpleRNN, SimpleRNN, Dense, Lambda, plt.figure(figsize=(10, 6))

S+P_Week_3_Lesson_4_LSTM: tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)), tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))

S+P_Week_4_Exercise: daily-min-temperatures.csv, csv.reader, model_forecast, tf.keras.layers.Conv1D, LSTM, LSTM, Dense, Dense, Dense, Lambda, plt.semilogx

S+P_Week_4_Lesson_1: tf.keras.layers.Conv1D, Bidirectional(LSTM), Bidirectional(LSTM), Dense, Lambda, tf.keras.losses.Huber, mean_absolute_error

S+P_Week_4_Lesson_3: sunspots.csv, tf.keras.layers.Dense, Dense, Dense, loss="mse"

S+P_Week_4_Lesson_5: sunspots.csv, windowed_datasettf.keras.layers.Conv1D, LSTM, LSTM, Dense, Dense, Dense, Lambda, plt.semilogx, windowed_dataset, model_forecast

S+P_Week_1_Exercise: plot_series, time = np.arange(4 * 365 + 1, dtype="float32"), mean_squared_error, mean_absolute_error, moving_average_forecast, diff_moving_avg_plus_past, diff_moving_avg_plus_smooth_past, series = baseline + trend + seasonality + noise


