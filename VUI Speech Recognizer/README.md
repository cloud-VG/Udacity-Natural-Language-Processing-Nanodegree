# VUI Speech Recognizer
In this notebook, I have built deep neural networks using [Keras](https://keras.io/) that functions as part of an end-to-end automatic speech recognition (ASR) pipeline! The completed pipeline will accept raw audio as input and return a predicted transcription of the spoken language. In this capstone project, 5 RNN architectures are used and compared with each other. The full pipeline is summarized in the figure below.

![pipeline](images/pipeline.png)


We begin by investigating the [LibriSpeech dataset](http://www.openslr.org/12/) that will be used to train and evaluate our models. Algorithm will first convert any raw audio to feature representations that are commonly used for ASR. Then we'll move on to building neural networks that can map these audio features to transcribed text.

### Final Model

```python
def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # Convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    
    # RNN layers
    # layer-1
    bidir_rnn_1 = Bidirectional(GRU(units, return_sequences=True, dropout=0.2, name='bidir_rnn_1'))(bn_cnn)
    bn_rnn_1 = BatchNormalization(name='bn_rnn_1')(bidir_rnn_1)
    
    # layer-2
    bidir_rnn_2 = Bidirectional(GRU(units, return_sequences=True, dropout=0.2, name='bidir_rnn_2'))(bn_rnn_1)
    bn_rnn_2 = BatchNormalization(name='bn_rnn_2')(bidir_rnn_2)
    
    # layer-3
    bidir_rnn_3 = Bidirectional(GRU(units, return_sequences=True, dropout=0.2, name='bidir_rnn_3'))(bn_rnn_2)
    bn_rnn_3 = BatchNormalization(name='bn_rnn_3')(bidir_rnn_3)
    
    # layer-4
    bidir_rnn_4 = Bidirectional(GRU(units, return_sequences=True, dropout=0.2, name='bidir_rnn_4'))(bn_rnn_3)
    bn_rnn_4 = BatchNormalization(name='bn_rnn_4')(bidir_rnn_4)
    
    # layer-5
    bidir_rnn_5 = Bidirectional(GRU(units, return_sequences=True, dropout=0.2, name='bidir_rnn_5'))(bn_rnn_4)
    bn_rnn_5 = BatchNormalization(name='bn_rnn_5')(bidir_rnn_5)
    
    # Dense layer
    time_dense = TimeDistributed(Dense(output_dim, name='time_dense'))(bn_rnn_5)
    
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)

    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model
```

### Example Prediction

After training the model on 20 epochs, `validation loss` falls from `257.4721` to `114.3742`.

```text
True transcription:
       
her father is a most remarkable person to say the least
-------------------------------------------------------------------
Predicted transcription: 

er fother s am mus ferm ma apo persind to sa the lacet
```

_Image credits: [Udacity](www.udacity.com)_