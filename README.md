# Tensorflow.jsx

Have you ever woken up one day and thought: My life would be so much better
if I could define and train my machine learning models in React. I haven't.

Define, train and visualize training of your ML models in your
favorite front-end library ([React](https://reactjs.org/)) backed by your second
favorite front-end library ([Tensorflow.js](https://js.tensorflow.org/)).

## Features:
- Define models in React/JSX
- Stream training data via [ES6 Generators](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/function*)
- Turn on training progress visualization with a single flag
- Pause training

# Demo App

Check out/clone the [demo app](openanissueifthisisntalink) to get started quicker.

# Getting Started

## Installation

```
yarn add react @tensorflow/tfjs tfjsx
```

OR

```
npm install react @tensorflow/tfjs tfjsx
```

## Write Some Machine Learning

```javascript
import React from 'react';
import ReactDOM from 'react-dom';
import { Train, Model, Dense } from 'tfjsx';

// Define a generator of train data
function* trainDataGenerator() {
  yield { x: 1, y: 1 };
  yield { x: 4, y: 4 };
  yield { x: 8, y: 8 };
}

function MyTrainedModel() {
  // Train the model with the training data generator defined above
  return (
    <Train
      trainData={trainDataGenerator}
      epochs={15}
      batchSize={3}
      samples={3}
      onTrainEnd={model => model.describe()}
      train
      display
    >
      {/* Define the model architecture */}
      <Model optimizer='sgd' loss='meanSquaredError'>
        <Dense units={1} inputShape={[1]} />
      </Model>
    </Train>
  );
}

ReactDOM.render(<MyTrainedModel />, document.getElementById('app'));
```

# API

## Train

Property Name | Type | Description
---|---|---
trainData | function* () | The generator should yield an object with `x` and `y` properties corresponding to training data and label.
validationData | function* () | Same as `trainData`, but should generate validation data. Will be used to output validation metrics during training.
epochs | Number | Number of epochs to train the model for
batchSize | Number | The number of samples to include in each training batch
samples | Number | Number of expected samples the generator will be able to generate.
onTrainEnd | function(tf.Model) | Called after the model is done training, the trained model is passed into the callback
onBatchEnd | function(Object metrics, tf.Model) | Called after each batch is done training, an object with that batch's training metrics along with the current model is passed into the callback.
train | Bool | Turn on and off training
display | Bool | Enable or disable graphing of training status

## Model

All valid config properties passed into `model.compile` are valid here.
[See config.](https://js.tensorflow.org/api/0.11.7/#tf.Model.compile)

## Layers

Similar to `Model`, all valid layers have their props passed through as
properties of the `config` object in Tensorflow.js.
The following layers are currently available:

- [Conv2D](https://js.tensorflow.org/api/0.11.7/#layers.conv2d)
- [Dense](https://js.tensorflow.org/api/0.11.7/#layers.dense)
- [Flatten](https://js.tensorflow.org/api/0.11.7/#layers.flatten)
- [MaxPooling2D](https://js.tensorflow.org/api/0.11.7/#layers.maxPooling2d)

Adding new layer types is simple, PRs are always welcome :)

# Future Roadmap/Wishlist

- Model summarization (adding `display` flag to `Model`)
- Layer activation visualization (adding `display` flag to any layer)
- Model evaluation visualizations
- Allow pre-trained models to be used as a layer
