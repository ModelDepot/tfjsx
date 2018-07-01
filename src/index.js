import React from 'react';
import * as tf from '@tensorflow/tfjs';

function parseLayerElement(element) {
  switch (element.type) {
    case Conv2D:
      return tf.layers.conv2d(element.props);
    case Dense:
      return tf.layers.dense(element.props);
    case Flatten:
      return tf.layers.flatten(element.props);
    case MaxPooling2D:
      return tf.layers.maxPooling2d(element.props);
    default:
      throw new Error('Invalid Layer', element);
  }
}

export class Model extends React.Component {
  render() {
    return null;
  }

  _compile() {
    console.log('compiling...')

    const {
      children,
      optimizer,
      loss,
      metrics,
      onCompile,
    } = this.props;

    const layerElements = React.Children.toArray(children);

    const model = tf.sequential();

    console.log(layerElements);

    layerElements.forEach(layerElement => {
      model.add(parseLayerElement(layerElement))
    });

    model.compile({
      optimizer,
      loss,
      metrics,
    });

    if (typeof onCompile === 'function') {
      onCompile(model);
    }
  }

  componentDidUpdate(prevProps) {
    if (this.props != prevProps)
      this._compile();
  }

  componentDidMount() {
    this._compile();
  }
}

export class Train extends React.Component {
  state = {
    metrics: {},
    modelElement: null,
  };
  trainer = null;

  // Initialize the child model and create trainer after model is compiled.
  componentDidMount() {
    const modelElement = React.Children.only(this.props.children);

    const wrappedModelElement = React.cloneElement(
      modelElement, {
        onCompile: model => {
          this.trainer = this._train(model);
          this.trainer.next();
        },
      });

    this.setState({
      modelElement: wrappedModelElement,
    });
  }

  render() {
    if (!this.props.display) {
      return this.state.modelElement;
    }

    const metricElems = Object.keys(this.state.metrics).map(metric => {
      return (
        <h3 key={metric}>
          {metric}: {this.state.metrics[metric]}
        </h3>
      );
    });

    return (
      <div>
        <h1>Train</h1>
        {metricElems}
        {this.state.modelElement}
      </div>
    );
  }

  shouldComponentUpdate(nextProps, nextState) {
    return (
      this.props.train != nextProps.train ||
      this.props.display != nextProps.display ||
      this.state.modelElement != nextState.modelElement ||
      this.state.metrics != nextState.metrics
    );
  }

  componentDidUpdate(prevProps) {
    // Resume training if train state was changed to true
    if (this.props.train && !prevProps.train && this.trainer != null) {
      this.trainer.next();
    }
  }

  async * _train(model) {
    if (!this.props.train) {
      yield;
    }

    const epochs = this.props.epochs;
    const batchSize = this.props.batchSize;
    const samples = this.props.samples;

    const onBatchEnd = typeof this.props.onBatchEnd === 'function' ? this.props.onBatchEnd : () => { };

    for (let epoch = 0; epoch < epochs; epoch++) {
      const trainGenerator = this.props.trainData();
      for (let batch = 0; batch * batchSize < samples; batch++) {
        // Pause training when train prop is false
        if (!this.props.train) {
          yield;
        }

        const trainBatch = this._getBatch(trainGenerator, batchSize);
        const history = await model.fit(trainBatch.xs, trainBatch.ys, { batchSize: trainBatch.xs.shape[0], epochs: 1 });
        this.setState({metrics: history.history});
        onBatchEnd(history.history);
        await tf.nextFrame();
      }
    }

    this.props.onTrainEnd(model);
  }

  _getBatch(generator, batchSize = 32) {
    const xs = [];
    const ys = [];

    for (let i = 0; i < batchSize; i++) {
      const sample = generator.next().value;

      if (sample == null) {
        break;
      }

      xs.push(sample.x);
      ys.push(sample.y);
    }

    if (xs.length === 0) {
      throw new Error('No data returned from data generator for batch, check sample length');
    }

    // Either stack if it's a generator of tensors, or convert to tensor if
    // it's a generator of JS arrs
    const stack = arr => arr[0] instanceof tf.Tensor ?
      tf.stack(arr) : tf.tensor(arr);

    return {
      xs: stack(xs),
      ys: stack(ys),
    }
  }
}

// Layer Types
export function Conv2D() { return null; }
export function Dense() { return null; }
export function Flatten() { return null; }
export function MaxPooling2D() { return null; }
