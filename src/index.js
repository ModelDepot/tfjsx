import React from 'react';
import * as tf from '@tensorflow/tfjs';

export { Train } from './Train';

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
    const {
      children,
      optimizer,
      loss,
      metrics,
      onCompile,
    } = this.props;

    const layerElements = React.Children.toArray(children);

    const model = tf.sequential();

    layerElements.forEach(layerElement => {
      model.add(parseLayerElement(layerElement));
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

// Layer Types
export function Conv2D() { return null; }
export function Dense() { return null; }
export function Flatten() { return null; }
export function MaxPooling2D() { return null; }
