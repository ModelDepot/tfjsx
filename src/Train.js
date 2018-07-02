import React from 'react';
import * as tf from '@tensorflow/tfjs';
import Plot from 'react-plotly.js';

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
      const values = this.state.metrics[metric];
      return (
        <Plot
          key={metric}
          data={[
            {
              x: Object.keys(values).map(i => parseInt(i, 10)),
              y: values,
              type: 'scatter',
              mode: 'lines+markers',
              marker: { color: '#1a9afc' },
            },
          ]}
          layout={{ width: 420, height: 340, title: metric }}
        />
      );
    });

    return (
      <div>
        <h1 style={{
          fontFamily: '"HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif',
          fontWeight: 300,
        }}>
          {'<'}Train{' />'}
        </h1>
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
    const {
      trainData,
      samples,
      validationData,
      epochs,
      batchSize,
      display,
    } = this.props;


    // TODO: Switch to PropTypes
    const onBatchEnd = typeof this.props.onBatchEnd === 'function' ? this.props.onBatchEnd : () => { };

    for (let epoch = 0; epoch < epochs; epoch++) {
      const trainGenerator = trainData();
      for (let batch = 0; batch * batchSize < samples; batch++) {
        // Pause training when train prop is false
        const train = this.props.train;
        if (!train) {
          yield;
        }

        const trainBatch = this._getBatch(trainGenerator, batchSize);
        const history = await model.fit(trainBatch.xs, trainBatch.ys, { batchSize: trainBatch.xs.shape[0], epochs: 1 });
        onBatchEnd(history.history, model);
        tf.dispose(trainBatch);

        if (display) {
          const fitMetrics = history.history;
          this._pushMetrics(fitMetrics);
          await tf.nextFrame();
        }
      }

      if (validationData) {
        const valGenerator = validationData();
        // Just get all the validation data at once
        const valBatch = this._getBatch(valGenerator, Infinity);
        const valMetrics = model.evaluate(valBatch.xs, valBatch.ys, { batchSize });
        const history = {};

        for (let i = 0; i < valMetrics.length; i++) {
          const metric = model.metricsNames[i];
          history[`validation-${metric}`] = await valMetrics[i].data();
        }

        this._pushMetrics(history);

        tf.dispose(valMetrics);
        tf.dispose(valBatch);
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
    };
  }

  _pushMetrics(metrics) {
    const updatedMetrics = { ...this.state.metrics };
    Object.keys(metrics).forEach(metric => {
      if (updatedMetrics[metric] == null) {
        updatedMetrics[metric] = [];
      }
      updatedMetrics[metric].push(metrics[metric][0]);
    });
    this.setState({
      metrics: updatedMetrics,
    });
  }
}
