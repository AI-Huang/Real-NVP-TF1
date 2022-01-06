# Real-NVP-TF1

An implementation of the Real NVP model in TensorFlow 1.x

## Requirements and Environment Setup

- Tensorflow (tested with v1.15.0)
- tensorflow_probability (tested with v0.8)

To setup a standalone environment step by step, with `tensorflow_probability` installed:

```bash
conda env create -n tf1.15
conda activate tf1.15
conda install python==3.7
pip install tensorflow-gpu==1.15
python -m pip install --upgrade --user "tensorflow<2" "tensorflow_probability<0.9"
```

See [2] for more details.

## Still in Construction

Some modules are to be implemented:

- [x] TF1 weight norm feature
- [x] TF1 coupling layer
- [ ] TF1 squeeze2x2
- [x] TF1 real nvp loss
- [ ] TF1 real nvp

## Usage

### Train

```bash
python main.py
```

### Generatation

## References

Thanks to [1]'s codes

- [1] https://github.com/chrischute/real-nvp
- [2] https://github.com/tensorflow/probability
- [3] Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. "Density estimation using real nvp." arXiv preprint arXiv:1605.08803 (2016).
