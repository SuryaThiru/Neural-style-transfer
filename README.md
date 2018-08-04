# Neural style transfer
Implementation of "A neural algorithm of artistic style" in keras

## Usage

```python
from neural_transer import style_transfer

cnt_image = 'img/river.jpg'
style_image = 'img/starry_night.jpg'
output = 'output/'

style_transfer(cnt_image, style_image, output)
```

## References:
* [arXiv:1508.06576v2](https://arxiv.org/abs/1508.06576)
* [arXiv:1705.04058v6](https://arxiv.org/abs/1705.04058v6)
* https://medium.com/mlreview/making-ai-art-with-style-transfer-using-keras-8bb5fa44b216
* https://www.bonaccorso.eu/2016/11/13/neural-artistic-style-transfer-experiments-with-keras
