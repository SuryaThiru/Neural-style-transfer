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

Few images are given in the `/img` folder. U can use your own images too.

## Requirements
* Keras
* Scipy
* Numpy
* Pillow

## References:
* [Gatys, L.A., Ecker, A.S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style. CoRR, abs/1508.06576.](https://arxiv.org/abs/1508.06576)
* [Jing, Y., Yang, Y., Feng, Z., Ye, J., & Song, M. (2017). Neural Style Transfer: A Review. CoRR, abs/1705.04058.](https://arxiv.org/abs/1705.04058v6)
* https://medium.com/mlreview/making-ai-art-with-style-transfer-using-keras-8bb5fa44b216
* https://www.bonaccorso.eu/2016/11/13/neural-artistic-style-transfer-experiments-with-keras
