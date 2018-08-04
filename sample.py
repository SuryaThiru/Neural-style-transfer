from neural_transer import style_transfer

cnt_image = 'img/neha.jpg'
style_image = 'img/monalisa.jpg'
output = 'output/neha/'

style_transfer(cnt_image, style_image, output)
