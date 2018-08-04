from neural_transer import style_transfer

cnt_image = 'img/river.jpg'
style_image = 'img/starry_night.jpg'
output = 'output/'

style_transfer(cnt_image, style_image, output)
