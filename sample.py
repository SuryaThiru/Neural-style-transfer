from neural_transer import style_transfer

cnt_image = 'img/neckarfront.jpg'
style_image = 'img/starry_night.jpg'
output = 'test/'

style_transfer(cnt_image, style_image, output, 500, 40)
