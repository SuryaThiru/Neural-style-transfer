from neural_transer import style_transfer

cnt_image = 'img/neckarfront.jpg'
style_image = 'img/starry_night.jpg'
output = 'test/'
epochs = 600
save_per_epoch = 40

style_transfer(cnt_image, style_image, output, epochs, save_per_epoch, random_canvas=True)
